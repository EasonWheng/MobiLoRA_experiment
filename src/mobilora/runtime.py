from __future__ import annotations

import importlib.util
import json
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mobilora.assets import prepare_hf_assets
from mobilora.types import AdapterSpec, AppConfig, PrepareManifest, RequestRecord, RuntimeArtifact
from mobilora.utils import stable_hash


@dataclass(slots=True)
class DependencyCheck:
    name: str
    available: bool
    detail: str


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_report() -> list[DependencyCheck]:
    modules = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "datasets",
        "yaml",
        "matplotlib",
        "huggingface_hub",
        "numpy",
    )
    return [
        DependencyCheck(
            name=module,
            available=_module_available(module),
            detail="ok" if _module_available(module) else "missing",
        )
        for module in modules
    ]


class BaseRuntime:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def artifact_from_request(self, request: RequestRecord, max_input: int) -> RuntimeArtifact:
        raise NotImplementedError

    def warmup(self, requests: list[RequestRecord], max_input: int) -> None:
        return None

    def validation_record(self) -> dict[str, object]:
        return {"backend": self.__class__.__name__, "status": "ok"}

    def response_with_compression(self, base_response: str, avg_error_bound: float) -> str:
        if avg_error_bound <= 0.0:
            return base_response
        tokens = base_response.split()
        if len(tokens) < 8:
            return base_response

        mutate_positions: list[int] = []
        if avg_error_bound >= 1.0e-2 and len(tokens) >= 12:
            mutate_positions.append(len(tokens) - 2)
        elif avg_error_bound >= 1.0e-3 and len(tokens) >= 24:
            mutate_positions.append(len(tokens) - 3)

        if not mutate_positions:
            return base_response

        mutated = list(tokens)
        for position in mutate_positions:
            if 0 <= position < len(mutated):
                mutated[position] = "compressed"
        return " ".join(mutated)


class MockRuntime(BaseRuntime):
    def artifact_from_request(self, request: RequestRecord, max_input: int) -> RuntimeArtifact:
        prompt_tokens = request.prompt.split()
        token_ids = tuple((stable_hash(token) % 10_007) for token in prompt_tokens[:max_input])
        if not token_ids:
            token_ids = (0,)

        prompt_seed = stable_hash(request.prompt) % 10_000
        adapter_seed = stable_hash(request.adapter_alias) % 10_000
        rng = random.Random(prompt_seed * 31 + adapter_seed)

        layer_count = self.config.runtime.layer_count
        hidden_size = self.config.runtime.hidden_size
        vector_width = min(16, hidden_size)

        layer_vectors = []
        for layer_index in range(layer_count):
            layer_base = (prompt_seed * (layer_index + 1) + adapter_seed * 0.15) / 10_000.0
            vector = []
            for position in range(vector_width):
                noise = rng.uniform(-0.025, 0.025)
                token_factor = token_ids[position % len(token_ids)] / 10_007.0
                vector.append(layer_base + token_factor + noise)
            layer_vectors.append(tuple(vector))

        shallow_key = layer_vectors[0]

        token_count = len(token_ids)
        raw_size_mb = (
            token_count
            * layer_count
            * hidden_size
            * 2
            * 2
            / (1024.0 * 1024.0)
        )
        response = self._build_response(request)
        metadata = {
            "backend": "mock",
            "timing_source": "synthetic_model",
            "token_count": token_count,
            "generated_tokens": len(response.split()),
            "raw_kv_mb": round(raw_size_mb, 6),
        }
        return RuntimeArtifact(
            token_ids=token_ids,
            shallow_key=tuple(shallow_key),
            layer_vectors=tuple(layer_vectors),
            raw_size_mb=raw_size_mb,
            response_text=response,
            reference_text=request.reference or response,
            metadata=metadata,
        )

    def _build_response(self, request: RequestRecord) -> str:
        words = request.prompt.split()
        focus = " ".join(words[: min(len(words), 18)])
        if request.workload == "writing":
            return f"[{request.adapter_alias}] Summary: {focus}"
        return f"[{request.adapter_alias}] Reply: {focus}"


class HuggingFaceRuntime(BaseRuntime):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._adapter_paths: dict[str, Path] = {}
        self._base_model_path: Path | None = None
        self._primary_device = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "The Hugging Face backend needs torch + transformers + bitsandbytes. "
                "Bootstrap the WSL environment first."
            ) from exc

        asset_bundle = prepare_hf_assets(self.config, download_assets=True)
        base_record = asset_bundle["base_model"]
        adapter_records = asset_bundle["adapters"]
        self._base_model_path = Path(str(base_record.get("resolved_path") or base_record["local_path"]))
        self._adapter_paths = {
            str(record["alias"]): Path(str(record.get("resolved_path") or record["local_path"]))
            for record in adapter_records
        }

        quantization_config = None
        if self.config.runtime.load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._base_model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(self._tokenizer, "eos_token", None):
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._base_model_path),
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        if not self._adapter_paths:
            raise RuntimeError("No LoRA adapters are available for the Hugging Face backend.")

        first_alias = self.config.adapter_specs[0].alias
        first_adapter_path = self._adapter_paths[first_alias]
        self._model = PeftModel.from_pretrained(
            self._model,
            str(first_adapter_path),
            adapter_name=first_alias,
            is_trainable=False,
        )
        for adapter in self.config.adapter_specs[1:]:
            adapter_path = self._adapter_paths[adapter.alias]
            self._model.load_adapter(str(adapter_path), adapter_name=adapter.alias, is_trainable=False)

        self._model.eval()
        self._torch = torch
        self._primary_device = getattr(self._model, "device", None)
        if self._primary_device is None:
            self._primary_device = next(self._model.parameters()).device
        self._loaded = True

    def artifact_from_request(self, request: RequestRecord, max_input: int) -> RuntimeArtifact:
        self.load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        if request.adapter_alias not in self._adapter_paths:
            raise KeyError(f"Adapter '{request.adapter_alias}' is not loaded.")

        adapter_switch_started_at = time.perf_counter()
        self._model.set_adapter(request.adapter_alias)
        adapter_switch_ms = (time.perf_counter() - adapter_switch_started_at) * 1000.0

        tokenize_started_at = time.perf_counter()
        tokenized = self._tokenizer(
            request.prompt,
            truncation=True,
            max_length=max_input,
            return_tensors="pt",
        )
        tokenize_ms = (time.perf_counter() - tokenize_started_at) * 1000.0
        tokenized = {key: value.to(self._primary_device) for key, value in tokenized.items()}
        token_ids = tuple(int(item) for item in tokenized["input_ids"][0].tolist())
        if not token_ids:
            token_ids = (0,)

        self._reset_peak_memory_stats()
        self._synchronize_device()
        prefill_started_at = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self._model(
                **tokenized,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            self._synchronize_device()
            prefill_ms = (time.perf_counter() - prefill_started_at) * 1000.0
            response_token_ids, decode_ms = self._greedy_decode(
                attention_mask=tokenized.get("attention_mask"),
                outputs=outputs,
                max_new_tokens=min(32, self.config.runtime.max_new_tokens),
            )

        hidden_states = tuple(outputs.hidden_states or ())
        layer_vectors = self._extract_layer_vectors(hidden_states)
        shallow_key = (
            layer_vectors[0]
            if layer_vectors
            else tuple(float(token_ids[index % len(token_ids)]) for index in range(16))
        )
        response = self._tokenizer.decode(response_token_ids, skip_special_tokens=True).strip()
        if not response:
            response = self._tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True).strip()
        raw_size_mb = self._estimate_kv_size_mb(getattr(outputs, "past_key_values", None))
        total_measured_ms = tokenize_ms + adapter_switch_ms + prefill_ms + decode_ms
        memory_stats = self._cuda_memory_snapshot()

        metadata = {
            "backend": "hf",
            "timing_source": "measured_runtime",
            "adapter_alias": request.adapter_alias,
            "base_model_path": str(self._base_model_path) if self._base_model_path else "",
            "adapter_path": str(self._adapter_paths[request.adapter_alias]),
            "token_count": len(token_ids),
            "hidden_state_layers": max(len(hidden_states) - 1, 0),
            "generated_tokens": len(response_token_ids),
            "raw_kv_mb": round(raw_size_mb, 6),
            "tokenize_ms": round(tokenize_ms, 6),
            "prefill_ms": round(prefill_ms, 6),
            "decode_ms": round(decode_ms, 6),
            "adapter_switch_ms": round(adapter_switch_ms, 6),
            "measured_end_to_end_ms": round(total_measured_ms, 6),
            "prefill_tokens_per_second": round(self._tokens_per_second(len(token_ids), prefill_ms), 6),
            "decode_tokens_per_second": round(
                self._tokens_per_second(max(len(response_token_ids) - 1, 0), decode_ms),
                6,
            ),
            "cuda_allocated_mb": round(memory_stats["allocated_mb"], 6),
            "cuda_reserved_mb": round(memory_stats["reserved_mb"], 6),
            "cuda_peak_allocated_mb": round(memory_stats["peak_allocated_mb"], 6),
        }
        return RuntimeArtifact(
            token_ids=token_ids,
            shallow_key=tuple(shallow_key),
            layer_vectors=tuple(layer_vectors),
            raw_size_mb=raw_size_mb,
            response_text=response,
            reference_text=request.reference or response,
            metadata=metadata,
        )

    def warmup(self, requests: list[RequestRecord], max_input: int) -> None:
        self.load()
        warmed_adapters: set[str] = set()
        for request in requests:
            if request.adapter_alias in warmed_adapters:
                continue
            self.artifact_from_request(request, max_input)
            warmed_adapters.add(request.adapter_alias)
            if len(warmed_adapters) >= len({item.adapter_alias for item in requests}):
                break

    def validation_record(self) -> dict[str, object]:
        self.load()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        model_config = getattr(self._model, "config", None)
        return {
            "backend": "hf",
            "status": "ok",
            "device": str(self._primary_device),
            "base_model_path": str(self._base_model_path) if self._base_model_path else "",
            "adapter_count": len(self._adapter_paths),
            "hidden_size": int(getattr(model_config, "hidden_size", self.config.runtime.hidden_size)),
            "layer_count": int(getattr(model_config, "num_hidden_layers", self.config.runtime.layer_count)),
            "cuda_available": bool(self._torch.cuda.is_available()),
        }

    def _extract_layer_vectors(self, hidden_states: tuple[Any, ...]) -> list[tuple[float, ...]]:
        if not hidden_states:
            return []
        vectors: list[tuple[float, ...]] = []
        for layer_hidden in hidden_states[1:]:
            last_token = layer_hidden[0, -1, : min(16, layer_hidden.shape[-1])].detach().float().cpu().tolist()
            vectors.append(tuple(float(item) for item in last_token))
        return vectors

    def _estimate_kv_size_mb(self, past_key_values: Any) -> float:
        kv_tensors: list[Any] = []
        cache = past_key_values
        if cache is None:
            return (
                len(self.config.adapter_specs)
                * self.config.runtime.layer_count
                * self.config.runtime.hidden_size
                * 2
                * 2
                / (1024.0 * 1024.0)
            )
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for key_tensor, value_tensor in zip(cache.key_cache, cache.value_cache):
                kv_tensors.extend([key_tensor, value_tensor])
        else:
            for layer_cache in cache or ():
                if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                    kv_tensors.extend([layer_cache[0], layer_cache[1]])

        total_bytes = 0
        for tensor in kv_tensors:
            total_bytes += int(tensor.numel() * tensor.element_size())
        return total_bytes / (1024.0 * 1024.0)

    def _greedy_decode(
        self,
        attention_mask: Any,
        outputs: Any,
        max_new_tokens: int,
    ) -> tuple[list[int], float]:
        assert self._model is not None and self._torch is not None
        if max_new_tokens <= 0:
            return [], 0.0

        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        generated_token_ids: list[int] = []
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = getattr(outputs, "past_key_values", None)
        running_attention_mask = attention_mask

        self._synchronize_device()
        decode_started_at = time.perf_counter()
        for step_index in range(max_new_tokens):
            token_value = int(next_token[0, 0].item())
            if eos_token_id is not None and token_value == int(eos_token_id):
                break
            generated_token_ids.append(token_value)
            if step_index == max_new_tokens - 1:
                break

            if running_attention_mask is not None:
                running_attention_mask = self._torch.cat(
                    [
                        running_attention_mask,
                        running_attention_mask.new_ones((running_attention_mask.shape[0], 1)),
                    ],
                    dim=1,
                )

            step_outputs = self._model(
                input_ids=next_token,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = getattr(step_outputs, "past_key_values", None)
            next_token = step_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self._synchronize_device()
        decode_ms = (time.perf_counter() - decode_started_at) * 1000.0
        return generated_token_ids, decode_ms

    def _device_is_cuda(self) -> bool:
        return self._primary_device is not None and str(self._primary_device).startswith("cuda")

    def _synchronize_device(self) -> None:
        if self._torch is None or not self._device_is_cuda() or not self._torch.cuda.is_available():
            return
        self._torch.cuda.synchronize(device=self._primary_device)

    def _reset_peak_memory_stats(self) -> None:
        if self._torch is None or not self._device_is_cuda() or not self._torch.cuda.is_available():
            return
        self._torch.cuda.reset_peak_memory_stats(device=self._primary_device)

    def _cuda_memory_snapshot(self) -> dict[str, float]:
        if self._torch is None or not self._device_is_cuda() or not self._torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
        return {
            "allocated_mb": self._torch.cuda.memory_allocated(device=self._primary_device) / (1024.0 * 1024.0),
            "reserved_mb": self._torch.cuda.memory_reserved(device=self._primary_device) / (1024.0 * 1024.0),
            "peak_allocated_mb": self._torch.cuda.max_memory_allocated(device=self._primary_device)
            / (1024.0 * 1024.0),
        }

    def _tokens_per_second(self, token_count: int, elapsed_ms: float) -> float:
        if token_count <= 0 or elapsed_ms <= 0.0:
            return 0.0
        return token_count / (elapsed_ms / 1000.0)


def build_runtime(config: AppConfig, backend: str | None = None) -> BaseRuntime:
    selected = (backend or config.runtime.backend).lower()
    if selected == "hf":
        return HuggingFaceRuntime(config)
    return MockRuntime(config)


def check_adapter_repo(repo: AdapterSpec) -> dict[str, object]:
    url = f"https://huggingface.co/{repo.repo_id}"
    started_at = time.perf_counter()
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
    except Exception as exc:
        status = 0
        return {
            "repo_id": repo.repo_id,
            "alias": repo.alias,
            "url": url,
            "status": "error",
            "detail": str(exc),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
    return {
        "repo_id": repo.repo_id,
        "alias": repo.alias,
        "url": url,
        "status": "ok" if status == 200 else "unexpected",
        "detail": f"http_{status}",
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }


def build_prepare_manifest(
    config: AppConfig,
    backend: str,
    skip_hf_check: bool,
    download_assets: bool = False,
) -> PrepareManifest:
    manifest = PrepareManifest(backend=backend, base_model=config.runtime.base_model)
    manifest.directories.extend(
        str(item)
        for item in (
            config.paths.asset_root,
            config.paths.hf_cache,
            config.paths.datasets_cache,
            config.paths.adapters_cache,
            config.paths.bench_outputs,
        )
    )

    manifest.dependency_checks.extend(
        {
            "name": item.name,
            "available": item.available,
            "detail": item.detail,
        }
        for item in dependency_report()
    )

    if skip_hf_check:
        manifest.notes.append("Skipped adapter reachability checks.")
    else:
        manifest.adapter_checks.extend(check_adapter_repo(item) for item in config.adapter_specs)

    if backend.lower() == "hf":
        asset_bundle = prepare_hf_assets(config, download_assets=download_assets)
        manifest.base_model_check = asset_bundle["base_model"]
        manifest.asset_checks.extend(asset_bundle["adapters"])
        manifest.notes.append(
            "HF backend can now validate remote metadata, download local assets on demand, and load LoRA adapters."
        )
        if download_assets:
            manifest.notes.append("HF assets were downloaded into the configured D-drive cache directories.")
    else:
        manifest.notes.append(
            "Mock backend exercises the full MobiLoRA control plane without requiring the base model to fit."
        )
    return manifest


def build_runtime_validation(config: AppConfig, backend: str) -> dict[str, object]:
    runtime = build_runtime(config, backend)
    return runtime.validation_record()


def manifest_to_json(manifest: PrepareManifest) -> str:
    return json.dumps(
        {
            "backend": manifest.backend,
            "base_model": manifest.base_model,
            "base_model_check": manifest.base_model_check,
            "adapter_checks": manifest.adapter_checks,
            "asset_checks": manifest.asset_checks,
            "dependency_checks": manifest.dependency_checks,
            "runtime_validation": manifest.runtime_validation,
            "directories": manifest.directories,
            "notes": manifest.notes,
        },
        ensure_ascii=False,
        indent=2,
    )

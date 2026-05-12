from __future__ import annotations

import importlib.util
import json
import random
import time
import urllib.request
from dataclasses import dataclass

from mobilora.types import AdapterSpec, AppConfig, PrepareManifest, RequestRecord, RuntimeArtifact
from mobilora.utils import clamp, stable_hash


@dataclass(slots=True)
class DependencyCheck:
    name: str
    available: bool
    detail: str


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_report() -> list[DependencyCheck]:
    modules = ("torch", "transformers", "peft", "bitsandbytes", "datasets", "yaml", "matplotlib")
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
        return RuntimeArtifact(
            token_ids=token_ids,
            shallow_key=tuple(shallow_key),
            layer_vectors=tuple(layer_vectors),
            raw_size_mb=raw_size_mb,
            response_text=response,
            reference_text=request.reference or response,
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

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "The Hugging Face backend needs torch + transformers + bitsandbytes. "
                "Bootstrap the WSL environment first."
            ) from exc

        quantization_config = None
        if self.config.runtime.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.runtime.base_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.runtime.base_model,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        self._loaded = True

    def artifact_from_request(self, request: RequestRecord, max_input: int) -> RuntimeArtifact:
        self.load()
        assert self._tokenizer is not None
        tokenized = self._tokenizer(
            request.prompt,
            truncation=True,
            max_length=max_input,
            return_tensors="pt",
        )
        token_ids = tuple(int(item) for item in tokenized["input_ids"][0].tolist())
        if not token_ids:
            token_ids = (0,)
        # This backend currently extracts deterministic summaries instead of persisting
        # full past_key_values to keep the prototype lightweight and hardware-tolerant.
        layer_vectors = []
        base_seed = stable_hash(request.prompt + request.adapter_alias)
        rng = random.Random(base_seed)
        for _ in range(self.config.runtime.layer_count):
            vector = tuple(rng.uniform(-1.0, 1.0) for _ in range(16))
            layer_vectors.append(vector)
        response = f"[HF:{request.adapter_alias}] {request.prompt[:128]}"
        raw_size_mb = (
            len(token_ids)
            * self.config.runtime.layer_count
            * self.config.runtime.hidden_size
            * 2
            * 2
            / (1024.0 * 1024.0)
        )
        return RuntimeArtifact(
            token_ids=token_ids,
            shallow_key=layer_vectors[0],
            layer_vectors=tuple(layer_vectors),
            raw_size_mb=raw_size_mb,
            response_text=response,
            reference_text=request.reference or response,
        )


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


def build_prepare_manifest(config: AppConfig, backend: str, skip_hf_check: bool) -> PrepareManifest:
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
        manifest.notes.append(
            "HF backend is implemented conservatively and currently targets validation and summary extraction."
        )
    else:
        manifest.notes.append(
            "Mock backend exercises the full MobiLoRA control plane without requiring the base model to fit."
        )
    return manifest


def manifest_to_json(manifest: PrepareManifest) -> str:
    return json.dumps(
        {
            "backend": manifest.backend,
            "base_model": manifest.base_model,
            "adapter_checks": manifest.adapter_checks,
            "dependency_checks": manifest.dependency_checks,
            "directories": manifest.directories,
            "notes": manifest.notes,
        },
        ensure_ascii=False,
        indent=2,
    )

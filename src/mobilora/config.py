from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mobilora.types import (
    AppConfig,
    BenchConfig,
    DashboardConfig,
    DeltaConfig,
    EvictionConfig,
    PaperConfig,
    PathConfig,
    RuntimeConfig,
    ScenarioConfig,
    SGLangConfig,
    adapter_specs_from_repo_ids,
)
from mobilora.utils import coerce_path, default_asset_root, ensure_directory, resolve_asset_subdir


def _scenario_from_dict(data: dict[str, Any]) -> ScenarioConfig:
    return ScenarioConfig(
        name=str(data["name"]),
        adapter_count=int(data["adapter_count"]),
        cache_budget_mb=int(data["cache_budget_mb"]),
        max_input=int(data["max_input"]),
        request_count=int(data.get("request_count", 300)),
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.resolve().parent.parent

    asset_root = coerce_path(
        data.get("paths", {}).get("asset_root") or default_asset_root(),
        base_dir=repo_root,
    )
    paths = PathConfig(
        asset_root=asset_root,
        hf_cache=resolve_asset_subdir(asset_root, data["paths"]["hf_cache_dir"]),
        datasets_cache=resolve_asset_subdir(asset_root, data["paths"]["datasets_dir"]),
        adapters_cache=resolve_asset_subdir(asset_root, data["paths"]["adapters_dir"]),
        bench_outputs=resolve_asset_subdir(asset_root, data["paths"]["bench_outputs_dir"]),
    )

    runtime_data = data["runtime"]
    runtime = RuntimeConfig(
        backend=str(runtime_data.get("backend", "mock")),
        base_model=str(runtime_data["base_model"]),
        quantization_bits=int(runtime_data["quantization_bits"]),
        load_in_4bit=bool(runtime_data.get("load_in_4bit", True)),
        device=str(runtime_data.get("device", "cuda")),
        hidden_size=int(runtime_data.get("hidden_size", 128)),
        layer_count=int(runtime_data.get("layer_count", 8)),
        max_new_tokens=int(runtime_data.get("max_new_tokens", 96)),
    )

    delta_data = data["delta"]
    delta = DeltaConfig(
        high_similarity_threshold=float(delta_data["high_similarity_threshold"]),
        medium_similarity_threshold=float(delta_data["medium_similarity_threshold"]),
        high_similarity_error_bound=float(delta_data["high_similarity_error_bound"]),
        medium_similarity_error_bound=float(delta_data["medium_similarity_error_bound"]),
        low_similarity_error_bound=float(delta_data["low_similarity_error_bound"]),
    )

    eviction_data = data["eviction"]
    eviction = EvictionConfig(
        foreground_score=float(eviction_data["foreground_score"]),
        background_score=float(eviction_data["background_score"]),
        killed_score=float(eviction_data["killed_score"]),
        lambda_state=float(eviction_data["lambda_state"]),
        lambda_recency=float(eviction_data["lambda_recency"]),
        lambda_length=float(eviction_data["lambda_length"]),
        recent_window=int(eviction_data.get("recent_window", 20)),
    )

    bench_data = data["bench"]
    bench = BenchConfig(
        random_seed=int(bench_data["random_seed"]),
        workloads=tuple(str(item) for item in bench_data["workloads"]),
        dataset_files={
            str(name): coerce_path(path, base_dir=repo_root)
            for name, path in bench_data["dataset_files"].items()
        },
        scenarios=tuple(_scenario_from_dict(item) for item in bench_data["scenarios"]),
    )

    sglang_data = data.get("sglang", {})
    sglang = SGLangConfig(
        wsl_distro=str(sglang_data.get("wsl_distro", "Ubuntu-24.04")),
        host=str(sglang_data.get("host", "127.0.0.1")),
        stock_port=int(sglang_data.get("stock_port", 30100)),
        mobilora_port=int(sglang_data.get("mobilora_port", 30101)),
        conda_root=str(sglang_data.get("conda_root", "/home/tomat/miniconda3")),
        conda_env_name=str(
            sglang_data.get("conda_env_name", "nn-lesson-SEU-sglang-cu128")
        ),
        python_executable=str(
            sglang_data.get(
                "python_executable",
                "/home/tomat/miniconda3/envs/nn-lesson-SEU-sglang-cu128/bin/python",
            )
        ),
        source_root=coerce_path(
            str(sglang_data.get("source_root", "third_party/sglang/python")),
            base_dir=repo_root,
        ),
        source_commit=str(sglang_data.get("source_commit", "")),
        stable_release_tag=str(sglang_data.get("stable_release_tag", "v0.5.9")),
        stable_package_version=str(
            sglang_data.get("stable_package_version", "0.5.9")
        ),
        torch_version=str(sglang_data.get("torch_version", "2.9.1+cu128")),
        torchvision_version=str(
            sglang_data.get("torchvision_version", "0.24.1+cu128")
        ),
        torchaudio_version=str(
            sglang_data.get("torchaudio_version", "2.9.1+cu128")
        ),
        torch_index_url=str(
            sglang_data.get(
                "torch_index_url", "https://download.pytorch.org/whl/cu128"
            )
        ),
        extra_index_url=str(
            sglang_data.get("extra_index_url", "https://pypi.org/simple")
        ),
        launch_model_name=str(
            sglang_data.get("launch_model_name", runtime.base_model)
        ),
        max_loras_per_batch=int(sglang_data.get("max_loras_per_batch", 8)),
        max_loaded_loras=int(sglang_data.get("max_loaded_loras", 16)),
        mem_fraction_static=float(sglang_data.get("mem_fraction_static", 0.35)),
        max_running_requests=int(sglang_data.get("max_running_requests", 1)),
        max_total_tokens=int(sglang_data.get("max_total_tokens", 1024)),
        context_length=int(sglang_data.get("context_length", 1024)),
        disable_cuda_graph=bool(sglang_data.get("disable_cuda_graph", True)),
    )

    dashboard_data = data.get("dashboard", {})
    dashboard = DashboardConfig(
        host=str(dashboard_data.get("host", "127.0.0.1")),
        backend_port=int(dashboard_data.get("backend_port", 8765)),
        frontend_port=int(dashboard_data.get("frontend_port", 4173)),
        frontend_dir=coerce_path(
            str(dashboard_data.get("frontend_dir", "frontend")),
            base_dir=repo_root,
        ),
    )

    paper_data = data.get("paper", {})
    paper = PaperConfig(
        conversation_source_url=str(paper_data.get("conversation_source_url", "")),
        writing_dataset_name=str(paper_data.get("writing_dataset_name", "EdinburghNLP/xsum")),
        writing_dataset_split=str(paper_data.get("writing_dataset_split", "train")),
        app_usage_source_url=str(paper_data.get("app_usage_source_url", "")),
        prepared_dir=resolve_asset_subdir(
            asset_root,
            paper_data.get("prepared_dir", "datasets/paper"),
        ),
        traces_dir=resolve_asset_subdir(
            asset_root,
            paper_data.get("traces_dir", "datasets/paper_traces"),
        ),
        baselines=tuple(
            str(item)
            for item in paper_data.get(
                "baselines",
                ["hf_peft", "sglang_stock_lora", "sglang_mobilora"],
            )
        ),
        smoke_request_count=int(paper_data.get("smoke_request_count", 12)),
        frontend_default_variant=str(
            paper_data.get("frontend_default_variant", "sglang_mobilora")
        ),
    )

    return AppConfig(
        paths=paths,
        runtime=runtime,
        adapter_specs=adapter_specs_from_repo_ids(data["adapter_repos"]),
        delta=delta,
        eviction=eviction,
        bench=bench,
        sglang=sglang,
        dashboard=dashboard,
        paper=paper,
    )


def ensure_runtime_directories(config: AppConfig) -> None:
    for path in (
        config.paths.asset_root,
        config.paths.hf_cache,
        config.paths.datasets_cache,
        config.paths.adapters_cache,
        config.paths.bench_outputs,
        config.paper.prepared_dir,
        config.paper.traces_dir,
    ):
        ensure_directory(path)

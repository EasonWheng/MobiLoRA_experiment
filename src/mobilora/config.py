from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mobilora.types import (
    AppConfig,
    BenchConfig,
    DeltaConfig,
    EvictionConfig,
    PathConfig,
    RuntimeConfig,
    ScenarioConfig,
    adapter_specs_from_repo_ids,
)
from mobilora.utils import default_asset_root, ensure_directory, resolve_asset_subdir


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

    asset_root = Path(data.get("paths", {}).get("asset_root") or default_asset_root())
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
            str(name): Path(path) for name, path in bench_data["dataset_files"].items()
        },
        scenarios=tuple(_scenario_from_dict(item) for item in bench_data["scenarios"]),
    )

    return AppConfig(
        paths=paths,
        runtime=runtime,
        adapter_specs=adapter_specs_from_repo_ids(data["adapter_repos"]),
        delta=delta,
        eviction=eviction,
        bench=bench,
    )


def ensure_runtime_directories(config: AppConfig) -> None:
    for path in (
        config.paths.asset_root,
        config.paths.hf_cache,
        config.paths.datasets_cache,
        config.paths.adapters_cache,
        config.paths.bench_outputs,
    ):
        ensure_directory(path)

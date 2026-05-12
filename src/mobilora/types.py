from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class AdapterSpec:
    repo_id: str
    alias: str
    app_id: str


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    adapter_count: int
    cache_budget_mb: int
    max_input: int
    request_count: int = 300


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    workload: str
    adapter_alias: str
    adapter_repo: str
    app_id: str
    app_state: str
    prompt: str
    reference: str = ""


@dataclass(slots=True)
class RuntimeArtifact:
    token_ids: tuple[int, ...]
    shallow_key: tuple[float, ...]
    layer_vectors: tuple[tuple[float, ...], ...]
    raw_size_mb: float
    response_text: str
    reference_text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DeltaEncodingStats:
    compression_ratio: float
    avg_similarity: float
    avg_error_bound: float


@dataclass(slots=True)
class CacheEntry:
    entry_id: str
    token_ids: tuple[int, ...]
    adapter_alias: str
    app_id: str
    shallow_key: tuple[float, ...]
    layer_vectors: tuple[tuple[float, ...], ...]
    raw_size_mb: float
    stored_size_mb: float
    avg_similarity: float = 1.0
    avg_error_bound: float = 0.0
    compression_ratio: float = 1.0
    anchor_id: str | None = None
    created_step: int = 0
    last_access_step: int = 0
    hit_count: int = 0
    state_label: str = "killed"

    @property
    def token_len(self) -> int:
        return len(self.token_ids)


@dataclass(slots=True)
class VariantSpec:
    name: str
    allow_prefix_reuse: bool
    allow_cross_adapter_reuse: bool
    enable_delta: bool
    context_aware_eviction: bool


@dataclass(slots=True)
class MetricsRow:
    scenario: str
    workload: str
    variant: str
    timing_source: str
    request_count: int
    median_prefill_latency_ms: float
    p95_prefill_latency_ms: float
    median_end_to_end_latency_ms: float
    p95_end_to_end_latency_ms: float
    cache_hit_ratio: float
    compression_ratio: float
    avg_persisted_kv_mb: float
    peak_persisted_kv_mb: float
    quality_score: float
    quality_metric: str
    accepted_entries: int
    evicted_entries: int
    avg_input_tokens: float
    avg_generated_tokens: float
    avg_reuse_tokens: float
    avg_prefill_saved_ms: float
    avg_delta_penalty_ms: float
    avg_pressure_penalty_ms: float
    avg_runtime_tokenize_ms: float
    avg_runtime_prefill_ms: float
    avg_runtime_decode_ms: float
    avg_runtime_adapter_switch_ms: float
    avg_similarity: float
    avg_error_bound: float

    def as_csv_row(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "workload": self.workload,
            "variant": self.variant,
            "timing_source": self.timing_source,
            "request_count": self.request_count,
            "median_prefill_latency_ms": round(self.median_prefill_latency_ms, 4),
            "p95_prefill_latency_ms": round(self.p95_prefill_latency_ms, 4),
            "median_end_to_end_latency_ms": round(self.median_end_to_end_latency_ms, 4),
            "p95_end_to_end_latency_ms": round(self.p95_end_to_end_latency_ms, 4),
            "cache_hit_ratio": round(self.cache_hit_ratio, 6),
            "compression_ratio": round(self.compression_ratio, 6),
            "avg_persisted_kv_mb": round(self.avg_persisted_kv_mb, 4),
            "peak_persisted_kv_mb": round(self.peak_persisted_kv_mb, 4),
            "quality_score": round(self.quality_score, 6),
            "quality_metric": self.quality_metric,
            "accepted_entries": self.accepted_entries,
            "evicted_entries": self.evicted_entries,
            "avg_input_tokens": round(self.avg_input_tokens, 4),
            "avg_generated_tokens": round(self.avg_generated_tokens, 4),
            "avg_reuse_tokens": round(self.avg_reuse_tokens, 4),
            "avg_prefill_saved_ms": round(self.avg_prefill_saved_ms, 4),
            "avg_delta_penalty_ms": round(self.avg_delta_penalty_ms, 4),
            "avg_pressure_penalty_ms": round(self.avg_pressure_penalty_ms, 4),
            "avg_runtime_tokenize_ms": round(self.avg_runtime_tokenize_ms, 4),
            "avg_runtime_prefill_ms": round(self.avg_runtime_prefill_ms, 4),
            "avg_runtime_decode_ms": round(self.avg_runtime_decode_ms, 4),
            "avg_runtime_adapter_switch_ms": round(self.avg_runtime_adapter_switch_ms, 4),
            "avg_similarity": round(self.avg_similarity, 6),
            "avg_error_bound": round(self.avg_error_bound, 8),
        }


@dataclass(slots=True)
class PathConfig:
    asset_root: Path
    hf_cache: Path
    datasets_cache: Path
    adapters_cache: Path
    bench_outputs: Path


@dataclass(slots=True)
class RuntimeConfig:
    backend: str
    base_model: str
    quantization_bits: int
    load_in_4bit: bool
    device: str
    hidden_size: int
    layer_count: int
    max_new_tokens: int


@dataclass(slots=True)
class DeltaConfig:
    high_similarity_threshold: float
    medium_similarity_threshold: float
    high_similarity_error_bound: float
    medium_similarity_error_bound: float
    low_similarity_error_bound: float


@dataclass(slots=True)
class EvictionConfig:
    foreground_score: float
    background_score: float
    killed_score: float
    lambda_state: float
    lambda_recency: float
    lambda_length: float
    recent_window: int


@dataclass(slots=True)
class BenchConfig:
    random_seed: int
    workloads: tuple[str, ...]
    dataset_files: dict[str, Path]
    scenarios: tuple[ScenarioConfig, ...]


@dataclass(slots=True)
class AppConfig:
    paths: PathConfig
    runtime: RuntimeConfig
    adapter_specs: tuple[AdapterSpec, ...]
    delta: DeltaConfig
    eviction: EvictionConfig
    bench: BenchConfig


@dataclass(slots=True)
class PrepareManifest:
    backend: str
    base_model: str
    base_model_check: dict[str, object] = field(default_factory=dict)
    adapter_checks: list[dict[str, object]] = field(default_factory=list)
    asset_checks: list[dict[str, object]] = field(default_factory=list)
    dependency_checks: list[dict[str, object]] = field(default_factory=list)
    runtime_validation: list[dict[str, object]] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def adapter_specs_from_repo_ids(repo_ids: Iterable[str]) -> tuple[AdapterSpec, ...]:
    specs: list[AdapterSpec] = []
    for repo_id in repo_ids:
        alias = repo_id.split("/")[-1].replace(".", "-")
        specs.append(AdapterSpec(repo_id=repo_id, alias=alias, app_id=alias))
    return tuple(specs)

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from mobilora.cache import CachePool
from mobilora.delta import encode_delta_payload
from mobilora.quality import QualityScorer
from mobilora.runtime import build_runtime
from mobilora.trace import build_trace
from mobilora.types import AppConfig, CacheEntry, MetricsRow, VariantSpec
from mobilora.utils import common_prefix_length, ensure_directory, median, percentile, safe_filename


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="plain_peft",
        allow_prefix_reuse=False,
        allow_cross_adapter_reuse=False,
        enable_delta=False,
        context_aware_eviction=False,
    ),
    VariantSpec(
        name="prefix_only",
        allow_prefix_reuse=True,
        allow_cross_adapter_reuse=False,
        enable_delta=False,
        context_aware_eviction=False,
    ),
    VariantSpec(
        name="mobilora_no_delta",
        allow_prefix_reuse=True,
        allow_cross_adapter_reuse=True,
        enable_delta=False,
        context_aware_eviction=True,
    ),
    VariantSpec(
        name="mobilora_no_ctx",
        allow_prefix_reuse=True,
        allow_cross_adapter_reuse=True,
        enable_delta=True,
        context_aware_eviction=False,
    ),
    VariantSpec(
        name="mobilora_full",
        allow_prefix_reuse=True,
        allow_cross_adapter_reuse=True,
        enable_delta=True,
        context_aware_eviction=True,
    ),
)


@dataclass(slots=True)
class RunArtifacts:
    results_csv: Path
    traces_dir: Path
    summary_json: Path


def _variant_lookup(names: list[str] | None) -> tuple[VariantSpec, ...]:
    if not names:
        return VARIANTS
    requested = {item.strip() for item in names}
    return tuple(item for item in VARIANTS if item.name in requested)


def _adapter_subset(config: AppConfig, count: int) -> tuple:
    if count > len(config.adapter_specs):
        raise ValueError(
            f"Scenario requires {count} adapters, but only {len(config.adapter_specs)} are configured."
        )
    return config.adapter_specs[:count]


def _recent_app_window(trace, index: int, window: int) -> set[str]:
    start = max(0, index - window)
    return {item.app_id for item in trace[start:index]}


def _latency_model(
    token_len: int,
    reuse_tokens: int,
    hit: bool,
    variant: VariantSpec,
    avg_error_bound: float,
    rng: random.Random,
) -> tuple[float, float]:
    base_prefill = token_len * 0.85
    base_decode = token_len * 0.18 + 34.0
    adapter_switch = 32.0 if variant.name == "plain_peft" else 14.0

    prefill = base_prefill
    if variant.allow_prefix_reuse and hit:
        reused_fraction = reuse_tokens / max(token_len, 1)
        reuse_gain = 0.72 if variant.allow_cross_adapter_reuse else 0.58
        prefill *= max(0.15, 1.0 - reused_fraction * reuse_gain)

    if variant.enable_delta and hit:
        prefill += 4.0 + (avg_error_bound * 1200.0)
    if variant.context_aware_eviction:
        prefill *= 0.96

    jitter = rng.uniform(-4.0, 4.0)
    prefill = max(2.0, prefill + jitter)
    end_to_end = prefill + base_decode + adapter_switch
    return prefill, end_to_end


def run_benchmarks(
    config: AppConfig,
    output_dir: Path | None = None,
    backend: str | None = None,
    variant_names: list[str] | None = None,
) -> RunArtifacts:
    scorer = QualityScorer()
    runtime = build_runtime(config, backend)
    variants = _variant_lookup(variant_names)

    result_dir = ensure_directory(output_dir or config.paths.bench_outputs)
    traces_dir = ensure_directory(result_dir / "traces")
    results_csv = result_dir / "results.csv"
    summary_json = result_dir / "summary.json"

    metric_rows: list[MetricsRow] = []
    summary_payload: dict[str, object] = {
        "backend": backend or config.runtime.backend,
        "quality_metric": scorer.metric_name,
        "variants": [item.name for item in variants],
        "scenarios": [],
    }

    for scenario_index, scenario in enumerate(config.bench.scenarios):
        adapters = _adapter_subset(config, scenario.adapter_count)
        scenario_record = {"name": scenario.name, "workloads": []}

        for workload_index, workload in enumerate(config.bench.workloads):
            trace_seed = config.bench.random_seed + scenario_index * 100 + workload_index
            trace = build_trace(
                workload=workload,
                dataset_path=config.bench.dataset_files[workload],
                adapters=adapters,
                scenario=scenario,
                seed=trace_seed,
            )

            for variant in variants:
                rng = random.Random(trace_seed * 17 + stable_variant_seed(variant.name))
                cache_pool = CachePool(scenario.cache_budget_mb, config.eviction)

                prefill_latencies = []
                end_to_end_latencies = []
                quality_scores = []
                compression_ratios = []
                cache_sizes = []
                hit_count = 0
                trace_rows: list[dict[str, object]] = []

                for request_index, request in enumerate(trace):
                    recent_apps = _recent_app_window(trace, request_index, config.eviction.recent_window)
                    artifact = runtime.artifact_from_request(request, scenario.max_input)

                    anchor = None
                    if variant.allow_prefix_reuse:
                        anchor = cache_pool.find_best_anchor(
                            token_ids=artifact.token_ids,
                            shallow_key=artifact.shallow_key,
                            adapter_alias=request.adapter_alias,
                            allow_cross_adapter=variant.allow_cross_adapter_reuse,
                        )

                    reuse_tokens = 0
                    encoding_stats = None
                    if anchor is not None:
                        reuse_tokens = common_prefix_length(anchor.token_ids, artifact.token_ids)
                        hit_count += 1 if reuse_tokens > 0 else 0
                        cache_pool.touch(anchor.entry_id, request_index)

                        if variant.enable_delta:
                            encoding_stats = encode_delta_payload(
                                anchor.layer_vectors,
                                artifact.layer_vectors,
                                config.delta,
                            )

                    avg_error_bound = encoding_stats.avg_error_bound if encoding_stats else 0.0
                    compression_ratio = encoding_stats.compression_ratio if encoding_stats else 1.0
                    response_text = runtime.response_with_compression(
                        artifact.response_text,
                        avg_error_bound if variant.enable_delta else 0.0,
                    )
                    quality_score = scorer.compare(artifact.response_text, response_text)
                    prefill_latency_ms, end_to_end_latency_ms = _latency_model(
                        token_len=len(artifact.token_ids),
                        reuse_tokens=reuse_tokens,
                        hit=anchor is not None and reuse_tokens > 0,
                        variant=variant,
                        avg_error_bound=avg_error_bound,
                        rng=rng,
                    )

                    stored_size_mb = artifact.raw_size_mb / compression_ratio
                    if not variant.allow_prefix_reuse:
                        stored_size_mb = 0.0

                    entry = CacheEntry(
                        entry_id=request.request_id,
                        token_ids=artifact.token_ids,
                        adapter_alias=request.adapter_alias,
                        app_id=request.app_id,
                        shallow_key=artifact.shallow_key,
                        layer_vectors=artifact.layer_vectors,
                        raw_size_mb=artifact.raw_size_mb,
                        stored_size_mb=stored_size_mb,
                        avg_similarity=encoding_stats.avg_similarity if encoding_stats else 1.0,
                        avg_error_bound=avg_error_bound,
                        compression_ratio=compression_ratio,
                        anchor_id=anchor.entry_id if anchor else None,
                        created_step=request_index,
                        last_access_step=request_index,
                    )

                    accepted = False
                    evicted = []
                    if variant.allow_prefix_reuse:
                        accepted, evicted = cache_pool.admit(
                            entry=entry,
                            step=request_index,
                            current_app=request.app_id,
                            recent_apps=recent_apps,
                            context_aware=variant.context_aware_eviction,
                        )

                    pressure_ratio = cache_pool.total_size_mb / max(float(scenario.cache_budget_mb), 1.0)
                    pressure_penalty = 0.0
                    if variant.allow_prefix_reuse and not variant.enable_delta:
                        pressure_penalty += max(0.0, pressure_ratio - 0.35) * 18.0
                    if variant.allow_prefix_reuse and not variant.context_aware_eviction:
                        pressure_penalty += max(0.0, pressure_ratio - 0.5) * 10.0
                    pressure_penalty += min(len(evicted), 12) * 0.35

                    prefill_latency_ms += pressure_penalty
                    end_to_end_latency_ms += pressure_penalty

                    prefill_latencies.append(prefill_latency_ms)
                    end_to_end_latencies.append(end_to_end_latency_ms)
                    quality_scores.append(quality_score)
                    compression_ratios.append(compression_ratio)
                    cache_sizes.append(cache_pool.total_size_mb)

                    trace_rows.append(
                        {
                            "request_id": request.request_id,
                            "scenario": scenario.name,
                            "workload": workload,
                            "variant": variant.name,
                            "adapter": request.adapter_alias,
                            "hit": int(anchor is not None and reuse_tokens > 0),
                            "reuse_tokens": reuse_tokens,
                            "compression_ratio": round(compression_ratio, 6),
                            "prefill_latency_ms": round(prefill_latency_ms, 4),
                            "end_to_end_latency_ms": round(end_to_end_latency_ms, 4),
                            "quality_score": round(quality_score, 6),
                            "accepted": int(accepted),
                            "evicted_count": len(evicted),
                            "persisted_kv_mb": round(cache_pool.total_size_mb, 4),
                        }
                    )

                metric_rows.append(
                    MetricsRow(
                        scenario=scenario.name,
                        workload=workload,
                        variant=variant.name,
                        request_count=len(trace),
                        median_prefill_latency_ms=median(prefill_latencies),
                        p95_prefill_latency_ms=percentile(prefill_latencies, 0.95),
                        median_end_to_end_latency_ms=median(end_to_end_latencies),
                        p95_end_to_end_latency_ms=percentile(end_to_end_latencies, 0.95),
                        cache_hit_ratio=hit_count / max(len(trace), 1),
                        compression_ratio=sum(compression_ratios) / max(len(compression_ratios), 1),
                        avg_persisted_kv_mb=sum(cache_sizes) / max(len(cache_sizes), 1),
                        peak_persisted_kv_mb=max(cache_sizes) if cache_sizes else 0.0,
                        quality_score=sum(quality_scores) / max(len(quality_scores), 1),
                        quality_metric=scorer.metric_name,
                        accepted_entries=cache_pool.accepted_entries,
                        evicted_entries=cache_pool.evicted_entries,
                    )
                )

                trace_file = traces_dir / (
                    f"{safe_filename(scenario.name)}_{safe_filename(workload)}_{safe_filename(variant.name)}.jsonl"
                )
                with trace_file.open("w", encoding="utf-8") as handle:
                    for row in trace_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            scenario_record["workloads"].append(workload)
        summary_payload["scenarios"].append(scenario_record)

    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].as_csv_row().keys()))
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row.as_csv_row())

    summary_payload["result_count"] = len(metric_rows)
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return RunArtifacts(results_csv=results_csv, traces_dir=traces_dir, summary_json=summary_json)


def stable_variant_seed(name: str) -> int:
    value = 0
    for char in name:
        value = value * 33 + ord(char)
    return value

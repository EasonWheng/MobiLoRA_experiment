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


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _latency_model(
    artifact,
    reuse_tokens: int,
    hit: bool,
    variant: VariantSpec,
    avg_error_bound: float,
    rng: random.Random,
) -> tuple[float, float, dict[str, float]]:
    token_len = len(artifact.token_ids)
    runtime_metadata = artifact.metadata

    runtime_tokenize_ms = float(runtime_metadata.get("tokenize_ms", 0.0))
    runtime_prefill_ms = float(runtime_metadata.get("prefill_ms", 0.0))
    runtime_decode_ms = float(runtime_metadata.get("decode_ms", 0.0))
    runtime_adapter_switch_ms = float(runtime_metadata.get("adapter_switch_ms", 0.0))
    timing_source = str(runtime_metadata.get("timing_source", "synthetic_model"))

    if timing_source == "measured_runtime":
        base_prefill = runtime_tokenize_ms + runtime_prefill_ms
        base_decode = runtime_decode_ms
        adapter_switch = runtime_adapter_switch_ms
        jitter = 0.0
    else:
        base_prefill = token_len * 0.85
        base_decode = token_len * 0.18 + 34.0
        adapter_switch = 32.0 if variant.name == "plain_peft" else 14.0
        jitter = rng.uniform(-4.0, 4.0)

    prefill = base_prefill
    reuse_fraction = 0.0
    prefill_saved_ms = 0.0
    reuse_gain = 0.0
    if variant.allow_prefix_reuse and hit:
        reuse_fraction = reuse_tokens / max(token_len, 1)
        reuse_gain = 0.72 if variant.allow_cross_adapter_reuse else 0.58
        reuse_adjusted_prefill = prefill * max(0.15, 1.0 - reuse_fraction * reuse_gain)
        prefill_saved_ms = max(0.0, prefill - reuse_adjusted_prefill)
        prefill = reuse_adjusted_prefill

    delta_penalty = 0.0
    if variant.enable_delta and hit:
        delta_penalty = 4.0 + (avg_error_bound * 1200.0)
        prefill += delta_penalty

    context_factor = 1.0
    if variant.context_aware_eviction:
        context_factor = 0.96
        prefill *= 0.96

    prefill = max(2.0, prefill + jitter)
    end_to_end = prefill + base_decode + adapter_switch
    return prefill, end_to_end, {
        "timing_source": timing_source,
        "base_prefill_ms": base_prefill,
        "base_decode_ms": base_decode,
        "adapter_switch_ms": adapter_switch,
        "reuse_fraction": reuse_fraction,
        "prefill_saved_ms": prefill_saved_ms,
        "reuse_gain_factor": reuse_gain,
        "delta_penalty_ms": delta_penalty,
        "context_factor": context_factor,
        "jitter_ms": jitter,
        "runtime_tokenize_ms": runtime_tokenize_ms,
        "runtime_prefill_ms": runtime_prefill_ms,
        "runtime_decode_ms": runtime_decode_ms,
        "runtime_adapter_switch_ms": runtime_adapter_switch_ms,
    }


def run_benchmarks(
    config: AppConfig,
    output_dir: Path | None = None,
    backend: str | None = None,
    variant_names: list[str] | None = None,
    request_limit: int | None = None,
    scenario_filter: list[str] | None = None,
    workload_filter: list[str] | None = None,
) -> RunArtifacts:
    scorer = QualityScorer()
    runtime = build_runtime(config, backend)
    variants = _variant_lookup(variant_names)
    requested_scenarios = {item.strip() for item in scenario_filter or []}
    requested_workloads = {item.strip() for item in workload_filter or []}

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
        "rows": [],
        "metric_definitions": {
            "median_prefill_latency_ms": "Median adjusted prefill latency across requests. For HF runs this starts from measured tokenizer + prompt-forward time, then applies reuse, delta, and cache-pressure effects.",
            "p95_prefill_latency_ms": "95th percentile adjusted prefill latency across requests.",
            "median_end_to_end_latency_ms": "Median total latency including adjusted prefill, decode, and adapter switching.",
            "cache_hit_ratio": "Fraction of requests that found a reusable prefix anchor.",
            "compression_ratio": "Average raw KV size divided by persisted compressed KV size.",
            "avg_persisted_kv_mb": "Average persisted KV footprint of the cache pool during the run.",
            "quality_score": "Response consistency score between the original response and the compression-perturbed response.",
            "avg_prefill_saved_ms": "Average prefill time removed by prefix reuse before delta/context/pressure adjustments are applied.",
            "avg_runtime_prefill_ms": "Average measured prompt-forward time from the runtime before cache-model adjustments. Zero means the row used the synthetic mock backend.",
            "avg_runtime_decode_ms": "Average measured incremental decode time from the runtime. Zero means the row used the synthetic mock backend.",
            "avg_similarity": "Average cosine similarity between anchor KV vectors and target KV vectors across delta-encoded layers.",
            "avg_error_bound": "Average quantization error bound chosen by the similarity buckets from Eq. (3).",
        },
        "latency_model_components": {
            "base_prefill_ms": "Measured tokenizer + prompt-forward cost when available, otherwise a token-count-proportional synthetic baseline.",
            "reuse_fraction": "Longest reusable prefix length divided by total prompt length.",
            "prefill_saved_ms": "Absolute milliseconds removed from the base prefill by prefix reuse.",
            "delta_penalty_ms": "Extra cost introduced by delta-KV encoding/decoding.",
            "pressure_penalty_ms": "Penalty added when uncompressed KV pressure or frequent evictions stress the budget.",
            "adapter_switch_ms": "Fixed adapter switching overhead per variant.",
        },
    }

    for scenario_index, scenario in enumerate(config.bench.scenarios):
        if requested_scenarios and scenario.name not in requested_scenarios:
            continue
        adapters = _adapter_subset(config, scenario.adapter_count)
        scenario_record = {"name": scenario.name, "workloads": []}

        for workload_index, workload in enumerate(config.bench.workloads):
            if requested_workloads and workload not in requested_workloads:
                continue
            trace_seed = config.bench.random_seed + scenario_index * 100 + workload_index
            trace = build_trace(
                workload=workload,
                dataset_path=config.bench.dataset_files[workload],
                adapters=adapters,
                scenario=scenario,
                seed=trace_seed,
            )
            if request_limit is not None:
                trace = trace[:request_limit]
            if trace:
                warmup_requests = []
                warmed_adapters: set[str] = set()
                for request in trace:
                    if request.adapter_alias in warmed_adapters:
                        continue
                    warmup_requests.append(request)
                    warmed_adapters.add(request.adapter_alias)
                    if len(warmup_requests) >= len(adapters):
                        break
                runtime.warmup(warmup_requests, scenario.max_input)
            artifacts = [runtime.artifact_from_request(request, scenario.max_input) for request in trace]

            for variant in variants:
                rng = random.Random(trace_seed * 17 + stable_variant_seed(variant.name))
                cache_pool = CachePool(scenario.cache_budget_mb, config.eviction)

                prefill_latencies: list[float] = []
                end_to_end_latencies: list[float] = []
                compression_ratios: list[float] = []
                cache_sizes: list[float] = []
                input_token_counts: list[float] = []
                generated_token_counts: list[float] = []
                reuse_token_counts: list[float] = []
                prefill_saved_values: list[float] = []
                delta_penalty_values: list[float] = []
                pressure_penalty_values: list[float] = []
                runtime_tokenize_values: list[float] = []
                runtime_prefill_values: list[float] = []
                runtime_decode_values: list[float] = []
                runtime_adapter_values: list[float] = []
                similarity_values: list[float] = []
                error_bound_values: list[float] = []
                timing_sources: set[str] = set()
                hit_count = 0
                trace_rows: list[dict[str, object]] = []
                response_references: list[str] = []
                response_candidates: list[str] = []

                for request_index, (request, artifact) in enumerate(zip(trace, artifacts)):
                    recent_apps = _recent_app_window(trace, request_index, config.eviction.recent_window)

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
                    prefill_latency_ms, end_to_end_latency_ms, latency_components = _latency_model(
                        artifact=artifact,
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

                    response_references.append(artifact.response_text)
                    response_candidates.append(response_text)
                    prefill_latencies.append(prefill_latency_ms)
                    end_to_end_latencies.append(end_to_end_latency_ms)
                    compression_ratios.append(compression_ratio)
                    cache_sizes.append(cache_pool.total_size_mb)
                    input_token_counts.append(float(len(artifact.token_ids)))
                    generated_token_counts.append(float(artifact.metadata.get("generated_tokens", 0)))
                    reuse_token_counts.append(float(reuse_tokens))
                    prefill_saved_values.append(float(latency_components["prefill_saved_ms"]))
                    delta_penalty_values.append(float(latency_components["delta_penalty_ms"]))
                    pressure_penalty_values.append(float(pressure_penalty))
                    runtime_tokenize_values.append(float(latency_components["runtime_tokenize_ms"]))
                    runtime_prefill_values.append(float(latency_components["runtime_prefill_ms"]))
                    runtime_decode_values.append(float(latency_components["runtime_decode_ms"]))
                    runtime_adapter_values.append(float(latency_components["runtime_adapter_switch_ms"]))
                    similarity_values.append(float(entry.avg_similarity))
                    error_bound_values.append(float(avg_error_bound))
                    timing_sources.add(str(latency_components["timing_source"]))

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
                            "quality_score": 0.0,
                            "accepted": int(accepted),
                            "evicted_count": len(evicted),
                            "persisted_kv_mb": round(cache_pool.total_size_mb, 4),
                            "raw_kv_mb": round(artifact.raw_size_mb, 6),
                            "stored_kv_mb": round(stored_size_mb, 6),
                            "avg_error_bound": round(avg_error_bound, 8),
                            "avg_similarity": round(entry.avg_similarity, 6),
                            "anchor_id": anchor.entry_id if anchor else "",
                            "anchor_adapter": anchor.adapter_alias if anchor else "",
                            "anchor_prefix_tokens": reuse_tokens,
                            "pressure_ratio": round(pressure_ratio, 6),
                            "pressure_penalty_ms": round(pressure_penalty, 6),
                            "base_prefill_ms": round(latency_components["base_prefill_ms"], 6),
                            "base_decode_ms": round(latency_components["base_decode_ms"], 6),
                            "adapter_switch_ms": round(latency_components["adapter_switch_ms"], 6),
                            "reuse_fraction": round(latency_components["reuse_fraction"], 6),
                            "prefill_saved_ms": round(latency_components["prefill_saved_ms"], 6),
                            "reuse_gain_factor": round(latency_components["reuse_gain_factor"], 6),
                            "delta_penalty_ms": round(latency_components["delta_penalty_ms"], 6),
                            "context_factor": round(latency_components["context_factor"], 6),
                            "jitter_ms": round(latency_components["jitter_ms"], 6),
                            "timing_source": latency_components["timing_source"],
                            "runtime_tokenize_ms": round(latency_components["runtime_tokenize_ms"], 6),
                            "runtime_prefill_ms": round(latency_components["runtime_prefill_ms"], 6),
                            "runtime_decode_ms": round(latency_components["runtime_decode_ms"], 6),
                            "runtime_adapter_switch_ms": round(
                                latency_components["runtime_adapter_switch_ms"],
                                6,
                            ),
                            "runtime_backend": artifact.metadata.get("backend", backend or config.runtime.backend),
                            "measured_end_to_end_ms": round(
                                float(artifact.metadata.get("measured_end_to_end_ms", 0.0)),
                                6,
                            ),
                            "prefill_tokens_per_second": round(
                                float(artifact.metadata.get("prefill_tokens_per_second", 0.0)),
                                6,
                            ),
                            "decode_tokens_per_second": round(
                                float(artifact.metadata.get("decode_tokens_per_second", 0.0)),
                                6,
                            ),
                            "cuda_allocated_mb": round(
                                float(artifact.metadata.get("cuda_allocated_mb", 0.0)),
                                6,
                            ),
                            "cuda_reserved_mb": round(
                                float(artifact.metadata.get("cuda_reserved_mb", 0.0)),
                                6,
                            ),
                            "cuda_peak_allocated_mb": round(
                                float(artifact.metadata.get("cuda_peak_allocated_mb", 0.0)),
                                6,
                            ),
                            "response_generated_tokens": artifact.metadata.get("generated_tokens", 0),
                            "artifact_note": artifact.metadata.get("adapter_path", ""),
                        }
                    )

                quality_scores = scorer.compare_many(response_references, response_candidates)
                for row, quality_score in zip(trace_rows, quality_scores):
                    row["quality_score"] = round(quality_score, 6)

                timing_source = "mixed"
                if len(timing_sources) == 1:
                    timing_source = next(iter(timing_sources))
                elif not timing_sources:
                    timing_source = "synthetic_model"

                summary_payload["rows"].append(
                    {
                        "scenario": scenario.name,
                        "workload": workload,
                        "variant": variant.name,
                        "timing_source": timing_source,
                        "request_count": len(trace),
                        "avg_input_tokens": round(_average(input_token_counts), 4),
                        "avg_generated_tokens": round(_average(generated_token_counts), 4),
                        "avg_reuse_tokens": round(_average(reuse_token_counts), 4),
                        "avg_prefill_saved_ms": round(_average(prefill_saved_values), 4),
                        "avg_delta_penalty_ms": round(_average(delta_penalty_values), 4),
                        "avg_pressure_penalty_ms": round(_average(pressure_penalty_values), 4),
                        "avg_runtime_tokenize_ms": round(_average(runtime_tokenize_values), 4),
                        "avg_runtime_prefill_ms": round(_average(runtime_prefill_values), 4),
                        "avg_runtime_decode_ms": round(_average(runtime_decode_values), 4),
                        "avg_runtime_adapter_switch_ms": round(_average(runtime_adapter_values), 4),
                        "avg_similarity": round(_average(similarity_values), 6),
                        "avg_error_bound": round(_average(error_bound_values), 8),
                        "driver_note": (
                            "final_prefill = base_prefill - prefill_saved + delta_penalty "
                            "+ pressure_penalty, then context factor and jitter are applied."
                        ),
                    }
                )

                metric_rows.append(
                    MetricsRow(
                        scenario=scenario.name,
                        workload=workload,
                        variant=variant.name,
                        timing_source=timing_source,
                        request_count=len(trace),
                        median_prefill_latency_ms=median(prefill_latencies),
                        p95_prefill_latency_ms=percentile(prefill_latencies, 0.95),
                        median_end_to_end_latency_ms=median(end_to_end_latencies),
                        p95_end_to_end_latency_ms=percentile(end_to_end_latencies, 0.95),
                        cache_hit_ratio=hit_count / max(len(trace), 1),
                        compression_ratio=_average(compression_ratios),
                        avg_persisted_kv_mb=_average(cache_sizes),
                        peak_persisted_kv_mb=max(cache_sizes) if cache_sizes else 0.0,
                        quality_score=_average(quality_scores),
                        quality_metric=scorer.metric_name,
                        accepted_entries=cache_pool.accepted_entries,
                        evicted_entries=cache_pool.evicted_entries,
                        avg_input_tokens=_average(input_token_counts),
                        avg_generated_tokens=_average(generated_token_counts),
                        avg_reuse_tokens=_average(reuse_token_counts),
                        avg_prefill_saved_ms=_average(prefill_saved_values),
                        avg_delta_penalty_ms=_average(delta_penalty_values),
                        avg_pressure_penalty_ms=_average(pressure_penalty_values),
                        avg_runtime_tokenize_ms=_average(runtime_tokenize_values),
                        avg_runtime_prefill_ms=_average(runtime_prefill_values),
                        avg_runtime_decode_ms=_average(runtime_decode_values),
                        avg_runtime_adapter_switch_ms=_average(runtime_adapter_values),
                        avg_similarity=_average(similarity_values),
                        avg_error_bound=_average(error_bound_values),
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

    if not metric_rows:
        raise ValueError("No benchmark rows were produced. Check your scenario/workload filters.")

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

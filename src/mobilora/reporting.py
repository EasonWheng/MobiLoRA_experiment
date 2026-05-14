from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from mobilora.utils import ensure_directory


def generate_report(results_csv: Path, output_dir: Path) -> Path:
    ensure_directory(output_dir)
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    has_system = "system" in rows[0]
    by_workload: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        system = row.get("system", "default")
        by_workload[(system, row["workload"])].append(row)
        by_key[(system, row["scenario"], row["workload"], row["variant"])] = row

    report_lines = [
        "# MobiLoRA Benchmark Report",
        "",
        f"Source CSV: `{results_csv}`",
        "",
        "## Metric Definitions",
        "",
        "- `median_prefill_latency_ms`: median prefill latency over all requests in the row.",
        "- `p95_prefill_latency_ms`: 95th percentile prefill latency, used to expose tail behavior.",
        "- `median_end_to_end_latency_ms`: total latency including prefill, decode, and adapter switch overhead.",
        "- `cache_hit_ratio`: fraction of requests that found a reusable cached prefix anchor.",
        "- `compression_ratio`: raw KV size divided by persisted KV size after delta compression.",
        "- `avg_persisted_kv_mb`: average KV memory footprint observed during the run.",
        "- `quality_score`: consistency between the original response and the compression-perturbed response.",
        "- `timing_source`: `measured_runtime` means the row starts from real HF runtime timings; `synthetic_model` means it comes from the mock timing model.",
        "- `avg_prefill_saved_ms`: average prefill time removed by prefix reuse before delta and pressure penalties are added back.",
        "- `avg_delta_penalty_ms`: average extra prefill cost introduced by delta KV encoding.",
        "- `avg_pressure_penalty_ms`: average penalty caused by cache budget pressure or evictions.",
        "- `avg_runtime_prefill_ms`: average measured prompt-forward time before cache-model adjustments.",
        "- `system`: serving stack or baseline family, such as `hf_peft` or `sglang_mobilora`.",
        "",
        "## Row Formula",
        "",
        "Each row can be interpreted as:",
        "",
        "- `final_prefill ~= base_prefill - prefill_saved + delta_penalty + pressure_penalty`, then context-aware scaling and mock jitter are applied when enabled.",
        "- For HF runs, `base_prefill` comes from measured tokenizer + prompt-forward time and `base_decode` comes from measured decode time.",
        "- For mock runs, `base_prefill` and `base_decode` come from the synthetic token-count model.",
        "",
        "## Summary",
        "",
    ]

    for (system, workload), workload_rows in by_workload.items():
        scenario_names = sorted({row["scenario"] for row in workload_rows})
        full_wins = {
            "plain_peft": 0,
            "prefix_only": 0,
            "mobilora_no_ctx": 0,
            "mobilora_no_delta": 0,
        }
        full_rows: list[dict[str, str]] = []

        for scenario in scenario_names:
            full_row = by_key.get((system, scenario, workload, "mobilora_full"))
            if full_row is None:
                continue
            full_rows.append(full_row)
            for baseline in tuple(full_wins):
                baseline_row = by_key.get((system, scenario, workload, baseline))
                if baseline_row is None:
                    continue
                if float(full_row["median_prefill_latency_ms"]) < float(baseline_row["median_prefill_latency_ms"]):
                    full_wins[baseline] += 1

        avg_full_compression = (
            sum(float(row["compression_ratio"]) for row in full_rows) / max(len(full_rows), 1)
        )
        avg_full_quality = sum(float(row["quality_score"]) for row in full_rows) / max(len(full_rows), 1)
        report_lines.append(f"### {system} / {workload}")
        if not full_rows:
            report_lines.append(
                "- No `mobilora_full` rows are present for this system, so ablation win counts are not applicable."
            )
            report_lines.append("")
            continue
        report_lines.append(
            f"- `mobilora_full` beats `plain_peft` in {full_wins['plain_peft']}/{len(scenario_names)} scenarios."
        )
        report_lines.append(
            f"- `mobilora_full` beats `prefix_only` in {full_wins['prefix_only']}/{len(scenario_names)} scenarios."
        )
        report_lines.append(
            f"- `mobilora_full` beats `mobilora_no_ctx` in {full_wins['mobilora_no_ctx']}/{len(scenario_names)} scenarios."
        )
        report_lines.append(
            f"- `mobilora_full` beats `mobilora_no_delta` in {full_wins['mobilora_no_delta']}/{len(scenario_names)} scenarios."
        )
        report_lines.append(f"- Average `mobilora_full` compression ratio: `{avg_full_compression:.4f}`.")
        report_lines.append(f"- Average `mobilora_full` quality score: `{avg_full_quality:.4f}`.")
        report_lines.append("")

    report_lines.extend(
        [
            "## Detailed Rows",
            "",
            "| system | scenario | workload | variant | timing | median prefill (ms) | p95 prefill (ms) | hit ratio | compression ratio | avg KV (MB) | quality |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        system = row.get("system", "default") if has_system else "default"
        report_lines.append(
            f"| {system} | "
            "{scenario} | {workload} | {variant} | {timing_source} | {median_prefill_latency_ms} | "
            "{p95_prefill_latency_ms} | {cache_hit_ratio} | {compression_ratio} | "
            "{avg_persisted_kv_mb} | {quality_score} |".format(**row)
        )

    report_lines.extend(
        [
            "",
            "## Driver Table",
            "",
            "| system | scenario | workload | variant | avg input toks | avg reuse toks | avg saved ms | avg delta ms | avg pressure ms | avg runtime prefill ms | avg runtime decode ms | avg sim | avg err bound |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        system = row.get("system", "default") if has_system else "default"
        report_lines.append(
            f"| {system} | "
            "{scenario} | {workload} | {variant} | {avg_input_tokens} | {avg_reuse_tokens} | "
            "{avg_prefill_saved_ms} | {avg_delta_penalty_ms} | {avg_pressure_penalty_ms} | "
            "{avg_runtime_prefill_ms} | {avg_runtime_decode_ms} | {avg_similarity} | "
            "{avg_error_bound} |".format(**row)
        )

    output_path = output_dir / "benchmark_report.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_path

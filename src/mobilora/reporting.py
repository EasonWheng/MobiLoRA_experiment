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

    by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        by_workload[row["workload"]].append(row)
        by_key[(row["scenario"], row["workload"], row["variant"])] = row

    report_lines = [
        "# MobiLoRA Benchmark Report",
        "",
        f"Source CSV: `{results_csv}`",
        "",
        "## Summary",
        "",
    ]

    for workload, workload_rows in by_workload.items():
        scenario_names = sorted({row["scenario"] for row in workload_rows})
        full_wins = {
            "plain_peft": 0,
            "prefix_only": 0,
            "mobilora_no_ctx": 0,
            "mobilora_no_delta": 0,
        }
        full_rows: list[dict[str, str]] = []

        for scenario in scenario_names:
            full_row = by_key.get((scenario, workload, "mobilora_full"))
            if full_row is None:
                continue
            full_rows.append(full_row)
            for baseline in tuple(full_wins):
                baseline_row = by_key.get((scenario, workload, baseline))
                if baseline_row is None:
                    continue
                if float(full_row["median_prefill_latency_ms"]) < float(baseline_row["median_prefill_latency_ms"]):
                    full_wins[baseline] += 1

        avg_full_compression = (
            sum(float(row["compression_ratio"]) for row in full_rows) / max(len(full_rows), 1)
        )
        avg_full_quality = sum(float(row["quality_score"]) for row in full_rows) / max(len(full_rows), 1)
        report_lines.append(f"### {workload}")
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
            "| scenario | workload | variant | median prefill (ms) | p95 prefill (ms) | hit ratio | compression ratio | avg KV (MB) | quality |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        report_lines.append(
            "| {scenario} | {workload} | {variant} | {median_prefill_latency_ms} | "
            "{p95_prefill_latency_ms} | {cache_hit_ratio} | {compression_ratio} | "
            "{avg_persisted_kv_mb} | {quality_score} |".format(**row)
        )

    output_path = output_dir / "benchmark_report.md"
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_path

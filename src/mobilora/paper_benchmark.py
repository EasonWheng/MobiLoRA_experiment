from __future__ import annotations

import csv
import json
from pathlib import Path

from mobilora.assets import base_model_local_dir
from mobilora.benchmark import run_benchmarks
from mobilora.config import ensure_runtime_directories
from mobilora.sglang_manager import probe_sglang_server
from mobilora.types import AppConfig
from mobilora.utils import ensure_directory


def _load_rows(results_csv: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(results_csv.open("r", encoding="utf-8")))


def _tag_system(rows: list[dict[str, str]], system_name: str) -> list[dict[str, str]]:
    updated = []
    for row in rows:
        mapped = dict(row)
        mapped["system"] = system_name
        updated.append(mapped)
    return updated


def run_paper_local(
    config: AppConfig,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, object]:
    ensure_runtime_directories(config)
    result_dir = ensure_directory(output_dir)
    request_limit = min(4, config.paper.smoke_request_count) if smoke else None
    scenario_filter = [config.bench.scenarios[0].name] if smoke and config.bench.scenarios else None
    workload_filter = [config.bench.workloads[0]] if smoke and config.bench.workloads else None

    baseline_outputs: dict[str, object] = {}

    hf_model_dir = base_model_local_dir(config)
    hf_assets_ready = hf_model_dir.exists() and any(hf_model_dir.rglob("*"))
    if smoke:
        hf_dir = ensure_directory(result_dir / "hf_peft")
        hf_artifacts = run_benchmarks(
            config=config,
            output_dir=hf_dir,
            backend="mock",
            variant_names=["plain_peft"],
            request_limit=request_limit,
            scenario_filter=scenario_filter,
            workload_filter=workload_filter,
        )
        baseline_outputs["hf_peft"] = {
            "results_csv": str(hf_artifacts.results_csv),
            "summary_json": str(hf_artifacts.summary_json),
            "note": "Smoke mode uses the mock control-plane backend for the HF PEFT baseline to keep turnaround short.",
        }
    elif not hf_assets_ready:
        baseline_outputs["hf_peft"] = {
            "skipped": True,
            "detail": (
                "HF base model assets are not present locally yet. Run "
                "`python main.py prepare --backend hf --download-assets` inside WSL first."
            ),
        }
    else:
        hf_dir = ensure_directory(result_dir / "hf_peft")
        hf_artifacts = run_benchmarks(
            config=config,
            output_dir=hf_dir,
            backend="hf",
            variant_names=["plain_peft"],
            request_limit=request_limit,
            scenario_filter=scenario_filter,
            workload_filter=workload_filter,
        )
        baseline_outputs["hf_peft"] = {
            "results_csv": str(hf_artifacts.results_csv),
            "summary_json": str(hf_artifacts.summary_json),
        }

    for variant in ("sglang_stock_lora", "sglang_mobilora"):
        live_dir = ensure_directory(result_dir / variant)
        probe = probe_sglang_server(config, variant)
        baseline_outputs[variant] = {"probe": probe}
        if not probe["healthy"]:
            continue
        live_backend = "sglang_stock" if variant == "sglang_stock_lora" else "sglang_mobilora"
        live_artifacts = run_benchmarks(
            config=config,
            output_dir=live_dir,
            backend=live_backend,
            request_limit=request_limit,
            scenario_filter=scenario_filter,
            workload_filter=workload_filter,
        )
        baseline_outputs[variant].update(
            {
                "results_csv": str(live_artifacts.results_csv),
                "summary_json": str(live_artifacts.summary_json),
            }
        )

    combined_rows: list[dict[str, str]] = []
    for baseline_name, payload in baseline_outputs.items():
        results_csv = payload.get("results_csv")
        if not results_csv:
            continue
        rows = _load_rows(Path(results_csv))
        combined_rows.extend(_tag_system(rows, baseline_name))

    combined_csv = result_dir / "paper_results.csv"
    if combined_rows:
        with combined_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)

    summary = {
        "smoke": smoke,
        "baselines": baseline_outputs,
        "combined_results_csv": str(combined_csv) if combined_rows else "",
    }
    summary_path = result_dir / "paper_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "paper_summary_json": str(summary_path),
        "paper_results_csv": str(combined_csv) if combined_rows else "",
        "baselines": baseline_outputs,
    }

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobilora.benchmark import run_benchmarks
from mobilora.config import ensure_runtime_directories, load_config
from mobilora.reporting import generate_report
from mobilora.runtime import build_prepare_manifest, manifest_to_json
from mobilora.utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobiLoRA engineering reproduction CLI")
    parser.add_argument(
        "--config",
        default="configs/runtime.yaml",
        help="Path to the runtime YAML config.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Check env, directories, and adapter metadata.")
    prepare.add_argument("--backend", default=None, help="Backend to validate: mock or hf.")
    prepare.add_argument(
        "--skip-hf-check",
        action="store_true",
        help="Skip remote adapter reachability checks.",
    )
    prepare.add_argument(
        "--output",
        default=None,
        help="Optional manifest JSON path. Defaults to <bench_outputs>/prepare_manifest.json",
    )

    bench = subparsers.add_parser("bench", help="Run the benchmark pipeline.")
    bench.add_argument("--backend", default=None, help="Backend to run: mock or hf.")
    bench.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for result CSV, traces, and summary JSON.",
    )
    bench.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variants to run.",
    )

    report = subparsers.add_parser("report", help="Generate a Markdown report from the latest CSV.")
    report.add_argument(
        "--results-file",
        default=None,
        help="Path to results.csv. Defaults to <bench_outputs>/results.csv",
    )
    report.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write benchmark_report.md. Defaults to <bench_outputs>/reports",
    )
    return parser


def command_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    backend = args.backend or config.runtime.backend
    manifest = build_prepare_manifest(config, backend=backend, skip_hf_check=args.skip_hf_check)
    output_path = Path(args.output) if args.output else config.paths.bench_outputs / "prepare_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest_to_json(manifest), encoding="utf-8")
    print(output_path)
    return 0


def command_bench(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.bench_outputs
    artifacts = run_benchmarks(
        config=config,
        output_dir=output_dir,
        backend=args.backend,
        variant_names=args.variants,
    )
    print(json.dumps(
        {
            "results_csv": str(artifacts.results_csv),
            "traces_dir": str(artifacts.traces_dir),
            "summary_json": str(artifacts.summary_json),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def command_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    results_csv = Path(args.results_file) if args.results_file else config.paths.bench_outputs / "results.csv"
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.bench_outputs / "reports"
    report_path = generate_report(results_csv=results_csv, output_dir=output_dir)
    print(report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "bench":
        return command_bench(args)
    if args.command == "report":
        return command_report(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobilora.benchmark import run_benchmarks
from mobilora.config import ensure_runtime_directories, load_config
from mobilora.dashboard import serve_dashboard
from mobilora.paper_benchmark import run_paper_local
from mobilora.paper_data import prepare_paper_data
from mobilora.reporting import generate_report
from mobilora.runtime import build_prepare_manifest, build_runtime_validation, manifest_to_json
from mobilora.sglang_manager import bootstrap_sglang, default_launch_command, launch_sglang_server


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
    prepare.add_argument(
        "--download-assets",
        action="store_true",
        help="For the HF backend, download the base model and adapter snapshots into the D-drive asset dirs.",
    )
    prepare.add_argument(
        "--warm-runtime",
        action="store_true",
        help="For the selected backend, load the runtime once and record a validation summary.",
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
    bench.add_argument(
        "--request-limit",
        type=int,
        default=None,
        help="Optional cap on requests per scenario/workload, useful for real HF validation runs.",
    )
    bench.add_argument(
        "--scenario-filter",
        nargs="*",
        default=None,
        help="Optional subset of scenario names to run.",
    )
    bench.add_argument(
        "--workload-filter",
        nargs="*",
        default=None,
        help="Optional subset of workloads to run.",
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

    prepare_paper = subparsers.add_parser(
        "prepare-paper-data",
        help="Prepare ShareGPT/XSum-compatible local paper datasets and trace manifests.",
    )
    prepare_paper.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for the prepared local paper-compatible dataset files.",
    )

    bootstrap = subparsers.add_parser(
        "bootstrap-sglang",
        help="Install or refresh the vendored SGLang fork inside WSL.",
    )
    bootstrap.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for bootstrap logs. Defaults to <bench_outputs>/sglang_bootstrap",
    )
    bootstrap.add_argument(
        "--skip-stock-install",
        action="store_true",
        help="Skip the initial `uv pip install sglang` step and only apply the editable vendored install.",
    )
    bootstrap.add_argument(
        "--timeout-minutes",
        type=int,
        default=60,
        help="Per-step timeout for WSL installation commands.",
    )

    paper = subparsers.add_parser(
        "run-paper-local",
        help="Run the local-machine paper reproduction workflow across HF and SGLang baselines.",
    )
    paper.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for paper-local outputs. Defaults to <bench_outputs>/paper_local",
    )
    paper.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small local smoke subset using the configured smoke request limit.",
    )

    dashboard = subparsers.add_parser(
        "serve-dashboard",
        help="Launch the local experiment dashboard backend.",
    )
    dashboard.add_argument(
        "--results-dir",
        default=None,
        help="Optional directory containing benchmark outputs to visualize.",
    )

    serve_sglang = subparsers.add_parser(
        "serve-sglang",
        help="Launch the stock or patched SGLang server for local LoRA experiments.",
    )
    serve_sglang.add_argument(
        "--variant",
        choices=["sglang_stock_lora", "sglang_mobilora"],
        default="sglang_mobilora",
        help="Which server variant to launch.",
    )
    serve_sglang.add_argument(
        "--adapter-count",
        type=int,
        default=None,
        help="How many configured adapters to expose to the server. Defaults to all configured adapters.",
    )
    serve_sglang.add_argument(
        "--detach",
        action="store_true",
        help="Launch the server in the background and write logs under <bench_outputs>/sglang_logs.",
    )
    serve_sglang.add_argument(
        "--print-command",
        action="store_true",
        help="Only print the resolved WSL launch command without executing it.",
    )
    serve_sglang.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Foreground launch timeout or detach submission timeout in milliseconds.",
    )
    return parser


def command_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    backend = args.backend or config.runtime.backend
    manifest = build_prepare_manifest(
        config,
        backend=backend,
        skip_hf_check=args.skip_hf_check,
        download_assets=args.download_assets,
    )
    if args.warm_runtime:
        manifest.runtime_validation.append(build_runtime_validation(config, backend=backend))
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
        request_limit=args.request_limit,
        scenario_filter=args.scenario_filter,
        workload_filter=args.workload_filter,
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


def command_prepare_paper_data(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    manifest_path = prepare_paper_data(config, max_records=args.max_records)
    print(manifest_path)
    return 0


def command_bootstrap_sglang(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else config.paths.bench_outputs / "sglang_bootstrap"
    )
    manifest_path = bootstrap_sglang(
        config,
        output_dir=output_dir,
        install_stock_first=not args.skip_stock_install,
        timeout_ms=args.timeout_minutes * 60 * 1000,
    )
    print(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = manifest.get("results", [])
    if results and int(results[-1].get("returncode", 0)) != 0:
        return int(results[-1]["returncode"])
    return 0


def command_run_paper_local(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    output_dir = (
        Path(args.output_dir) if args.output_dir else config.paths.bench_outputs / "paper_local"
    )
    payload = run_paper_local(config, output_dir=output_dir, smoke=args.smoke)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_serve_dashboard(args: argparse.Namespace) -> int:
    serve_dashboard(args.config, results_dir=args.results_dir)
    return 0


def command_serve_sglang(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    specs = list(config.adapter_specs)
    if args.adapter_count is not None:
        specs = specs[: max(args.adapter_count, 0)]
    aliases = [spec.alias for spec in specs]
    if args.print_command:
        print(default_launch_command(config, args.variant, aliases))
        return 0
    payload = launch_sglang_server(
        config=config,
        variant=args.variant,
        lora_aliases=aliases,
        detach=args.detach,
        timeout_ms=args.timeout_ms,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
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
    if args.command == "prepare-paper-data":
        return command_prepare_paper_data(args)
    if args.command == "bootstrap-sglang":
        return command_bootstrap_sglang(args)
    if args.command == "run-paper-local":
        return command_run_paper_local(args)
    if args.command == "serve-dashboard":
        return command_serve_dashboard(args)
    if args.command == "serve-sglang":
        return command_serve_sglang(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2

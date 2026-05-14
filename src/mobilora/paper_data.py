from __future__ import annotations

import json
import random
from pathlib import Path

from mobilora.types import AppConfig
from mobilora.utils import ensure_directory


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_synthetic_app_trace(config: AppConfig, request_count: int) -> list[dict[str, object]]:
    rng = random.Random(config.bench.random_seed + 2025)
    adapters = list(config.adapter_specs)
    weights = [1.0 / ((index + 1) ** 1.35) for index in range(len(adapters))]
    recent: list[str] = []
    rows: list[dict[str, object]] = []
    for index in range(request_count):
        adapter = rng.choices(adapters, weights=weights, k=1)[0]
        if adapter.app_id not in recent:
            state = "foreground"
        elif recent and recent[-1] == adapter.app_id:
            state = "foreground"
        elif adapter.app_id in recent[-8:]:
            state = "background"
        else:
            state = "killed"
        recent.append(adapter.app_id)
        recent[:] = recent[-20:]
        rows.append(
            {
                "request_id": f"china-telecom-style-{index:04d}",
                "adapter_alias": adapter.alias,
                "adapter_repo": adapter.repo_id,
                "app_id": adapter.app_id,
                "app_state": state,
            }
        )
    return rows


def prepare_paper_data(
    config: AppConfig,
    max_records: int | None = None,
) -> Path:
    prepared_dir = ensure_directory(config.paper.prepared_dir)
    traces_dir = ensure_directory(config.paper.traces_dir)
    record_limit = max_records or max(
        config.paper.smoke_request_count,
        max((scenario.request_count for scenario in config.bench.scenarios), default=0),
    )

    conversation_rows = _read_jsonl(config.bench.dataset_files["conversation"])[:record_limit]
    writing_rows = _read_jsonl(config.bench.dataset_files["writing"])[:record_limit]

    conversation_target = prepared_dir / "sharegpt_compat.jsonl"
    writing_target = prepared_dir / "xsum_compat.jsonl"
    trace_target = traces_dir / "china_telecom_style_trace.jsonl"

    _write_jsonl(conversation_target, conversation_rows)
    _write_jsonl(writing_target, writing_rows)
    _write_jsonl(trace_target, _build_synthetic_app_trace(config, record_limit))

    manifest = {
        "conversation": {
            "source_url": config.paper.conversation_source_url,
            "prepared_path": str(conversation_target),
            "record_count": len(conversation_rows),
            "mode": "sample_compat",
            "note": (
                "Current local paper workflow keeps a lightweight ShareGPT-compatible sample in-repo. "
                "Replace this file with the full public ShareGPT export when available."
            ),
        },
        "writing": {
            "dataset_name": config.paper.writing_dataset_name,
            "dataset_split": config.paper.writing_dataset_split,
            "prepared_path": str(writing_target),
            "record_count": len(writing_rows),
            "mode": "sample_compat",
            "note": "Current local workflow uses an XSum-compatible sample to avoid large downloads during setup.",
        },
        "app_usage": {
            "source_url": config.paper.app_usage_source_url,
            "prepared_path": str(trace_target),
            "record_count": record_limit,
            "mode": "synthetic_from_public_description",
            "note": (
                "The app-usage trace is synthesized from the public Tsinghua/China Telecom dataset description, "
                "with Pareto-skewed adapter popularity and foreground/background/killed state progression."
            ),
        },
    }
    manifest_path = prepared_dir / "paper_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path

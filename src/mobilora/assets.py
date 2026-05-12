from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mobilora.types import AdapterSpec, AppConfig
from mobilora.utils import directory_stats, ensure_directory, repo_cache_name


def base_model_local_dir(config: AppConfig) -> Path:
    return ensure_directory(config.paths.hf_cache / "base_models" / repo_cache_name(config.runtime.base_model))


def adapter_local_dir(config: AppConfig, adapter: AdapterSpec) -> Path:
    return ensure_directory(config.paths.adapters_cache / adapter.alias)


def _sum_remote_file_bytes(model_info: Any) -> int:
    total = 0
    for sibling in getattr(model_info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
    return total


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _snapshot_record(
    repo_id: str,
    local_dir: Path,
    remote_info: Any | None,
    asset_type: str,
    alias: str | None = None,
) -> dict[str, object]:
    stats = directory_stats(local_dir)
    record: dict[str, object] = {
        "asset_type": asset_type,
        "repo_id": repo_id,
        "alias": alias or repo_id.split("/")[-1],
        "local_path": str(local_dir),
        "exists": stats["exists"],
        "file_count": stats["file_count"],
        "total_bytes": stats["total_bytes"],
        "largest_files": stats["largest_files"],
        "downloaded": bool(stats["exists"] and stats["file_count"]),
    }
    if remote_info is not None:
        record["remote_revision"] = getattr(remote_info, "sha", "")
        record["remote_sibling_count"] = len(getattr(remote_info, "siblings", []) or [])
        record["remote_total_bytes"] = _sum_remote_file_bytes(remote_info)
    return record


def _download_snapshot(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=["*.onnx", "*.gguf", "*.msgpack", "*.h5"],
        resume_download=True,
    )


def prepare_hf_assets(
    config: AppConfig,
    download_assets: bool = False,
) -> dict[str, object]:
    from huggingface_hub import HfApi

    api = HfApi()
    base_dir = base_model_local_dir(config)
    base_info = api.model_info(config.runtime.base_model, files_metadata=True)
    base_record = _snapshot_record(
        repo_id=config.runtime.base_model,
        local_dir=base_dir,
        remote_info=base_info,
        asset_type="base_model",
        alias=config.runtime.base_model.split("/")[-1],
    )

    if download_assets:
        resolved_path = _download_snapshot(
            repo_id=config.runtime.base_model,
            local_dir=base_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.safetensors",
                "*.bin",
                "*.py",
                "tokenizer*",
                "vocab*",
                "merges.txt",
            ],
        )
        base_record = _snapshot_record(
            repo_id=config.runtime.base_model,
            local_dir=Path(resolved_path),
            remote_info=base_info,
            asset_type="base_model",
            alias=config.runtime.base_model.split("/")[-1],
        )
        base_record["resolved_path"] = resolved_path

    adapter_records: list[dict[str, object]] = []
    for adapter in config.adapter_specs:
        local_dir = adapter_local_dir(config, adapter)
        remote_info = api.model_info(adapter.repo_id, files_metadata=True)
        record = _snapshot_record(
            repo_id=adapter.repo_id,
            local_dir=local_dir,
            remote_info=remote_info,
            asset_type="adapter",
            alias=adapter.alias,
        )

        if download_assets:
            resolved_path = _download_snapshot(
                repo_id=adapter.repo_id,
                local_dir=local_dir,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.md",
                    "*.safetensors",
                    "*.bin",
                ],
            )
            record = _snapshot_record(
                repo_id=adapter.repo_id,
                local_dir=Path(resolved_path),
                remote_info=remote_info,
                asset_type="adapter",
                alias=adapter.alias,
            )
            record["resolved_path"] = resolved_path

        adapter_config = _read_json_if_exists(local_dir / "adapter_config.json")
        expected_base_model = adapter_config.get("base_model_name_or_path", "")
        record["adapter_base_model_name_or_path"] = expected_base_model
        record["adapter_base_model_matches"] = (
            not expected_base_model or expected_base_model == config.runtime.base_model
        )
        adapter_records.append(record)

    return {
        "base_model": base_record,
        "adapters": adapter_records,
    }

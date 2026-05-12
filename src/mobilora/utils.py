from __future__ import annotations

import hashlib
import math
import os
import statistics
from pathlib import Path
from typing import Iterable


def default_asset_root() -> Path:
    env_value = os.environ.get("MOBILORA_ASSET_ROOT")
    if env_value:
        return Path(env_value)
    if os.name == "nt":
        return Path("D:/MobiLoRA_assets")
    return Path("/mnt/d/MobiLoRA_assets")


def resolve_asset_subdir(root: Path, maybe_relative: str | os.PathLike[str]) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return root / path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def percentile(values: Iterable[float], q: float) -> float:
    samples = sorted(values)
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    rank = max(0.0, min(1.0, q)) * (len(samples) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(samples[low])
    weight = rank - low
    return float(samples[low] * (1.0 - weight) + samples[high] * weight)


def median(values: Iterable[float]) -> float:
    samples = list(values)
    if not samples:
        return 0.0
    return float(statistics.median(samples))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    prefix_len = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix_len += 1
    return prefix_len


def safe_filename(text: str) -> str:
    invalid = '<>:"/\\|?*'
    sanitized = "".join("_" if char in invalid else char for char in text)
    return sanitized.strip().strip(".") or "output"

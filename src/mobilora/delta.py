from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from mobilora.types import DeltaConfig, DeltaEncodingStats
from mobilora.utils import clamp


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_vec = np.asarray(tuple(left), dtype=np.float32)
    right_vec = np.asarray(tuple(right), dtype=np.float32)
    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def select_error_bound(similarity: float, config: DeltaConfig) -> float:
    if similarity >= config.high_similarity_threshold:
        return config.high_similarity_error_bound
    if similarity >= config.medium_similarity_threshold:
        return config.medium_similarity_error_bound
    return config.low_similarity_error_bound


def quantize_delta(anchor: np.ndarray, target: np.ndarray, error_bound: float) -> np.ndarray:
    scale = max(2.0 * math.log1p(error_bound), 1.0e-8)
    quantized = np.floor((target - anchor) / scale + 0.5)
    max_abs = np.max(np.abs(quantized)) if quantized.size else 0
    if max_abs <= 127:
        return quantized.astype(np.int8)
    if max_abs <= 32767:
        return quantized.astype(np.int16)
    return quantized.astype(np.int32)


def encode_delta_payload(
    anchor_layers: tuple[tuple[float, ...], ...],
    target_layers: tuple[tuple[float, ...], ...],
    config: DeltaConfig,
) -> DeltaEncodingStats:
    if not anchor_layers or not target_layers:
        return DeltaEncodingStats(compression_ratio=1.0, avg_similarity=1.0, avg_error_bound=0.0)

    raw_bytes = 0
    encoded_bytes = 0
    similarities: list[float] = []
    error_bounds: list[float] = []

    for anchor_layer, target_layer in zip(anchor_layers, target_layers):
        anchor = np.asarray(anchor_layer, dtype=np.float32)
        target = np.asarray(target_layer, dtype=np.float32)
        similarity = cosine_similarity(anchor_layer, target_layer)
        error_bound = select_error_bound(similarity, config)
        quantized = quantize_delta(anchor, target, error_bound)
        raw_bytes += int(target.nbytes)
        encoded_bytes += int(quantized.nbytes)
        similarities.append(similarity)
        error_bounds.append(error_bound)

    ratio = raw_bytes / max(encoded_bytes, 1)
    return DeltaEncodingStats(
        compression_ratio=clamp(ratio, 1.0, 64.0),
        avg_similarity=float(sum(similarities) / len(similarities)),
        avg_error_bound=float(sum(error_bounds) / len(error_bounds)),
    )

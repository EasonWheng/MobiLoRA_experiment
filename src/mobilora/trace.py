from __future__ import annotations

import json
import random
from pathlib import Path

from mobilora.types import AdapterSpec, RequestRecord, ScenarioConfig


def load_prompts(dataset_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    if not records:
        raise ValueError(f"No records found in dataset file: {dataset_path}")
    return records


def build_trace(
    workload: str,
    dataset_path: Path,
    adapters: tuple[AdapterSpec, ...],
    scenario: ScenarioConfig,
    seed: int,
) -> list[RequestRecord]:
    rng = random.Random(seed)
    prompts = load_prompts(dataset_path)

    adapter_weights = [1.0 / ((index + 1) ** 1.4) for index in range(len(adapters))]
    requests: list[RequestRecord] = []
    for index in range(scenario.request_count):
        payload = prompts[index % len(prompts)]
        adapter = rng.choices(adapters, weights=adapter_weights, k=1)[0]
        requests.append(
            RequestRecord(
                request_id=f"{scenario.name}-{workload}-{index:04d}",
                workload=workload,
                adapter_alias=adapter.alias,
                adapter_repo=adapter.repo_id,
                app_id=adapter.app_id,
                app_state="foreground",
                prompt=str(payload["prompt"]),
                reference=str(payload.get("reference", "")),
            )
        )
    return requests

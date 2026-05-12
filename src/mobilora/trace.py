from __future__ import annotations

import json
import random
from pathlib import Path

from mobilora.types import AdapterSpec, RequestRecord, ScenarioConfig


def _conversation_prompt(base_prompt: str, history: list[str]) -> str:
    prefix = (
        "System: You are a helpful mobile assistant.\n"
        "Mode: multi-app continuation.\n"
        "Conversation:\n"
    )
    history_block = "\n".join(history[-2:])
    if history_block:
        return f"{prefix}{history_block}\nUser: {base_prompt}"
    return f"{prefix}User: {base_prompt}"


def _writing_prompt(base_prompt: str, history: list[str]) -> str:
    prefix = (
        "Instruction: Summarize the following article for a mobile user.\n"
        "Output: concise bullet-free abstract.\n"
        "Article:\n"
    )
    if history:
        prior_lead = history[-1].splitlines()[-1]
        return f"{prefix}{prior_lead}\n\nFollow-up article:\n{base_prompt}"
    return f"{prefix}{base_prompt}"


def _compose_prompt(workload: str, base_prompt: str, history: list[str]) -> str:
    if workload == "writing":
        return _writing_prompt(base_prompt, history)
    return _conversation_prompt(base_prompt, history)


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
    session_histories: dict[str, list[str]] = {}

    adapter_weights = [1.0 / ((index + 1) ** 1.4) for index in range(len(adapters))]
    requests: list[RequestRecord] = []
    for index in range(scenario.request_count):
        payload = prompts[index % len(prompts)]
        adapter = rng.choices(adapters, weights=adapter_weights, k=1)[0]
        history = session_histories.setdefault(adapter.app_id, [])
        prompt_text = _compose_prompt(workload, str(payload["prompt"]), history)
        requests.append(
            RequestRecord(
                request_id=f"{scenario.name}-{workload}-{index:04d}",
                workload=workload,
                adapter_alias=adapter.alias,
                adapter_repo=adapter.repo_id,
                app_id=adapter.app_id,
                app_state="foreground",
                prompt=prompt_text,
                reference=str(payload.get("reference", "")),
            )
        )
        history.append(f"User: {payload['prompt']}")
        if len(history) > 4:
            del history[:-4]
    return requests

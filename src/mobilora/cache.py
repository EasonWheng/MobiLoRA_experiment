from __future__ import annotations

import math
from dataclasses import dataclass, field

from mobilora.types import CacheEntry, EvictionConfig
from mobilora.utils import common_prefix_length
from mobilora.delta import cosine_similarity


@dataclass(slots=True)
class PrefixTreeNode:
    children: dict[int, "PrefixTreeNode"] = field(default_factory=dict)
    entry_ids: set[str] = field(default_factory=set)


class PrefixTree:
    def __init__(self) -> None:
        self.root = PrefixTreeNode()

    def insert(self, token_ids: tuple[int, ...], entry_id: str) -> None:
        node = self.root
        for token_id in token_ids:
            node = node.children.setdefault(token_id, PrefixTreeNode())
            node.entry_ids.add(entry_id)

    def remove(self, token_ids: tuple[int, ...], entry_id: str) -> None:
        node = self.root
        for token_id in token_ids:
            next_node = node.children.get(token_id)
            if next_node is None:
                return
            next_node.entry_ids.discard(entry_id)
            node = next_node

    def longest_prefix_candidates(self, token_ids: tuple[int, ...]) -> set[str]:
        node = self.root
        candidates: set[str] = set()
        for token_id in token_ids:
            node = node.children.get(token_id)
            if node is None:
                break
            candidates = set(node.entry_ids)
        return candidates


class CachePool:
    def __init__(self, budget_mb: int, eviction: EvictionConfig) -> None:
        self.budget_mb = float(budget_mb)
        self.eviction = eviction
        self.tree = PrefixTree()
        self.entries: dict[str, CacheEntry] = {}
        self.evicted_entries = 0
        self.accepted_entries = 0

    @property
    def total_size_mb(self) -> float:
        return sum(entry.stored_size_mb for entry in self.entries.values())

    def update_states(self, current_app: str, recent_apps: set[str]) -> None:
        for entry in self.entries.values():
            if entry.app_id == current_app:
                entry.state_label = "foreground"
            elif entry.app_id in recent_apps:
                entry.state_label = "background"
            else:
                entry.state_label = "killed"

    def state_score(self, state_label: str) -> float:
        mapping = {
            "foreground": self.eviction.foreground_score,
            "background": self.eviction.background_score,
            "killed": self.eviction.killed_score,
        }
        return mapping.get(state_label, self.eviction.killed_score)

    def utility(self, entry: CacheEntry, step: int) -> float:
        age = max(0, step - entry.last_access_step)
        recency_score = 1.0 / (1.0 + age)
        length_score = math.log1p(entry.token_len)
        return (
            self.eviction.lambda_state * self.state_score(entry.state_label)
            + self.eviction.lambda_recency * recency_score
            + self.eviction.lambda_length * length_score
        )

    def find_best_anchor(
        self,
        token_ids: tuple[int, ...],
        shallow_key: tuple[float, ...],
        adapter_alias: str,
        allow_cross_adapter: bool,
    ) -> CacheEntry | None:
        candidates = [self.entries[item] for item in self.tree.longest_prefix_candidates(token_ids)]
        if not allow_cross_adapter:
            candidates = [item for item in candidates if item.adapter_alias == adapter_alias]
        if not candidates:
            return None

        best_prefix = max(common_prefix_length(item.token_ids, token_ids) for item in candidates)
        longest_prefix = [
            item for item in candidates if common_prefix_length(item.token_ids, token_ids) == best_prefix
        ]
        if not longest_prefix or best_prefix == 0:
            return None

        return max(longest_prefix, key=lambda item: cosine_similarity(item.shallow_key, shallow_key))

    def touch(self, entry_id: str, step: int) -> None:
        entry = self.entries[entry_id]
        entry.last_access_step = step
        entry.hit_count += 1

    def _insert(self, entry: CacheEntry) -> None:
        self.entries[entry.entry_id] = entry
        self.tree.insert(entry.token_ids, entry.entry_id)
        self.accepted_entries += 1

    def _delete(self, entry_id: str) -> None:
        entry = self.entries.pop(entry_id)
        self.tree.remove(entry.token_ids, entry_id)
        self.evicted_entries += 1

    def admit(
        self,
        entry: CacheEntry,
        step: int,
        current_app: str,
        recent_apps: set[str],
        context_aware: bool,
    ) -> tuple[bool, list[str]]:
        self.update_states(current_app, recent_apps)

        if self.total_size_mb + entry.stored_size_mb <= self.budget_mb:
            self._insert(entry)
            return True, []

        shadow_entries = dict(self.entries)
        shadow_entries[entry.entry_id] = entry
        pending_size = sum(item.stored_size_mb for item in shadow_entries.values())
        evicted: list[str] = []

        while pending_size > self.budget_mb and shadow_entries:
            if context_aware:
                victim_id = min(shadow_entries, key=lambda item: self.utility(shadow_entries[item], step))
            else:
                victim_id = min(shadow_entries, key=lambda item: shadow_entries[item].last_access_step)
            victim = shadow_entries.pop(victim_id)
            pending_size -= victim.stored_size_mb
            evicted.append(victim_id)

        if entry.entry_id in evicted:
            return False, evicted

        for victim_id in evicted:
            if victim_id in self.entries:
                self._delete(victim_id)

        self._insert(entry)
        return True, evicted

from __future__ import annotations

import random
from collections import deque

from backend.model.graph_builder import GNNExperience


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, historical_fraction: float = 0.2):
        recent_capacity = int(capacity * (1.0 - historical_fraction))
        historical_capacity = max(capacity - recent_capacity, 1)
        self._recent: deque[GNNExperience] = deque(maxlen=max(recent_capacity, 1))
        self._historical: deque[GNNExperience] = deque(maxlen=historical_capacity)
        self._historical_sample_rate: float = 0.05

    def add(self, experiences: list[GNNExperience]):
        for exp in experiences:
            self._recent.append(exp)
            if random.random() < self._historical_sample_rate:
                self._historical.append(exp)

    def sample(self, batch_size: int) -> list[GNNExperience]:
        if not len(self):
            return []
        recent_target = batch_size - min(batch_size // 5, len(self._historical))
        recent_sample = random.sample(list(self._recent), min(recent_target, len(self._recent)))
        historical_target = min(batch_size - len(recent_sample), len(self._historical))
        historical_sample = (
            random.sample(list(self._historical), historical_target)
            if historical_target
            else []
        )
        combined = recent_sample + historical_sample
        if len(combined) < batch_size:
            pool = list(self._recent) + list(self._historical)
            needed = min(batch_size - len(combined), len(pool))
            combined.extend(random.sample(pool, needed))
        return combined

    def clear(self):
        self._recent.clear()
        self._historical.clear()

    def serialize(self) -> dict[str, list[GNNExperience]]:
        return {
            "recent": list(self._recent),
            "historical": list(self._historical),
        }

    def load(self, experiences: list[GNNExperience] | dict[str, list[GNNExperience]]):
        self.clear()
        if isinstance(experiences, dict):
            self._recent.extend(experiences.get("recent", []))
            self._historical.extend(experiences.get("historical", []))
            return
        self.add(experiences)

    def turn_number_histogram(self, buckets: int = 10) -> list[dict[str, int]]:
        all_items = list(self._recent) + list(self._historical)
        if not all_items:
            return []
        max_turn = max(exp.turn_number for exp in all_items) + 1
        width = max(max_turn // buckets, 1)
        histogram: dict[int, int] = {}
        for exp in all_items:
            bucket = exp.turn_number // width
            histogram[bucket] = histogram.get(bucket, 0) + 1
        return [
            {"bucket": bucket, "count": count}
            for bucket, count in sorted(histogram.items())
        ]

    def __len__(self) -> int:
        return len(self._recent) + len(self._historical)

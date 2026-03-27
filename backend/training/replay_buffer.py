from __future__ import annotations

import random
from collections import deque

from backend.model.graph_builder import GNNExperience


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        historical_fraction: float = 0.2,
        tactical_fraction: float = 0.35,
    ):
        recent_capacity = int(capacity * (1.0 - historical_fraction))
        historical_capacity = max(capacity - recent_capacity, 1)
        self._recent: deque[GNNExperience] = deque(maxlen=max(recent_capacity, 1))
        self._historical: deque[GNNExperience] = deque(maxlen=historical_capacity)
        self._historical_sample_rate: float = 0.05
        self._tactical_fraction = min(max(tactical_fraction, 0.0), 0.9)

    def _is_tactical(self, experience: GNNExperience) -> bool:
        aux = getattr(experience, "aux_targets", None)
        if aux is None or len(aux) < 4:
            return False
        return bool(aux[0] > 0.5 or aux[1] > 0.5 or aux[2] > 0.0 or aux[3] > 0.0)

    def add(self, experiences: list[GNNExperience]):
        for exp in experiences:
            self._recent.append(exp)
            if random.random() < self._historical_sample_rate:
                self._historical.append(exp)

    def sample(self, batch_size: int) -> list[GNNExperience]:
        if not len(self):
            return []
        recent_items = list(self._recent)
        historical_items = list(self._historical)
        tactical_pool = [exp for exp in recent_items if self._is_tactical(exp)]
        tactical_target = min(int(batch_size * self._tactical_fraction), len(tactical_pool))
        tactical_sample = random.sample(tactical_pool, tactical_target) if tactical_target else []

        recent_target = batch_size - len(tactical_sample) - min(batch_size // 5, len(historical_items))
        tactical_ids = {id(exp) for exp in tactical_sample}
        non_tactical_recent = [exp for exp in recent_items if id(exp) not in tactical_ids]
        recent_sample = random.sample(non_tactical_recent, min(recent_target, len(non_tactical_recent)))
        historical_target = min(batch_size - len(recent_sample), len(self._historical))
        historical_sample = (
            random.sample(historical_items, historical_target)
            if historical_target
            else []
        )
        combined = tactical_sample + recent_sample + historical_sample
        if len(combined) < batch_size:
            pool = recent_items + historical_items
            needed = min(batch_size - len(combined), len(pool))
            combined.extend(random.sample(pool, needed))
        return combined

    def clear(self):
        self._recent.clear()
        self._historical.clear()

    def _downsample(self, items: list[GNNExperience], max_items: int | None, keep_recent: bool) -> list[GNNExperience]:
        if max_items is None or max_items <= 0 or len(items) <= max_items:
            return list(items)
        if keep_recent:
            return list(items[-max_items:])
        if max_items == 1:
            return [items[-1]]
        step = (len(items) - 1) / float(max_items - 1)
        selected = []
        seen: set[int] = set()
        for index in range(max_items):
            raw_idx = int(round(index * step))
            clamped = max(0, min(len(items) - 1, raw_idx))
            while clamped in seen and clamped < len(items) - 1:
                clamped += 1
            seen.add(clamped)
            selected.append(items[clamped])
        return selected

    def serialize(
        self,
        max_recent: int | None = None,
        max_historical: int | None = None,
    ) -> dict[str, list[GNNExperience]]:
        return {
            "recent": self._downsample(list(self._recent), max_recent, keep_recent=True),
            "historical": self._downsample(list(self._historical), max_historical, keep_recent=False),
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

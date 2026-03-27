from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from backend.config import ModelConfig
from backend.model.inference import ModelAgent
from backend.model.network import HexGNNModel


@dataclass
class ModelSnapshot:
    path: str
    iteration: int
    state_dict: dict
    is_reference: bool = False


class OpponentPool:
    def __init__(self, model_config: ModelConfig, max_size: int = 15, device: str = "cpu"):
        self.model_config = model_config
        self.snapshots: list[ModelSnapshot] = []
        self.max_size = max_size
        self.device = device

    def add_snapshot(
        self,
        model: HexGNNModel,
        iteration: int,
        path: str,
        is_reference: bool = False,
    ):
        snapshot = ModelSnapshot(
            path=path,
            iteration=iteration,
            state_dict=copy.deepcopy(model.state_dict()),
            is_reference=is_reference,
        )
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def _sample_snapshot(self, strategy: str = "uniform") -> ModelSnapshot | None:
        if not self.snapshots:
            return None
        if strategy == "recent":
            weights = np.exp(np.linspace(0.0, 1.0, len(self.snapshots)))
            weights = weights / weights.sum()
            return self.snapshots[int(np.random.choice(len(self.snapshots), p=weights))]
        if strategy == "oldest":
            return self.snapshots[0]
        return random.choice(self.snapshots)

    def sample_snapshot(self, strategy: str = "uniform") -> ModelSnapshot | None:
        return self._sample_snapshot(strategy)

    def _agent_from_snapshot(self, snapshot: ModelSnapshot) -> ModelAgent:
        model = HexGNNModel(self.model_config)
        model.load_state_dict(snapshot.state_dict, strict=False)
        model.to(self.device)
        return ModelAgent(model, self.device)

    def agent_from_snapshot(self, snapshot: ModelSnapshot) -> ModelAgent:
        return self._agent_from_snapshot(snapshot)

    def sample_opponent(self, strategy: str = "uniform") -> ModelAgent | None:
        snapshot = self._sample_snapshot(strategy)
        if snapshot is None:
            return None
        return self._agent_from_snapshot(snapshot)

    def get_reference_model(self) -> ModelAgent | None:
        reference = next((snap for snap in reversed(self.snapshots) if snap.is_reference), None)
        if reference is None:
            return None
        return self._agent_from_snapshot(reference)

    def list_snapshots(self) -> list[dict]:
        return [
            {
                "path": snapshot.path,
                "iteration": snapshot.iteration,
                "is_reference": snapshot.is_reference,
            }
            for snapshot in self.snapshots
        ]

    def serialize(self) -> list[dict]:
        return [
            {
                "path": snapshot.path,
                "iteration": snapshot.iteration,
                "state_dict": copy.deepcopy(snapshot.state_dict),
                "is_reference": snapshot.is_reference,
            }
            for snapshot in self.snapshots
        ]

    def load(self, payload: list[dict]):
        self.snapshots = [
            ModelSnapshot(
                path=item["path"],
                iteration=item["iteration"],
                state_dict=item["state_dict"],
                is_reference=item.get("is_reference", False),
            )
            for item in payload
        ]

    def __len__(self) -> int:
        return len(self.snapshots)

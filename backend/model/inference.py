from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.game.game_state import Coord, GameState
from backend.model.graph_builder import GraphBuilder
from backend.training.baselines import forced_move_policy

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class InferenceResult:
    coord: Coord
    policy: dict[Coord, float]
    entropy: float
    raw_coord: Optional[Coord] = None
    was_overridden: bool = False
    override_reason: Optional[str] = None
    forced_policy: Optional[dict[Coord, float]] = None


class ModelAgent:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.graph_builder = GraphBuilder()

    def select_move(
        self,
        state: GameState,
        temperature: float = 1.0,
        greedy: bool = False,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: float = 0.25,
        prior_policy: Optional[dict[Coord, float]] = None,
        prior_weight: float = 0.0,
        allow_tactical_override: bool = False,
    ) -> InferenceResult:
        if torch is None:
            raise RuntimeError("torch is required for inference")
        data = self.graph_builder.build_data(state)
        data = data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(data)
        legal_mask = data.legal_mask.detach().cpu().numpy()
        coords = data.node_coords.detach().cpu().numpy()
        legal_logits = logits.detach().cpu().numpy()[legal_mask]
        legal_coords = [tuple(map(int, coord)) for coord in coords[legal_mask]]
        scaled = legal_logits if greedy else legal_logits / max(temperature, 1e-6)
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        if prior_policy and prior_weight > 0.0:
            prior = np.asarray([prior_policy.get(coord, 0.0) for coord in legal_coords], dtype=np.float64)
            prior_sum = prior.sum()
            if prior_sum > 0:
                prior = prior / prior_sum
                probs = ((1.0 - prior_weight) * probs) + (prior_weight * prior)
                probs = probs / probs.sum()
        if dirichlet_alpha is not None and not greedy and len(probs) > 1:
            noise = np.random.dirichlet(np.full(len(probs), dirichlet_alpha))
            probs = ((1.0 - dirichlet_epsilon) * probs) + (dirichlet_epsilon * noise)
            probs = probs / probs.sum()
        move_idx = int(np.argmax(probs)) if greedy else int(np.random.choice(len(legal_coords), p=probs))
        raw_coord = legal_coords[move_idx]
        entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum())
        forced = forced_move_policy(state) if allow_tactical_override else None
        selected_coord = raw_coord
        was_overridden = False
        override_reason = None
        forced_policy = forced.policy if forced is not None else None
        if forced is not None and raw_coord not in forced.forced_moves:
            selected_coord = forced.coord
            was_overridden = True
            override_reason = forced.reason
        return InferenceResult(
            coord=selected_coord,
            policy={coord: float(prob) for coord, prob in zip(legal_coords, probs)},
            entropy=entropy,
            raw_coord=raw_coord,
            was_overridden=was_overridden,
            override_reason=override_reason,
            forced_policy=forced_policy,
        )

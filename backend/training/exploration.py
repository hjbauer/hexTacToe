from __future__ import annotations

import numpy as np

from backend.config import ExplorationConfig


class ExplorationScheduler:
    def __init__(self, config: ExplorationConfig):
        self.config = config

    def get_temperature(self, iteration: int) -> float:
        warmup = max(self.config.warmup_iterations, 1)
        progress = min(max(iteration, 0) / warmup, 1.0)
        return self.config.initial_temperature + (
            self.config.final_temperature - self.config.initial_temperature
        ) * progress

    def apply_dirichlet_noise(self, policy_probs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        legal_count = int(legal_mask.sum())
        if not self.config.use_dirichlet_noise or legal_count <= 1:
            return policy_probs
        legal_probs = policy_probs[legal_mask]
        noise = np.random.dirichlet(np.full(legal_count, self.config.dirichlet_alpha))
        mixed = (
            (1.0 - self.config.dirichlet_epsilon) * legal_probs
            + self.config.dirichlet_epsilon * noise
        )
        output = np.array(policy_probs, copy=True)
        output[legal_mask] = mixed / mixed.sum()
        return output

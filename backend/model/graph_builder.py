from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

from backend.game.game_state import Coord, GameState
from backend.game.hex_coord import centroid, hex_distance
from backend.game.rules import get_legal_moves

try:
    import torch
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover
    torch = None
    Data = None


@dataclass
class GNNExperience:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_coords: np.ndarray
    is_legal: np.ndarray
    mcts_probs: np.ndarray
    outcome: float
    turn_number: int
    aux_targets: Optional[np.ndarray] = None


class GraphBuilder:
    def __init__(self, max_distance_norm: float = 8.0):
        self.max_distance_norm = max_distance_norm

    @staticmethod
    @lru_cache(maxsize=18)
    def _edge_offsets() -> tuple[tuple[int, int, float], ...]:
        offsets: list[tuple[int, int, float]] = []
        for dq in range(-2, 3):
            for dr in range(-2, 3):
                if dq == 0 and dr == 0:
                    continue
                distance = hex_distance(0, 0, dq, dr)
                if distance <= 2:
                    offsets.append((dq, dr, distance / 2.0))
        offsets.sort()
        return tuple(offsets)

    def _node_features(
        self,
        state: GameState,
        node_coords: list[Coord],
        legal_moves: set[Coord],
    ) -> np.ndarray:
        occupied = list(state.all_hexes)
        centroid_q, centroid_r = centroid(occupied)
        placements_remaining = state.placements_remaining_this_turn / 2.0
        turn_norm = state.turn_number / 200.0
        hexes_required_norm = state.hexes_required_this_turn / 6.0
        current_player_is_red = 1.0 if state.current_player == "red" else 0.0

        features: list[list[float]] = []
        for q, r in node_coords:
            is_red = 1.0 if (q, r) in state.red_hexes else 0.0
            is_blue = 1.0 if (q, r) in state.blue_hexes else 0.0
            is_empty_legal = 1.0 if (q, r) in legal_moves else 0.0
            is_last_move = 1.0 if (q, r) in state.last_move_hexes else 0.0
            if is_red or is_blue:
                nearest_norm = 0.0
            else:
                occupied_distances = [
                    hex_distance(q, r, oq, or_)
                    for oq, or_ in occupied
                ] or [0]
                nearest_norm = min(occupied_distances) / self.max_distance_norm
            features.append(
                [
                    (q - centroid_q) / self.max_distance_norm,
                    (r - centroid_r) / self.max_distance_norm,
                    is_red,
                    is_blue,
                    is_empty_legal,
                    is_last_move,
                    current_player_is_red,
                    placements_remaining,
                    turn_norm,
                    hexes_required_norm,
                    nearest_norm,
                ]
            )
        return np.asarray(features, dtype=np.float32)

    def _edge_index_and_attr(self, node_coords: list[Coord]) -> tuple[np.ndarray, np.ndarray]:
        coord_to_index = {coord: index for index, coord in enumerate(node_coords)}
        edges: list[list[int]] = []
        edge_attr: list[list[float]] = []
        for src, (sq, sr) in enumerate(node_coords):
            for dq, dr, distance_norm in self._edge_offsets():
                dst = coord_to_index.get((sq + dq, sr + dr))
                if dst is None:
                    continue
                edges.append([src, dst])
                edge_attr.append([distance_norm])
        if not edges:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, 1), dtype=np.float32),
            )
        return (
            np.asarray(edges, dtype=np.int64).T,
            np.asarray(edge_attr, dtype=np.float32),
        )

    @staticmethod
    @lru_cache(maxsize=131072)
    def _template_for_state(
        state: GameState,
        max_distance_norm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        builder = GraphBuilder(max_distance_norm=max_distance_norm)
        legal_moves = get_legal_moves(state)
        occupied = list(state.all_hexes)
        node_coords = sorted(set(occupied) | set(legal_moves))
        node_features = builder._node_features(state, node_coords, set(legal_moves))
        edge_index, edge_attr = builder._edge_index_and_attr(node_coords)
        is_legal = np.asarray([(coord in legal_moves) for coord in node_coords], dtype=bool)
        node_features.setflags(write=False)
        edge_index.setflags(write=False)
        edge_attr.setflags(write=False)
        is_legal.setflags(write=False)
        node_coords_array = np.asarray(node_coords, dtype=np.int16)
        node_coords_array.setflags(write=False)
        return (
            node_features,
            edge_index,
            edge_attr,
            node_coords_array,
            is_legal,
        )

    def build_experience(
        self,
        state: GameState,
        policy_targets: Optional[dict[Coord, float]] = None,
        outcome: float = 0.0,
        aux_targets: Optional[np.ndarray] = None,
    ) -> GNNExperience:
        node_features, edge_index, edge_attr, node_coords, is_legal = self._template_for_state(
            state,
            self.max_distance_norm,
        )
        mcts_probs = np.zeros(len(node_coords), dtype=np.float32)
        if policy_targets:
            for idx, coord in enumerate(node_coords):
                coord_key = (int(coord[0]), int(coord[1]))
                if coord_key in policy_targets:
                    mcts_probs[idx] = float(policy_targets[coord_key])
        elif is_legal.any():
            mcts_probs[is_legal] = 1.0 / float(is_legal.sum())
        return GNNExperience(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_coords=np.asarray(node_coords, dtype=np.int16),
            is_legal=is_legal,
            mcts_probs=mcts_probs,
            outcome=float(outcome),
            turn_number=state.turn_number,
            aux_targets=(
                np.asarray(aux_targets, dtype=np.float32)
                if aux_targets is not None
                else np.zeros(4, dtype=np.float32)
            ),
        )

    def build_data(self, state: GameState):
        if torch is None or Data is None:
            raise RuntimeError("torch and torch_geometric are required for graph inference")
        exp = self.build_experience(state)
        return experience_to_data(exp)


def experience_to_data(experience: GNNExperience):
    if torch is None or Data is None:
        raise RuntimeError("torch and torch_geometric are required for graph inference")
    aux_targets = getattr(experience, "aux_targets", None)
    return Data(
        x=torch.tensor(experience.node_features, dtype=torch.float32),
        edge_index=torch.tensor(experience.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(experience.edge_attr, dtype=torch.float32),
        legal_mask=torch.tensor(experience.is_legal, dtype=torch.bool),
        policy_target=torch.tensor(experience.mcts_probs, dtype=torch.float32),
        outcome=torch.tensor([experience.outcome], dtype=torch.float32),
        aux_target=(
            torch.tensor(aux_targets, dtype=torch.float32)
            if aux_targets is not None
            else torch.zeros(4, dtype=torch.float32)
        ),
        node_coords=torch.tensor(experience.node_coords, dtype=torch.int16),
    )


def color_swap(experience: GNNExperience) -> GNNExperience:
    swapped = np.array(experience.node_features, copy=True)
    swapped[:, [2, 3]] = swapped[:, [3, 2]]
    swapped[:, 6] = 1.0 - swapped[:, 6]
    swapped_aux = np.array(getattr(experience, "aux_targets", np.zeros(4, dtype=np.float32)), copy=True)
    if swapped_aux.shape[0] >= 4:
        swapped_aux[[0, 1]] = swapped_aux[[1, 0]]
        swapped_aux[[2, 3]] = swapped_aux[[3, 2]]
    return GNNExperience(
        node_features=swapped,
        edge_index=np.array(experience.edge_index, copy=True),
        edge_attr=np.array(experience.edge_attr, copy=True),
        node_coords=np.array(experience.node_coords, copy=True),
        is_legal=np.array(experience.is_legal, copy=True),
        mcts_probs=np.array(experience.mcts_probs, copy=True),
        outcome=-float(experience.outcome),
        turn_number=experience.turn_number,
        aux_targets=swapped_aux,
    )


def _rotate_axial(q: float, r: float, steps: int) -> tuple[float, float]:
    steps = steps % 6
    for _ in range(steps):
        q, r = -r, q + r
    return q, r


def rotate_experience(experience: GNNExperience, steps: int) -> GNNExperience:
    if steps % 6 == 0:
        return GNNExperience(
            node_features=np.array(experience.node_features, copy=True),
            edge_index=np.array(experience.edge_index, copy=True),
            edge_attr=np.array(experience.edge_attr, copy=True),
            node_coords=np.array(experience.node_coords, copy=True),
            is_legal=np.array(experience.is_legal, copy=True),
            mcts_probs=np.array(experience.mcts_probs, copy=True),
            outcome=float(experience.outcome),
            turn_number=experience.turn_number,
            aux_targets=np.array(getattr(experience, "aux_targets", np.zeros(4, dtype=np.float32)), copy=True),
        )

    rotated_features = np.array(experience.node_features, copy=True)
    rotated_coords = np.array(experience.node_coords, copy=True)
    for index, (q, r) in enumerate(experience.node_coords):
        rq, rr = _rotate_axial(int(q), int(r), steps)
        rotated_coords[index, 0] = int(rq)
        rotated_coords[index, 1] = int(rr)
    for index, (q_norm, r_norm) in enumerate(experience.node_features[:, :2]):
        rq_norm, rr_norm = _rotate_axial(float(q_norm), float(r_norm), steps)
        rotated_features[index, 0] = rq_norm
        rotated_features[index, 1] = rr_norm
    return GNNExperience(
        node_features=rotated_features,
        edge_index=np.array(experience.edge_index, copy=True),
        edge_attr=np.array(experience.edge_attr, copy=True),
        node_coords=rotated_coords,
        is_legal=np.array(experience.is_legal, copy=True),
        mcts_probs=np.array(experience.mcts_probs, copy=True),
        outcome=float(experience.outcome),
        turn_number=experience.turn_number,
        aux_targets=np.array(getattr(experience, "aux_targets", np.zeros(4, dtype=np.float32)), copy=True),
    )


def experiences_to_batch(experiences: Iterable[GNNExperience]):
    if torch is None or Data is None:
        raise RuntimeError("torch and torch_geometric are required for batching")
    from torch_geometric.data import Batch

    return Batch.from_data_list([experience_to_data(exp) for exp in experiences])

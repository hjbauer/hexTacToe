from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from backend.config import TrainingConfig
from backend.game.game_state import GameState
from backend.game.hex_coord import hex_distance
from backend.game.rules import apply_move
from backend.game.rules import get_legal_moves
from backend.model.graph_builder import GNNExperience, GraphBuilder, color_swap, rotate_experience
from backend.model.inference import ModelAgent
from backend.model.network import HexGNNModel
from backend.training.baselines import (
    RandomBaseline,
    adjacent_friendly_count,
    average_nearest_friendly_distance,
    connected_components,
    forced_move_policy,
    heuristic_policy,
    immediate_winning_moves,
    longest_line_for_player,
    next_turn_winning_plans,
    would_win_if_played,
)
from backend.training.exploration import ExplorationScheduler
from backend.training.opponent_pool import OpponentPool

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class SelfPlayResult:
    experiences: list[GNNExperience]
    winner: str | None
    candidate_color: str
    move_count: int
    entropy_warning: bool
    forced_override_count: int
    model_decision_count: int
    tactical_opportunity_count: int
    tactical_blunder_count: int
    win_opportunity_count: int
    missed_win_count: int
    block_opportunity_count: int
    missed_block_count: int
    final_snapshot: dict | None = None


@dataclass
class SelfPlayJob:
    config: TrainingConfig
    iteration: int
    game_id: str
    candidate_label: str
    candidate_state_dict: dict
    opponent_label: str
    opponent_kind: str
    opponent_state_dict: dict | None = None
    progress_path: str | None = None


@dataclass
class CandidateTurnRecord:
    state: GameState
    move: tuple[int, int]
    experience: GNNExperience
    immediate_reward: float


def augment_experiences(
    experiences: list[GNNExperience],
    rotation_samples: int = 0,
) -> list[GNNExperience]:
    augmented: list[GNNExperience] = []
    for exp in experiences:
        augmented.append(exp)
        augmented.append(color_swap(exp))
        for _ in range(max(rotation_samples, 0)):
            augmented.append(rotate_experience(exp, random.randint(1, 5)))
    return augmented


class SelfPlayWorker:
    def __init__(
        self,
        candidate_agent: ModelAgent,
        reference_agent: Optional[ModelAgent],
        opponent_pool: OpponentPool,
        config: TrainingConfig,
        spectate_callback: Optional[Callable[[dict], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
        opponent_override: Optional[tuple[str, object]] = None,
        candidate_label: str = "model-1",
    ):
        self.candidate_agent = candidate_agent
        self.reference_agent = reference_agent
        self.opponent_pool = opponent_pool
        self.config = config
        self.random_baseline = RandomBaseline()
        self.graph_builder = GraphBuilder()
        self.scheduler = ExplorationScheduler(config.exploration)
        self.spectate_callback = spectate_callback
        self.stop_requested = stop_requested
        self.opponent_override = opponent_override
        self.candidate_label = candidate_label

    def _turn_limit(self, iteration: int) -> int:
        return self.config.turn_limit_for_iteration(iteration)

    def _sample_opponent(self):
        weights = self.config.opponent_sampling_weights
        choices = ["reference", "pool", "random"]
        selection = random.choices(choices, weights=[weights[k] for k in choices], k=1)[0]
        if selection == "reference" and self.reference_agent is not None:
            return selection, self.reference_agent
        if selection == "pool":
            strategy_weights = self.config.pool_strategy_weights
            strategy_choices = list(strategy_weights.keys())
            strategy = random.choices(
                strategy_choices,
                weights=[strategy_weights[key] for key in strategy_choices],
                k=1,
            )[0]
            pool_agent = self.opponent_pool.sample_opponent(strategy)
            if pool_agent is not None:
                return selection, pool_agent
        return "random", self.random_baseline

    def play_game(self, iteration: int, game_id: str = "spectate") -> SelfPlayResult:
        state = GameState()
        move_count = 0
        max_turns = self._turn_limit(iteration)
        candidate_color = "red" if random.random() < 0.5 else "blue"
        if self.opponent_override is not None:
            opponent_type, opponent_agent = self.opponent_override
        else:
            opponent_type, opponent_agent = self._sample_opponent()
        recorded: list[CandidateTurnRecord] = []
        entropy_warning = False
        forced_override_count = 0
        model_decision_count = 0
        tactical_opportunity_count = 0
        tactical_blunder_count = 0
        win_opportunity_count = 0
        missed_win_count = 0
        block_opportunity_count = 0
        missed_block_count = 0

        while not state.is_terminal and move_count < max_turns:
            if self.stop_requested is not None and self.stop_requested():
                break
            moving_player = state.current_player
            current_agent = self.candidate_agent if state.current_player == candidate_color else opponent_agent
            temperature = self.scheduler.get_temperature(iteration)
            greedy = move_count >= self.config.exploration.greedy_after_move
            forced = forced_move_policy(state) if self.config.use_tactical_selector else None
            forced_override_applied = False
            forced_override_reason: str | None = None

            if forced is not None and not isinstance(current_agent, ModelAgent):
                coord = forced.coord
                policy_for_record = forced.policy
            else:
                opening_policy = None
                if (
                    forced is None
                    and move_count < self.config.opening_random_moves
                    and random.random() < self.config.opening_random_move_probability
                ):
                    opening_policy = heuristic_policy(state)

                if opening_policy:
                    coord = self._sample_from_policy(opening_policy)
                    policy_for_record = opening_policy
                    entropy = float(
                        -(sum(prob * math.log(max(prob, 1e-9)) for prob in opening_policy.values()))
                    )
                    if state.current_player == candidate_color and entropy < self.config.exploration.min_policy_entropy_threshold:
                        entropy_warning = True
                elif isinstance(current_agent, ModelAgent):
                    heuristic_weight = self._heuristic_weight(iteration)
                    inference = current_agent.select_move(
                        state,
                        temperature=temperature,
                        greedy=greedy,
                        dirichlet_alpha=(
                            self.config.exploration.dirichlet_alpha
                            if self.config.exploration.use_dirichlet_noise
                            else None
                        ),
                        dirichlet_epsilon=self.config.exploration.dirichlet_epsilon,
                        prior_policy=heuristic_policy(state) if heuristic_weight > 0.0 else None,
                        prior_weight=heuristic_weight,
                        allow_tactical_override=self.config.use_tactical_selector,
                        forced=forced,
                    )
                    if (
                        state.current_player == candidate_color
                        and inference.entropy < self.config.exploration.min_policy_entropy_threshold
                    ):
                        entropy_warning = True
                    coord = inference.coord
                    policy_for_record = inference.forced_policy or inference.policy
                    if state.current_player == candidate_color:
                        model_decision_count += 1
                        if forced is not None:
                            tactical_opportunity_count += 1
                            if forced.reason in {"win_now", "win_this_turn"}:
                                win_opportunity_count += 1
                            elif forced.reason in {"block_now", "block_next_turn"}:
                                block_opportunity_count += 1
                        if inference.was_overridden:
                            forced_override_count += 1
                            forced_override_applied = True
                            forced_override_reason = inference.override_reason
                            tactical_blunder_count += 1
                            if inference.override_reason in {"win_now", "win_this_turn"}:
                                missed_win_count += 1
                            elif inference.override_reason in {"block_now", "block_next_turn"}:
                                missed_block_count += 1
                else:
                    coord = current_agent.select_move(state)
                    policy_for_record = forced.policy if forced is not None else heuristic_policy(state)

            previous_state = state
            state = apply_move(state, coord)
            move_count += 1

            if moving_player == candidate_color:
                immediate_reward, _tactical_blunder = self._immediate_reward(
                    previous_state,
                    coord,
                    state,
                    forced_override_applied=forced_override_applied,
                    forced_override_reason=forced_override_reason,
                )
                recorded.append(
                    CandidateTurnRecord(
                        state=previous_state,
                        move=coord,
                        experience=self.graph_builder.build_experience(
                            previous_state,
                            policy_targets=policy_for_record,
                            outcome=0.0,
                            aux_targets=self._aux_targets(previous_state),
                        ),
                        immediate_reward=immediate_reward,
                    )
                )

            if self.spectate_callback is not None:
                self.spectate_callback(
                    {
                        "type": "spectate_move",
                        "game_id": game_id,
                        "player": moving_player,
                        "hexes": [list(coord)],
                        "turn_number": state.turn_number,
                        "is_terminal": state.is_terminal,
                        "winner": state.winner,
                        "board_snapshot": {
                            "red": [list(item) for item in sorted(state.red_hexes)],
                            "blue": [list(item) for item in sorted(state.blue_hexes)],
                        },
                        "opponent_type": opponent_type,
                        "candidate_model": self.candidate_label,
                        "candidate_color": candidate_color,
                        "opponent_color": "blue" if candidate_color == "red" else "red",
                        "move_count": move_count,
                        "max_turns": max_turns,
                    }
                )

        winner = state.winner
        finalized: list[GNNExperience] = []
        running_return = self._terminal_reward(winner, candidate_color)
        for record in reversed(recorded):
            experience = record.experience
            immediate_reward = record.immediate_reward
            running_return = immediate_reward + (self.config.shaped_reward_discount * running_return)
            outcome = self._normalize_return(running_return)
            finalized.append(
                GNNExperience(
                    node_features=experience.node_features,
                    edge_index=experience.edge_index,
                    edge_attr=experience.edge_attr,
                    node_coords=experience.node_coords,
                    is_legal=experience.is_legal,
                    mcts_probs=experience.mcts_probs,
                    outcome=outcome,
                    turn_number=experience.turn_number,
                    aux_targets=experience.aux_targets,
                )
            )
        finalized.reverse()
        finalized.extend(self._loss_hindsight_experiences(recorded, winner, candidate_color))
        finalized.extend(self._tactical_correction_experiences(recorded))
        return SelfPlayResult(
            experiences=augment_experiences(finalized, self.config.rotation_augmentation_samples),
            winner=winner,
            candidate_color=candidate_color,
            move_count=move_count,
            entropy_warning=entropy_warning,
            forced_override_count=forced_override_count,
            model_decision_count=model_decision_count,
            tactical_opportunity_count=tactical_opportunity_count,
            tactical_blunder_count=tactical_blunder_count,
            win_opportunity_count=win_opportunity_count,
            missed_win_count=missed_win_count,
            block_opportunity_count=block_opportunity_count,
            missed_block_count=missed_block_count,
            final_snapshot={
                "type": "spectate_move",
                "game_id": game_id,
                "turn_number": state.turn_number,
                "is_terminal": state.is_terminal,
                "winner": state.winner,
                "board_snapshot": {
                    "red": [list(item) for item in sorted(state.red_hexes)],
                    "blue": [list(item) for item in sorted(state.blue_hexes)],
                },
                "opponent_type": opponent_type,
                "candidate_model": self.candidate_label,
                "candidate_color": candidate_color,
                "opponent_color": "blue" if candidate_color == "red" else "red",
                "move_count": move_count,
                "max_turns": max_turns,
            },
        )

    def _heuristic_weight(self, iteration: int) -> float:
        warmup = max(self.config.heuristic_guidance_warmup_iterations, 1)
        progress = min(max(iteration, 0) / warmup, 1.0)
        return self.config.heuristic_guidance_initial_weight + (
            self.config.heuristic_guidance_final_weight - self.config.heuristic_guidance_initial_weight
        ) * progress

    def _sample_from_policy(self, policy: dict[tuple[int, int], float]) -> tuple[int, int]:
        coords = list(policy.keys())
        probs = [policy[coord] for coord in coords]
        return random.choices(coords, weights=probs, k=1)[0]

    def _tactical_counts(self, state: GameState, player: str) -> tuple[int, int]:
        plans = next_turn_winning_plans(state, player)
        immediate = sum(1 for plan in plans if len(plan) == 1)
        next_turn = sum(1 for plan in plans if len(plan) == 2)
        return immediate, next_turn

    def _aux_targets(self, state: GameState) -> np.ndarray:
        player = state.current_player
        opponent = "blue" if player == "red" else "red"
        player_immediate, player_next_turn = self._tactical_counts(state, player)
        opponent_immediate, opponent_next_turn = self._tactical_counts(state, opponent)
        return np.asarray(
            [
                1.0 if player_immediate > 0 else 0.0,
                1.0 if opponent_immediate > 0 else 0.0,
                min(player_next_turn / 3.0, 1.0),
                min(opponent_next_turn / 3.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _component_sizes(self, state: GameState, player: str) -> list[int]:
        return sorted((len(component) for component in connected_components(state, player)), reverse=True)

    def _component_containing_move(self, state: GameState, player: str, move: tuple[int, int]) -> set[tuple[int, int]] | None:
        for component in connected_components(state, player):
            if move in component:
                return component
        return None

    def _nearest_existing_friendly_distance(self, state: GameState, player: str, move: tuple[int, int]) -> int:
        owned = state.red_hexes if player == "red" else state.blue_hexes
        if not owned:
            return 0
        return min(hex_distance(move[0], move[1], coord[0], coord[1]) for coord in owned)

    def _loss_hindsight_experiences(
        self,
        recorded: list[CandidateTurnRecord],
        winner: str | None,
        candidate_color: str,
    ) -> list[GNNExperience]:
        if winner is None or winner == candidate_color:
            return []

        hindsight: list[GNNExperience] = []
        seen_signatures: set[tuple[int, tuple[int, int]]] = set()
        correction_outcome = self._normalize_return(self.config.terminal_loss_reward)

        for record in reversed(recorded):
            correction = forced_move_policy(record.state)
            if correction is None or record.move in correction.forced_moves:
                continue
            signature = (record.state.turn_number, record.move)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            hindsight.append(
                self.graph_builder.build_experience(
                    record.state,
                    policy_targets=correction.policy,
                    outcome=correction_outcome,
                    aux_targets=self._aux_targets(record.state),
                )
            )
            if len(hindsight) >= self.config.hindsight_corrections_per_loss:
                break
        hindsight.reverse()
        return hindsight

    def _tactical_correction_experiences(
        self,
        recorded: list[CandidateTurnRecord],
    ) -> list[GNNExperience]:
        hindsight: list[GNNExperience] = []
        seen_turns: set[int] = set()
        correction_outcome = self._normalize_return(-self.config.forced_override_penalty)
        for record in reversed(recorded):
            correction = forced_move_policy(record.state)
            if correction is None or record.move in correction.forced_moves:
                continue
            if record.state.turn_number in seen_turns:
                continue
            seen_turns.add(record.state.turn_number)
            hindsight.append(
                self.graph_builder.build_experience(
                    record.state,
                    policy_targets=correction.policy,
                    outcome=correction_outcome,
                    aux_targets=self._aux_targets(record.state),
                )
            )
            if len(hindsight) >= self.config.tactical_corrections_per_game:
                break
        hindsight.reverse()
        return hindsight

    def _immediate_reward(
        self,
        state: GameState,
        move: tuple[int, int],
        next_state: GameState,
        forced_override_applied: bool = False,
        forced_override_reason: str | None = None,
    ) -> tuple[float, bool]:
        player = state.current_player
        opponent = "blue" if player == "red" else "red"
        before_line = longest_line_for_player(state, player)
        after_line = longest_line_for_player(next_state, player)
        reward = max(after_line - before_line, 0) * self.config.line_extension_reward
        player_immediate_before, player_next_turn_before = self._tactical_counts(state, player)
        player_immediate_after, player_next_turn_after = self._tactical_counts(next_state, player)
        opponent_immediate_before, opponent_next_turn_before = self._tactical_counts(state, opponent)
        opponent_immediate_after, opponent_next_turn_after = self._tactical_counts(next_state, opponent)
        must_defend = opponent_immediate_before > 0 or opponent_next_turn_before > 0

        threat_reduction = (opponent_immediate_before + opponent_next_turn_before) - (
            opponent_immediate_after + opponent_next_turn_after
        )
        if not must_defend:
            reward += max(player_next_turn_after - player_next_turn_before, 0) * self.config.next_turn_threat_reward
            created_pressure = player_immediate_after + player_next_turn_after
            if created_pressure >= 2:
                reward += min(created_pressure - 1, 2) * self.config.multi_threat_bonus
            reward += adjacent_friendly_count(state, player, move) * self.config.adjacency_reward
        if move in get_legal_moves(state) and would_win_if_played(state, opponent, move):
            reward += self.config.block_threat_reward
        if threat_reduction > 0:
            reward += threat_reduction * self.config.block_threat_reward

        before_components = connected_components(state, player)
        after_components = connected_components(next_state, player)
        before_component_sizes = sorted((len(component) for component in before_components), reverse=True)
        after_component_sizes = sorted((len(component) for component in after_components), reverse=True)
        second_before = before_component_sizes[1] if len(before_component_sizes) > 1 else 0
        second_after = after_component_sizes[1] if len(after_component_sizes) > 1 else 0
        move_component = self._component_containing_move(next_state, player, move)
        move_component_size = len(move_component) if move_component is not None else 0
        nearest_existing_distance = self._nearest_existing_friendly_distance(state, player, move)
        is_second_placement = state.placements_this_turn > 0
        before_nearest = average_nearest_friendly_distance(state, player)
        after_nearest = average_nearest_friendly_distance(next_state, player)
        is_free_expansion_move = (
            not must_defend
            and not next_state.is_terminal
            and
            is_second_placement
            and opponent_immediate_before == 0
            and opponent_next_turn_before == 0
        )
        seeded_new_front = bool(
            is_free_expansion_move
            and len(before_components) == 1
            and len(after_components) == 2
            and move_component_size == 1
            and 3 <= nearest_existing_distance <= 8
        )
        if not must_defend:
            reward += max(before_nearest - after_nearest, 0.0) * self.config.compactness_reward
            if len(before_components) > len(after_components):
                reward += self.config.compactness_reward
            if seeded_new_front:
                reward += self.config.colony_seed_reward
            if second_after > second_before:
                reward += min(second_after - second_before, 2) * self.config.multi_front_growth_reward
            if second_after >= 2 and player_next_turn_after > 0:
                reward += self.config.multi_front_threat_bonus
        reward -= max(len(after_components) - 2, 0) * self.config.colony_penalty
        isolated_components = sum(1 for component in after_components if len(component) == 1)
        if seeded_new_front and isolated_components > 0:
            isolated_components -= 1
        reward -= isolated_components * self.config.isolated_stone_penalty

        if next_state.is_terminal and next_state.winner == player and move in immediate_winning_moves(state, player):
            reward -= self.config.trivial_win_penalty
        reward -= opponent_immediate_after * self.config.trivial_loss_penalty
        reward -= opponent_next_turn_after * self.config.threat_exposure_penalty
        if forced_override_applied:
            reward -= self.config.forced_override_penalty
            if forced_override_reason in {"win_now", "win_this_turn"}:
                reward -= self.config.forced_override_penalty
            elif forced_override_reason in {"block_now", "block_next_turn"}:
                reward -= self.config.forced_override_penalty
        tactical_blunder = forced_override_applied or opponent_immediate_after > 0 or opponent_next_turn_after > 1
        return reward, tactical_blunder

    def _terminal_reward(self, winner: str | None, candidate_color: str) -> float:
        if winner is None:
            return self.config.draw_reward
        if winner == candidate_color:
            return self.config.terminal_win_reward
        return self.config.terminal_loss_reward

    def _normalize_return(self, reward: float) -> float:
        scale = max(self.config.reward_normalization_scale, 1e-6)
        return max(-1.0, min(1.0, reward / scale))


def _build_model_agent(config: TrainingConfig, state_dict: dict) -> ModelAgent:
    if torch is None:
        raise RuntimeError("torch is required for process self-play")
    device = getattr(config, "self_play_device", "cpu")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = HexGNNModel(config.model).to(device)
    model.load_state_dict(state_dict, strict=False)
    return ModelAgent(model, device)


def _write_progress_snapshot(progress_path: str, snapshot: dict):
    temp_path = f"{progress_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle)
    os.replace(temp_path, progress_path)


def run_self_play_job(job: SelfPlayJob) -> SelfPlayResult:
    if torch is not None:
        torch.set_num_threads(max(1, job.config.self_play_worker_torch_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    candidate_agent = _build_model_agent(job.config, job.candidate_state_dict)
    if job.opponent_kind == "random":
        opponent_agent = RandomBaseline()
    elif job.opponent_kind == "heuristic":
        from backend.training.baselines import HeuristicBaseline

        opponent_agent = HeuristicBaseline()
    elif job.opponent_state_dict is not None:
        opponent_agent = _build_model_agent(job.config, job.opponent_state_dict)
    else:
        opponent_agent = RandomBaseline()

    spectate_callback = None
    if job.progress_path:
        def spectate_callback(message: dict):
            if message.get("type") == "spectate_move":
                _write_progress_snapshot(job.progress_path, message)

    worker = SelfPlayWorker(
        candidate_agent=candidate_agent,
        reference_agent=None,
        opponent_pool=OpponentPool(
            model_config=job.config.model,
            max_size=job.config.eval.opponent_pool_max_size,
            device=job.config.device,
        ),
        config=job.config,
        spectate_callback=spectate_callback,
        stop_requested=None,
        opponent_override=(job.opponent_label, opponent_agent),
        candidate_label=job.candidate_label,
    )
    result = worker.play_game(job.iteration, job.game_id)
    if job.progress_path and result.final_snapshot is not None:
        _write_progress_snapshot(job.progress_path, result.final_snapshot)
    return result

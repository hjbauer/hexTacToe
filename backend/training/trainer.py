from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing
import math
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from backend.config import TrainingConfig
from backend.model.graph_builder import experiences_to_batch
from backend.model.inference import ModelAgent
from backend.model.network import HexGNNModel
from backend.state.app_state import AppState, EvalMetrics
from backend.training.baselines import HeuristicBaseline, RandomBaseline
from backend.training.checkpoint import (
    list_checkpoints,
    load_checkpoint,
    mark_reference,
    save_checkpoint,
)
from backend.training.evaluator import EvaluationManager
from backend.training.exploration import ExplorationScheduler
from backend.training.opponent_pool import OpponentPool
from backend.training.replay_buffer import ReplayBuffer
from backend.training.self_play import SelfPlayJob, SelfPlayResult, SelfPlayWorker, run_self_play_job


@dataclass
class PopulationMember:
    name: str
    lineage: str
    role: str
    generation: int
    parent_names: list[str]
    model: HexGNNModel
    optimizer: Any
    replay_buffer: ReplayBuffer
    elo: float
    peak_elo: float
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    score: float = 0.0
    last_result: str = "seed"


class TrainingLoop:
    def __init__(self, app_state: AppState, config: Optional[TrainingConfig] = None):
        if torch is None:
            raise RuntimeError("torch is required for training")
        self.app_state = app_state
        self.config = config or TrainingConfig()
        self.device = self.config.device
        self.reference_model = self._new_model()
        self.reference_elo = self.config.initial_elo
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_capacity,
            historical_fraction=self.config.replay_buffer_historical_fraction,
        )
        self.population = self._create_population()
        self.model = self.population[0].model
        self.optimizer = self.population[0].optimizer
        self.reference_model.load_state_dict(self.model.state_dict())
        self.opponent_pool = OpponentPool(
            model_config=self.config.model,
            max_size=self.config.eval.opponent_pool_max_size,
            device=self.device,
        )
        self.evaluator = EvaluationManager(self.config.eval, self.config)
        self.scheduler = ExplorationScheduler(self.config.exploration)
        self._stop_event = asyncio.Event()
        self._spectate_game_index = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._self_play_executor: Optional[ProcessPoolExecutor] = None
        self._self_play_executor_workers: int = 0
        self._assign_lineage_roles()
        self._sync_serving_member()
        self._set_reference_from_leader()

    def _new_model(self) -> HexGNNModel:
        return HexGNNModel(self.config.model).to(self.device)

    def _new_optimizer(self, model: HexGNNModel):
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _lineage_name(self, lineage_index: int) -> str:
        return f"lineage-{lineage_index + 1}"

    def _create_member(self, lineage_index: int, slot_index: int) -> PopulationMember:
        model = self._new_model()
        return PopulationMember(
            name=f"L{lineage_index + 1}-M{slot_index + 1}",
            lineage=self._lineage_name(lineage_index),
            role="seed",
            generation=0,
            parent_names=[],
            model=model,
            optimizer=self._new_optimizer(model),
            replay_buffer=ReplayBuffer(
                capacity=self.config.replay_buffer_capacity,
                historical_fraction=self.config.replay_buffer_historical_fraction,
            ),
            elo=self.config.initial_elo,
            peak_elo=self.config.initial_elo,
        )

    def _create_population(self) -> list[PopulationMember]:
        lineage_counts = [0 for _ in range(max(self.config.lineage_count, 1))]
        members: list[PopulationMember] = []
        for index in range(self.config.population_size):
            lineage_index = index % max(self.config.lineage_count, 1)
            slot_index = lineage_counts[lineage_index]
            lineage_counts[lineage_index] += 1
            members.append(self._create_member(lineage_index, slot_index))
        return members

    def _population_groups(self) -> dict[str, list[PopulationMember]]:
        groups: dict[str, list[PopulationMember]] = {}
        for member in self.population:
            groups.setdefault(member.lineage, []).append(member)
        return groups

    def _member_by_name(self, name: str) -> PopulationMember:
        for member in self.population:
            if member.name == name:
                return member
        raise KeyError(f"unknown population member {name}")

    def _leader_member(self) -> PopulationMember:
        return max(
            self.population,
            key=lambda member: (member.elo, member.score, member.peak_elo, -member.losses, member.name),
        )

    def _lineage_champion(self, lineage: str) -> PopulationMember:
        return max(
            (member for member in self.population if member.lineage == lineage),
            key=lambda member: (member.elo, member.score, member.peak_elo, member.name),
        )

    def _assign_lineage_roles(self):
        for members in self._population_groups().values():
            ranked = sorted(
                members,
                key=lambda member: (member.elo, member.score, member.peak_elo, member.name),
                reverse=True,
            )
            for index, member in enumerate(ranked):
                if index == 0:
                    member.role = "champion"
                elif index == 1:
                    member.role = "contender"
                else:
                    member.role = "offspring"

    def _sync_serving_member(self):
        leader = self._leader_member()
        self.model = leader.model
        self.optimizer = leader.optimizer
        self.app_state.training_status.population_size = len(self.population)
        self.app_state.training_status.leader_model_name = leader.name
        self.app_state.training_status.leader_model_elo = leader.elo

    def _set_reference_from_leader(self):
        leader = self._leader_member()
        self.reference_model.load_state_dict(leader.model.state_dict())
        self.reference_elo = leader.elo

    def leaderboard(self) -> list[dict[str, Any]]:
        ranked = sorted(
            self.population,
            key=lambda member: (member.elo, member.score, member.peak_elo, member.name),
            reverse=True,
        )
        return [
            {
                "name": member.name,
                "lineage": member.lineage,
                "role": member.role,
                "generation": member.generation,
                "elo": member.elo,
                "peak_elo": member.peak_elo,
                "games": member.games,
                "wins": member.wins,
                "losses": member.losses,
                "draws": member.draws,
                "score": member.score,
                "parents": list(member.parent_names),
                "last_result": member.last_result,
            }
            for member in ranked
        ]

    def available_opponents(self) -> list[dict[str, Any]]:
        leader = self._leader_member()
        opponents = [
            {
                "id": "leader",
                "label": f"Leader ({leader.name})",
                "elo": leader.elo,
                "kind": "leader",
            },
            {
                "id": "reference",
                "label": "Reference",
                "elo": self.reference_elo,
                "kind": "reference",
            },
            {
                "id": "heuristic",
                "label": "Heuristic",
                "elo": 1000.0,
                "kind": "baseline",
            },
            {
                "id": "random",
                "label": "Random",
                "elo": 800.0,
                "kind": "baseline",
            },
        ]
        opponents.extend(
            {
                "id": member["name"],
                "label": f"{member['name']} [{member['lineage']}]",
                "elo": member["elo"],
                "kind": "population",
                "role": member["role"],
                "generation": member["generation"],
            }
            for member in self.leaderboard()
        )
        return opponents

    def get_opponent_descriptor(self, opponent_id: str | None) -> tuple[str, float, object]:
        if opponent_id in {None, "", "leader"}:
            leader = self._leader_member()
            return leader.name, leader.elo, ModelAgent(leader.model, self.device)
        if opponent_id == "reference":
            return "reference", self.reference_elo, ModelAgent(self.reference_model, self.device)
        if opponent_id == "random":
            return "random", 800.0, RandomBaseline()
        if opponent_id == "heuristic":
            return "heuristic", 1000.0, HeuristicBaseline()
        try:
            member = self._member_by_name(opponent_id)
        except KeyError:
            leader = self._leader_member()
            return leader.name, leader.elo, ModelAgent(leader.model, self.device)
        return member.name, member.elo, ModelAgent(member.model, self.device)

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self.app_state.training_status.is_training = True
        self._sync_serving_member()
        self._record_event("Training started")
        while not self._stop_event.is_set():
            await self.run_iteration()
            await asyncio.sleep(0)
        self.app_state.training_status.is_training = False
        self.app_state.training_task = None
        self._set_phase("idle", "Training stopped")
        await self._broadcast_training_status()

    async def stop(self):
        self._stop_event.set()
        self._record_event("Stop requested")

    async def run_iteration(self):
        status = self.app_state.training_status
        status.iteration += 1
        status.entropy_warning = False
        temperature = self.scheduler.get_temperature(status.iteration)
        self._assign_lineage_roles()
        self._sync_serving_member()
        population_agents = {
            member.name: ModelAgent(member.model, self.device)
            for member in self.population
        }
        population_state_dicts = {
            member.name: copy.deepcopy(member.model.state_dict())
            for member in self.population
        }
        leader_name = self._leader_member().name
        total_games = self.config.self_play_games_per_iteration
        total_progress = total_games * self.config.max_turns_per_game
        self._set_phase(
            "self_play",
            f"League self-play iteration {status.iteration}",
            progress=0,
            total=total_progress,
        )
        await self._broadcast_training_status()

        candidates = self._scheduled_candidates(total_games)
        worker_count = max(1, min(self.config.parallel_self_play_workers, total_games))
        completed_games = 0
        completed_move_count = 0
        self.app_state.spectate_games = {}
        self.app_state.current_spectate = None
        await self._broadcast_spectate_state()
        await self._broadcast_training_status()

        progress_dir = Path(tempfile.mkdtemp(prefix="hex-selfplay-progress-"))
        next_candidate_index = 0
        active_local_games = 0
        pending_tasks: set[asyncio.Task] = set()
        active_game_ids: set[str] = set()
        active_process_game_ids: set[str] = set()
        move_progress: dict[str, int] = {}

        def observed_moves() -> int:
            return completed_move_count + sum(move_progress.get(game_id, 0) for game_id in active_game_ids)

        async def run_local_game(entry: dict[str, Any]):
            member: PopulationMember = entry["candidate"]
            opponent = entry["opponent"]
            worker = SelfPlayWorker(
                candidate_agent=population_agents[member.name],
                reference_agent=ModelAgent(self.reference_model, self.device),
                opponent_pool=self.opponent_pool,
                config=self.config,
                spectate_callback=self._broadcast_spectate,
                stop_requested=self._stop_event.is_set,
                opponent_override=(opponent["label"], opponent["agent"]),
                candidate_label=member.name,
            )
            result = await asyncio.to_thread(worker.play_game, status.iteration, entry["game_id"])
            return entry, result

        async def run_process_game(entry: dict[str, Any]):
            member: PopulationMember = entry["candidate"]
            opponent = entry["opponent"]
            job = SelfPlayJob(
                config=self.config,
                iteration=status.iteration,
                game_id=entry["game_id"],
                candidate_label=member.name,
                candidate_state_dict=population_state_dicts[member.name],
                opponent_label=opponent["label"],
                opponent_kind=opponent["kind"],
                opponent_state_dict=opponent.get("state_dict"),
                progress_path=str(progress_dir / f"{entry['game_id']}.json"),
            )
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._ensure_self_play_executor(), run_self_play_job, job)
            return entry, result

        async def launch_next_game() -> bool:
            nonlocal next_candidate_index, active_local_games
            if next_candidate_index >= total_games or self._stop_event.is_set():
                return False
            member = candidates[next_candidate_index]
            next_candidate_index += 1
            opponent = self._select_opponent(member, population_agents)
            self._spectate_game_index += 1
            game_id = f"sp-{self._spectate_game_index}"
            snapshot = {
                "type": "spectate_new_game",
                "game_id": game_id,
                "board_snapshot": {"red": [], "blue": []},
                "opponent_type": opponent["label"],
                "opponent_kind": opponent["kind"],
                "opponent_name": opponent["name"],
                "candidate_model": member.name,
                "candidate_role": member.role,
                "candidate_lineage": member.lineage,
                "candidate_is_leader": member.name == leader_name,
                "candidate_color": "waiting",
                "opponent_color": "waiting",
                "candidate_elo": member.elo,
                "opponent_elo": opponent["elo"],
                "opponent_role": opponent.get("member").role if opponent.get("member") is not None else opponent["kind"],
                "move_count": 0,
                "max_turns": self.config.max_turns_per_game,
                "is_terminal": False,
                "winner": None,
            }
            self.app_state.spectate_games[game_id] = snapshot
            if self.app_state.current_spectate is None:
                self.app_state.current_spectate = snapshot
            entry = {
                "candidate": member,
                "opponent": opponent,
                "game_id": game_id,
            }
            use_local = active_local_games < min(self.config.local_spectate_games_per_batch, worker_count)
            if use_local:
                active_local_games += 1
                entry["execution"] = "local"
                task = asyncio.create_task(run_local_game(entry))
            else:
                entry["execution"] = "process"
                active_process_game_ids.add(game_id)
                move_progress[game_id] = 0
                task = asyncio.create_task(run_process_game(entry))
            pending_tasks.add(task)
            active_game_ids.add(game_id)
            await self._broadcast_spectate_state()
            return True

        async def poll_live_progress():
            while (pending_tasks or active_process_game_ids) and not self._stop_event.is_set():
                changed = False
                for game_id in list(active_process_game_ids):
                    progress_path = progress_dir / f"{game_id}.json"
                    if not progress_path.exists():
                        continue
                    try:
                        snapshot = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    previous_move_count = move_progress.get(game_id, 0)
                    move_count = int(snapshot.get("move_count") or 0)
                    move_progress[game_id] = move_count
                    merged = {**self.app_state.spectate_games.get(game_id, {}), **snapshot}
                    self.app_state.spectate_games[game_id] = merged
                    self.app_state.current_spectate = merged
                    changed = changed or move_count != previous_move_count
                if changed:
                    self._set_phase(
                        "self_play",
                        f"Played {completed_games}/{total_games} league games | {observed_moves()} moves observed",
                        progress=observed_moves(),
                        total=total_progress,
                    )
                    await self._broadcast_spectate_state()
                    await self._broadcast_training_status()
                await asyncio.sleep(0.2)

        for _ in range(min(worker_count, total_games)):
            await launch_next_game()

        poller = asyncio.create_task(poll_live_progress())
        try:
            while pending_tasks:
                done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                pending_tasks = set(pending)
                for task in done:
                    entry, result = await task
                    game_id = entry["game_id"]
                    active_game_ids.discard(game_id)
                    active_process_game_ids.discard(game_id)
                    if entry.get("execution") == "local":
                        active_local_games = max(active_local_games - 1, 0)
                    move_progress.pop(game_id, None)
                    self.app_state.spectate_games.pop(game_id, None)
                    if self.app_state.current_spectate is not None and self.app_state.current_spectate.get("game_id") == game_id:
                        self.app_state.current_spectate = None
                    if self._stop_event.is_set():
                        continue
                    member = entry["candidate"]
                    opponent = entry["opponent"]
                    member.replay_buffer.add(result.experiences)
                    self.replay_buffer.add(result.experiences)
                    self._apply_match_result(member, opponent, result)
                    status.episode += 1
                    status.games_played += 1
                    status.forced_override_count += result.forced_override_count
                    status.model_decision_count += result.model_decision_count
                    status.forced_override_rate = (
                        status.forced_override_count / status.model_decision_count
                        if status.model_decision_count
                        else 0.0
                    )
                    completed_games += 1
                    completed_move_count += result.move_count
                    status.entropy_warning = status.entropy_warning or result.entropy_warning
                    self._assign_lineage_roles()
                    self._sync_serving_member()
                    await launch_next_game()
                    self._set_phase(
                        "self_play",
                        f"Played {completed_games}/{total_games} league games | {observed_moves()} moves observed",
                        progress=observed_moves(),
                        total=total_progress,
                    )
                    await self._broadcast_spectate_state()
                    await self._broadcast_training_status()
        finally:
            poller.cancel()
            try:
                await poller
            except asyncio.CancelledError:
                pass
            shutil.rmtree(progress_dir, ignore_errors=True)

        if self._stop_event.is_set():
            self._set_phase("idle", "Stopping training", progress=0, total=0)
            await self._broadcast_training_status()
            return

        await self._train_population()
        self._assign_lineage_roles()

        status.replay_buffer_size = len(self.replay_buffer)
        status.current_temperature = temperature
        status.opponent_pool_size = len(self.opponent_pool)
        self._sync_serving_member()
        await self._broadcast_training_status()

        if status.iteration % self.config.eval.snapshot_interval == 0:
            leader = self._leader_member()
            snapshot_path = f"{leader.name}_iter_{status.iteration:04d}"
            self.opponent_pool.add_snapshot(leader.model, status.iteration, snapshot_path)
            status.opponent_pool_size = len(self.opponent_pool)
            self._record_event(f"Snapshot added for {leader.name} at iteration {status.iteration}")

        if status.iteration % self.config.eval.eval_interval == 0:
            self._set_phase("evaluation", "Running evaluation match", progress=0, total=1)
            await self._broadcast_training_status()
            await self._run_evaluation()

        if status.iteration % self.config.evolution_interval == 0:
            self._evolve_population()
            self._assign_lineage_roles()
            self._sync_serving_member()
            await self._broadcast_training_status()

        if self.config.checkpoint_interval > 0 and status.iteration % self.config.checkpoint_interval == 0:
            self._set_phase("checkpoint", "Saving checkpoint", progress=1, total=1)
            await self._broadcast_training_status()
            path = await asyncio.to_thread(self.save_checkpoint, False)
            await self.app_state.ws_manager.broadcast(
                {"type": "checkpoint_saved", "path": path},
                channels=["training"],
            )
            self._record_event(f"Checkpoint saved: {path.split('/')[-1]}")

        self._set_phase("idle", "Waiting for next iteration", progress=0, total=0)
        await self._broadcast_training_status()

    def _scheduled_candidates(self, total_games: int) -> list[PopulationMember]:
        ranked = sorted(
            self.population,
            key=lambda member: (member.games, member.elo, random.random()),
        )
        return [ranked[index % len(ranked)] for index in range(total_games)]

    def _select_opponent(self, candidate: PopulationMember, population_agents: dict[str, ModelAgent]) -> dict[str, Any]:
        choices = ["population", "reference", "pool", "random"]
        weights = [
            self.config.opponent_sampling_weights.get("population", 0.0),
            self.config.opponent_sampling_weights.get("reference", 0.0),
            self.config.opponent_sampling_weights.get("pool", 0.0) if len(self.opponent_pool) else 0.0,
            self.config.opponent_sampling_weights.get("random", 0.0),
        ]
        selection = random.choices(choices, weights=weights, k=1)[0]
        if selection == "population":
            return self._select_population_opponent(candidate, population_agents)
        if selection == "reference":
            return {
                "label": "reference",
                "kind": "reference",
                "name": "reference",
                "elo": self.reference_elo,
                "agent": ModelAgent(self.reference_model, self.device),
                "state_dict": copy.deepcopy(self.reference_model.state_dict()),
                "member": None,
            }
        if selection == "pool":
            strategy_weights = self.config.pool_strategy_weights
            strategy_choices = list(strategy_weights.keys())
            strategy = random.choices(
                strategy_choices,
                weights=[strategy_weights[key] for key in strategy_choices],
                k=1,
            )[0]
            pool_snapshot = self.opponent_pool.sample_snapshot(strategy)
            if pool_snapshot is not None:
                return {
                    "label": f"pool:{strategy}",
                    "kind": "pool",
                    "name": f"pool:{strategy}",
                    "elo": self.reference_elo,
                    "agent": self.opponent_pool.agent_from_snapshot(pool_snapshot),
                    "state_dict": copy.deepcopy(pool_snapshot.state_dict),
                    "member": None,
                }
        return {
            "label": "random",
            "kind": "random",
            "name": "random",
            "elo": 800.0,
            "agent": RandomBaseline(),
            "state_dict": None,
            "member": None,
        }

    def _select_population_opponent(self, candidate: PopulationMember, population_agents: dict[str, ModelAgent]) -> dict[str, Any]:
        same_lineage = [member for member in self.population if member.lineage == candidate.lineage and member.name != candidate.name]
        cross_lineage = [member for member in self.population if member.lineage != candidate.lineage]
        prefer_same = random.random() < self.config.same_lineage_match_weight
        pool = same_lineage if prefer_same and same_lineage else cross_lineage if cross_lineage else same_lineage
        if not pool:
            pool = [member for member in self.population if member.name != candidate.name]
        weighted: list[tuple[PopulationMember, float]] = []
        for member in pool:
            distance = abs(candidate.elo - member.elo)
            weight = 1.0 / (1.0 + (distance / 200.0))
            if member.lineage == candidate.lineage:
                weight *= 1.1
            weighted.append((member, weight))
        opponent = random.choices([item[0] for item in weighted], weights=[item[1] for item in weighted], k=1)[0]
        return {
            "label": f"population:{opponent.name}",
            "kind": "population",
            "name": opponent.name,
            "elo": opponent.elo,
            "agent": population_agents[opponent.name],
            "state_dict": copy.deepcopy(opponent.model.state_dict()),
            "member": opponent,
        }

    def _ensure_self_play_executor(self) -> ProcessPoolExecutor:
        target_workers = max(1, self.config.parallel_self_play_workers - self.config.local_spectate_games_per_batch)
        if (
            self._self_play_executor is None
            or self._self_play_executor_workers != target_workers
        ):
            if self._self_play_executor is not None:
                self._self_play_executor.shutdown(wait=False, cancel_futures=True)
            context = multiprocessing.get_context("spawn")
            self._self_play_executor = ProcessPoolExecutor(
                max_workers=target_workers,
                mp_context=context,
            )
            self._self_play_executor_workers = target_workers
        return self._self_play_executor

    def _apply_match_result(self, candidate: PopulationMember, opponent: dict[str, Any], result: SelfPlayResult):
        score = 0.5 if result.winner is None else (1.0 if result.winner == result.candidate_color else 0.0)
        candidate.games += 1
        if score == 1.0:
            candidate.wins += 1
            candidate.score += 1.0
            candidate.last_result = "win"
        elif score == 0.5:
            candidate.draws += 1
            candidate.score += 0.25
            candidate.last_result = "draw"
        else:
            candidate.losses += 1
            candidate.score -= 0.25
            candidate.last_result = "loss"

        if opponent["kind"] == "population" and opponent["member"] is not None:
            other: PopulationMember = opponent["member"]
            other.games += 1
            if score == 1.0:
                other.losses += 1
                other.score -= 0.25
                other.last_result = "loss"
            elif score == 0.5:
                other.draws += 1
                other.score += 0.25
                other.last_result = "draw"
            else:
                other.wins += 1
                other.score += 1.0
                other.last_result = "win"
            self._update_elo_pair(candidate, other, score)
        elif opponent["kind"] == "reference":
            candidate.elo = self._updated_elo(candidate.elo, self.reference_elo, score)
        candidate.peak_elo = max(candidate.peak_elo, candidate.elo)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + pow(10.0, (rating_b - rating_a) / 400.0))

    def _updated_elo(self, rating_a: float, rating_b: float, score_a: float) -> float:
        expected = self._expected_score(rating_a, rating_b)
        return rating_a + self.config.elo_k_factor * (score_a - expected)

    def _update_elo_pair(self, member_a: PopulationMember, member_b: PopulationMember, score_a: float):
        next_a = self._updated_elo(member_a.elo, member_b.elo, score_a)
        next_b = self._updated_elo(member_b.elo, member_a.elo, 1.0 - score_a)
        member_a.elo = next_a
        member_b.elo = next_b
        member_a.peak_elo = max(member_a.peak_elo, next_a)
        member_b.peak_elo = max(member_b.peak_elo, next_b)

    async def _train_population(self):
        trainable = [member for member in self.population if len(member.replay_buffer) >= self.config.min_buffer_size_to_train]
        if not trainable:
            return

        self._set_phase(
            "training",
            "Running league gradient updates",
            progress=0,
            total=len(trainable),
        )
        await self._broadcast_training_status()

        tasks = [
            asyncio.create_task(asyncio.to_thread(self._train_member_epoch, member))
            for member in trainable
        ]
        completed = 0
        policy_losses: list[float] = []
        value_losses: list[float] = []

        for task in asyncio.as_completed(tasks):
            result = await task
            if result is None:
                continue
            completed += 1
            if result["policy_loss"] is not None:
                policy_losses.append(result["policy_loss"])
            if result["value_loss"] is not None:
                value_losses.append(result["value_loss"])
            self.app_state.training_status.phase_progress = completed
            self.app_state.training_status.phase_total = len(trainable)
            self.app_state.training_status.status_message = f"Updated {completed}/{len(trainable)} models"
            await self._broadcast_training_status()

        avg_policy = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        avg_value = sum(value_losses) / len(value_losses) if value_losses else 0.0
        self.app_state.training_status.loss_policy = avg_policy
        self.app_state.training_status.loss_value = avg_value
        self.app_state.training_status.loss_history.append(
            {
                "iter": self.app_state.training_status.iteration,
                "lp": avg_policy,
                "lv": avg_value,
            }
        )
        self.app_state.training_status.loss_history = self.app_state.training_status.loss_history[-200:]

    def _train_member_epoch(self, member: PopulationMember) -> dict[str, float | None] | None:
        last_policy_loss: Optional[float] = None
        last_value_loss: Optional[float] = None
        for _ in range(self.config.gradient_steps_per_iteration):
            if self._stop_event.is_set():
                return None
            batch = member.replay_buffer.sample(self.config.batch_size)
            if not batch:
                return None
            pyg_batch = experiences_to_batch(batch).to(self.device)
            member.model.train()
            member.optimizer.zero_grad()
            _, values = member.model(pyg_batch)
            log_probs = member.model.policy_log_probs(pyg_batch)

            legal_mask = pyg_batch.legal_mask
            policy_targets = pyg_batch.policy_target[legal_mask]
            policy_loss = -(policy_targets * log_probs[legal_mask]).sum() / max(
                int(getattr(pyg_batch, "num_graphs", 1)),
                1,
            )
            value_targets = pyg_batch.outcome.view(-1)
            value_loss = F.mse_loss(values, value_targets)
            if not math.isfinite(float(policy_loss.item())) or not math.isfinite(float(value_loss.item())):
                self._record_event(f"Skipped non-finite gradient step for {member.name}")
                continue
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(member.model.parameters(), self.config.gradient_clip_norm)
            member.optimizer.step()
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
        return {"policy_loss": last_policy_loss, "value_loss": last_value_loss}

    async def _run_evaluation(self):
        leader = self._leader_member()
        candidate = ModelAgent(leader.model, self.device)
        reference = ModelAgent(self.reference_model, self.device)
        metrics = await self.evaluator.run_evaluation(
            candidate,
            reference,
            self.opponent_pool,
            self.app_state.training_status.iteration,
        )
        promoted = self.evaluator.should_promote(metrics)
        metrics.was_promoted = promoted
        if promoted:
            self.reference_model.load_state_dict(leader.model.state_dict())
            self.reference_elo = leader.elo
            self.opponent_pool.add_snapshot(
                self.reference_model,
                self.app_state.training_status.iteration,
                f"{leader.name}_reference_iter_{self.app_state.training_status.iteration:04d}",
                is_reference=True,
            )
            metrics.promotion_history = self.app_state.eval_metrics.promotion_history + [
                {
                    "iteration": self.app_state.training_status.iteration,
                    "checkpoint": f"{leader.name}_reference_iter_{self.app_state.training_status.iteration:04d}",
                }
            ]
            self.app_state.training_status.reference_checkpoint_path = None
            self._record_event(f"{leader.name} promoted at iteration {self.app_state.training_status.iteration}")
            await self.app_state.ws_manager.broadcast(
                {
                    "type": "model_promoted",
                    "iteration": self.app_state.training_status.iteration,
                    "new_checkpoint": f"{leader.name}_reference_iter_{self.app_state.training_status.iteration:04d}",
                },
                channels=["training"],
            )
        else:
            metrics.promotion_history = self.app_state.eval_metrics.promotion_history
            self._record_event(f"Evaluation finished for {leader.name} at iteration {self.app_state.training_status.iteration}")

        self.app_state.eval_metrics = metrics
        await self.app_state.ws_manager.broadcast(
            {
                "type": "eval_result",
                "iteration": metrics.last_eval_iteration,
                "vs_reference": {
                    "win": metrics.win_rate_vs_reference,
                    "loss": metrics.loss_rate_vs_reference,
                    "draw": max(0.0, 1.0 - metrics.win_rate_vs_reference - metrics.loss_rate_vs_reference),
                    "promoted": promoted,
                },
                "vs_random": {"win": metrics.win_rate_vs_random},
                "as_red_win_rate": metrics.win_rate_as_red,
                "as_blue_win_rate": metrics.win_rate_as_blue,
                "avg_game_length": metrics.avg_game_length,
                "vs_pool": metrics.pool_win_rates,
            },
            channels=["training"],
        )

    def _evolve_population(self):
        self._assign_lineage_roles()
        protected_names = {self._lineage_champion(lineage).name for lineage in self._population_groups()}
        replaceable = [
            member for member in sorted(self.population, key=lambda item: (item.elo, item.score, item.games, item.name))
            if member.name not in protected_names
        ]
        if not replaceable:
            return
        replacements = min(self.config.replacements_per_evolution, len(replaceable))
        for target in replaceable[:replacements]:
            champion = self._lineage_champion(target.lineage)
            secondary = None
            if random.random() < self.config.cross_lineage_parent_rate:
                foreign_champions = [
                    self._lineage_champion(lineage)
                    for lineage in self._population_groups()
                    if lineage != target.lineage
                ]
                if foreign_champions:
                    secondary = max(foreign_champions, key=lambda member: (member.elo, member.peak_elo))
            self._spawn_offspring(target, champion, secondary)
        self._assign_lineage_roles()

    def _spawn_offspring(
        self,
        target: PopulationMember,
        primary_parent: PopulationMember,
        secondary_parent: Optional[PopulationMember],
    ):
        state_dict = copy.deepcopy(primary_parent.model.state_dict())
        if secondary_parent is not None:
            secondary_state = secondary_parent.model.state_dict()
            for key, tensor in state_dict.items():
                if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                    state_dict[key] = (0.8 * tensor) + (0.2 * secondary_state[key].to(tensor.device))
        for key, tensor in state_dict.items():
            if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                state_dict[key] = tensor + (torch.randn_like(tensor) * self.config.mutation_std)
        target.model = self._new_model()
        target.model.load_state_dict(state_dict)
        target.optimizer = self._new_optimizer(target.model)
        target.replay_buffer.clear()
        target.generation = max(primary_parent.generation, secondary_parent.generation if secondary_parent else 0) + 1
        target.parent_names = [primary_parent.name] + ([secondary_parent.name] if secondary_parent else [])
        inherited_elo = primary_parent.elo if secondary_parent is None else (primary_parent.elo + secondary_parent.elo) / 2.0
        target.elo = max(800.0, inherited_elo - 25.0)
        target.peak_elo = target.elo
        target.games = 0
        target.wins = 0
        target.losses = 0
        target.draws = 0
        target.score = 0.0
        target.role = "offspring"
        target.last_result = "spawned"
        parent_text = ", ".join(target.parent_names)
        self._record_event(f"Spawned {target.name} from {parent_text}")

    async def _broadcast_training_status(self):
        payload = {"type": "training_status", **self.app_state.training_status.to_dict(include_history=False)}
        await self.app_state.ws_manager.broadcast(payload, channels=["training"])

    def _broadcast_spectate(self, message: dict):
        if message.get("type") != "spectate_move":
            return
        merged = {**self.app_state.spectate_games.get(message["game_id"], {}), **dict(message)}
        self.app_state.current_spectate = merged
        self.app_state.spectate_games[message["game_id"]] = merged
        if self.app_state.ws_manager is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._broadcast_spectate_state()))

    async def _broadcast_spectate_state(self):
        games = sorted(self.app_state.spectate_games.values(), key=lambda game: game["game_id"])
        await self.app_state.ws_manager.broadcast(
            {
                "type": "spectate_state",
                "games": games,
                "population_size": len(self.population),
            },
            channels=["spectate"],
        )

    def checkpoint_payload(self, is_reference: bool) -> dict:
        leader = self._leader_member()
        model_state = copy.deepcopy(leader.model.state_dict())
        reference_state = (
            copy.deepcopy(model_state)
            if is_reference
            else copy.deepcopy(self.reference_model.state_dict())
        )
        optimizer_state = copy.deepcopy(leader.optimizer.state_dict())
        return {
            "model_state_dict": model_state,
            "reference_model_state_dict": reference_state,
            "optimizer_state_dict": optimizer_state,
            "model_config": self.config.model.to_dict(),
            "training_config": self.config.to_dict(),
            "iteration": self.app_state.training_status.iteration,
            "episode": self.app_state.training_status.episode,
            "games_played": self.app_state.training_status.games_played,
            "loss_history": self.app_state.training_status.loss_history,
            "forced_override_count": self.app_state.training_status.forced_override_count,
            "model_decision_count": self.app_state.training_status.model_decision_count,
            "reference_elo": self.reference_elo,
            "population": self._serialize_population(),
            "leader_model_name": leader.name,
            "eval_history": list(self.app_state.eval_metrics.promotion_history),
            "is_reference": is_reference,
            "replay_buffer": self.replay_buffer.serialize(
                max_recent=self.config.checkpoint_recent_samples,
                max_historical=self.config.checkpoint_historical_samples,
            ),
            "opponent_pool": self.opponent_pool.serialize(),
            "eval_metrics": self.app_state.eval_metrics.to_dict(),
        }

    def _serialize_population(self) -> list[dict[str, Any]]:
        return [
            {
                "name": member.name,
                "lineage": member.lineage,
                "role": member.role,
                "generation": member.generation,
                "parent_names": list(member.parent_names),
                "model_state_dict": copy.deepcopy(member.model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(member.optimizer.state_dict()),
                "replay_buffer": member.replay_buffer.serialize(
                    max_recent=self.config.checkpoint_member_recent_samples,
                    max_historical=self.config.checkpoint_member_historical_samples,
                ),
                "games": member.games,
                "wins": member.wins,
                "losses": member.losses,
                "draws": member.draws,
                "score": member.score,
                "elo": member.elo,
                "peak_elo": member.peak_elo,
                "last_result": member.last_result,
            }
            for member in self.population
        ]

    def save_checkpoint(self, is_reference: bool = False) -> str:
        payload = self.checkpoint_payload(is_reference)
        path = save_checkpoint(payload, keep_last=self.config.keep_last_checkpoints)
        if is_reference:
            mark_reference(path)
        self.app_state.training_status.latest_checkpoint_path = path
        return path

    def load_checkpoint(self, path: str):
        payload = load_checkpoint(path)
        population_payload = payload.get("population")
        if population_payload:
            self._load_population(population_payload)
        else:
            self.population = self._create_population()
            leader = self.population[0]
            leader.model.load_state_dict(payload["model_state_dict"])
            leader.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.reference_model.load_state_dict(
            payload.get("reference_model_state_dict", payload["model_state_dict"])
        )
        self.reference_elo = payload.get("reference_elo", self.config.initial_elo)
        self.app_state.training_status.iteration = payload["iteration"]
        self.app_state.training_status.episode = payload["episode"]
        self.app_state.training_status.games_played = payload["games_played"]
        self.app_state.training_status.loss_history = payload.get("loss_history", [])
        self.app_state.training_status.forced_override_count = payload.get("forced_override_count", 0)
        self.app_state.training_status.model_decision_count = payload.get("model_decision_count", 0)
        self.app_state.training_status.forced_override_rate = (
            self.app_state.training_status.forced_override_count
            / self.app_state.training_status.model_decision_count
            if self.app_state.training_status.model_decision_count
            else 0.0
        )
        self.app_state.training_status.latest_checkpoint_path = path
        self.app_state.training_status.reference_checkpoint_path = (
            path if payload.get("is_reference") else self.app_state.training_status.reference_checkpoint_path
        )
        self.replay_buffer.load(payload.get("replay_buffer") or [])
        self.opponent_pool.load(payload.get("opponent_pool") or [])
        if payload.get("eval_metrics"):
            self.app_state.eval_metrics = EvalMetrics(**payload["eval_metrics"])
        self.app_state.training_status.replay_buffer_size = len(self.replay_buffer)
        self.app_state.training_status.opponent_pool_size = len(self.opponent_pool)
        self._assign_lineage_roles()
        self._sync_serving_member()
        self._set_reference_from_leader()
        self._record_event(f"Checkpoint loaded: {path.split('/')[-1]}")

    def _load_population(self, payload: list[dict[str, Any]]):
        self.population = []
        for item in payload:
            model = self._new_model()
            model.load_state_dict(item["model_state_dict"])
            optimizer = self._new_optimizer(model)
            optimizer.load_state_dict(item["optimizer_state_dict"])
            buffer = ReplayBuffer(
                capacity=self.config.replay_buffer_capacity,
                historical_fraction=self.config.replay_buffer_historical_fraction,
            )
            buffer.load(item.get("replay_buffer") or [])
            self.population.append(
                PopulationMember(
                    name=item["name"],
                    lineage=item.get("lineage", self._lineage_name(0)),
                    role=item.get("role", "seed"),
                    generation=item.get("generation", 0),
                    parent_names=item.get("parent_names", []),
                    model=model,
                    optimizer=optimizer,
                    replay_buffer=buffer,
                    elo=item.get("elo", self.config.initial_elo),
                    peak_elo=item.get("peak_elo", item.get("elo", self.config.initial_elo)),
                    games=item.get("games", 0),
                    wins=item.get("wins", 0),
                    losses=item.get("losses", 0),
                    draws=item.get("draws", 0),
                    score=item.get("score", 0.0),
                    last_result=item.get("last_result", "loaded"),
                )
            )
        if not self.population:
            self.population = self._create_population()

    def reset(self):
        if self._self_play_executor is not None:
            self._self_play_executor.shutdown(wait=False, cancel_futures=True)
            self._self_play_executor = None
            self._self_play_executor_workers = 0
        self.population = self._create_population()
        self.reference_model = self._new_model()
        self.reference_model.load_state_dict(self.population[0].model.state_dict())
        self.reference_elo = self.config.initial_elo
        self.replay_buffer.clear()
        self.opponent_pool.snapshots.clear()
        self.app_state.training_status = type(self.app_state.training_status)()
        self.app_state.eval_metrics = EvalMetrics()
        self.app_state.current_spectate = None
        self.app_state.spectate_games = {}
        self._assign_lineage_roles()
        self._sync_serving_member()
        self._set_reference_from_leader()

    def list_checkpoints(self) -> list[str]:
        return list_checkpoints()

    def _set_phase(self, phase: str, message: str, progress: int = 0, total: int = 0):
        self.app_state.training_status.current_phase = phase
        self.app_state.training_status.status_message = message
        self.app_state.training_status.phase_progress = progress
        self.app_state.training_status.phase_total = total

    def _record_event(self, message: str):
        events = self.app_state.training_status.recent_events
        events.append(message)
        self.app_state.training_status.recent_events = events[-12:]

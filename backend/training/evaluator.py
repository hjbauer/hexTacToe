from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from backend.config import EvalConfig, TrainingConfig
from backend.game.game_state import GameState
from backend.game.rules import apply_move
from backend.model.inference import ModelAgent
from backend.state.app_state import EvalMetrics
from backend.training.baselines import RandomBaseline
from backend.training.opponent_pool import OpponentPool


class EvaluationManager:
    def __init__(self, config: EvalConfig, training_config: TrainingConfig):
        self.config = config
        self.training_config = training_config

    async def run_evaluation(
        self,
        candidate_agent: ModelAgent,
        reference_agent: ModelAgent,
        opponent_pool: OpponentPool,
        iteration: int,
    ) -> EvalMetrics:
        return await asyncio.to_thread(
            self._run_sync_evaluation,
            candidate_agent,
            reference_agent,
            opponent_pool,
            iteration,
        )

    def _run_sync_evaluation(
        self,
        candidate_agent: ModelAgent,
        reference_agent: ModelAgent,
        opponent_pool: OpponentPool,
        iteration: int,
    ) -> EvalMetrics:
        versus_reference = self._match_agents(candidate_agent, reference_agent, self.config.eval_games, iteration)
        versus_random = self._match_against_random(candidate_agent, max(self.config.eval_games // 4, 4), iteration)
        pool_win_rates = []
        for snapshot in opponent_pool.snapshots[:3]:
            pool_agent = opponent_pool.agent_from_snapshot(snapshot)
            result = self._match_agents(candidate_agent, pool_agent, max(self.config.eval_games // 8, 2), iteration)
            pool_win_rates.append({"checkpoint": snapshot.path, "win_rate": result["win_rate"]})

        return EvalMetrics(
            last_eval_iteration=iteration,
            win_rate_vs_reference=versus_reference["win_rate"],
            win_rate_vs_random=versus_random["win_rate"],
            win_rate_as_red=versus_reference["as_red"],
            win_rate_as_blue=versus_reference["as_blue"],
            avg_game_length=versus_reference["avg_game_length"],
            loss_rate_vs_reference=versus_reference["loss_rate"],
            pool_win_rates=pool_win_rates,
            was_promoted=False,
        )

    def _match_agents(self, candidate_agent: ModelAgent, opponent_agent: ModelAgent, games: int, iteration: int) -> dict:
        wins = 0
        losses = 0
        draws = 0
        as_red_wins = 0
        as_blue_wins = 0
        total_moves = 0
        with ThreadPoolExecutor(max_workers=min(games, 4) or 1) as executor:
            futures = []
            for index in range(games):
                candidate_color = "red" if index % 2 == 0 else "blue"
                futures.append(
                    (
                        candidate_color,
                        executor.submit(self._play_match, candidate_agent, opponent_agent, candidate_color, iteration),
                    )
                )
            for candidate_color, future in futures:
                winner, move_count = future.result()
                total_moves += move_count
                if winner is None:
                    draws += 1
                elif winner == candidate_color:
                    wins += 1
                    if candidate_color == "red":
                        as_red_wins += 1
                    else:
                        as_blue_wins += 1
                else:
                    losses += 1
        return {
            "win_rate": wins / games if games else 0.0,
            "loss_rate": losses / games if games else 0.0,
            "draw_rate": draws / games if games else 0.0,
            "as_red": as_red_wins / max(games // 2, 1),
            "as_blue": as_blue_wins / max(games // 2, 1),
            "avg_game_length": total_moves / games if games else 0.0,
        }

    def _match_against_random(self, candidate_agent: ModelAgent, games: int, iteration: int) -> dict:
        random_baseline = RandomBaseline()
        wins = 0
        with ThreadPoolExecutor(max_workers=min(games, 4) or 1) as executor:
            futures = []
            for index in range(games):
                candidate_color = "red" if index % 2 == 0 else "blue"
                futures.append(
                    (
                        candidate_color,
                        executor.submit(self._play_match, candidate_agent, random_baseline, candidate_color, iteration),
                    )
                )
            for candidate_color, future in futures:
                winner, _ = future.result()
                if winner == candidate_color:
                    wins += 1
        return {"win_rate": wins / games if games else 0.0}

    def _play_match(self, candidate_agent, opponent_agent, candidate_color: str, iteration: int) -> tuple[str | None, int]:
        state = GameState()
        move_count = 0
        turn_limit = self.training_config.turn_limit_for_iteration(iteration)
        while not state.is_terminal and move_count < turn_limit:
            current_agent = candidate_agent if state.current_player == candidate_color else opponent_agent
            if isinstance(current_agent, ModelAgent):
                coord = current_agent.select_move(
                    state,
                    temperature=0.1,
                    greedy=True,
                    allow_tactical_override=True,
                ).coord
            else:
                coord = current_agent.select_move(state)
            state = apply_move(state, coord)
            move_count += 1
        return state.winner, move_count

    def should_promote(self, metrics: EvalMetrics) -> bool:
        return metrics.win_rate_vs_reference > self.config.promotion_threshold

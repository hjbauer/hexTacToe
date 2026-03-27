from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.game.game_state import GameState


def _safe_number(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return _safe_number(value)


@dataclass
class TrainingStatus:
    is_training: bool = False
    iteration: int = 0
    episode: int = 0
    games_played: int = 0
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_aux: float = 0.0
    replay_buffer_size: int = 0
    current_temperature: float = 1.0
    current_max_turns: int = 0
    latest_checkpoint_path: Optional[str] = None
    reference_checkpoint_path: Optional[str] = None
    loss_history: list[dict[str, float | int]] = field(default_factory=list)
    opponent_pool_size: int = 0
    entropy_warning: bool = False
    current_phase: str = "idle"
    phase_progress: int = 0
    phase_total: int = 0
    status_message: str = "Idle"
    recent_events: list[str] = field(default_factory=list)
    forced_override_count: int = 0
    model_decision_count: int = 0
    forced_override_rate: float = 0.0
    tactical_opportunity_count: int = 0
    tactical_blunder_count: int = 0
    tactical_blunder_rate: float = 0.0
    win_opportunity_count: int = 0
    missed_win_count: int = 0
    missed_win_rate: float = 0.0
    block_opportunity_count: int = 0
    missed_block_count: int = 0
    missed_block_rate: float = 0.0
    training_wins: int = 0
    training_losses: int = 0
    training_draws: int = 0
    training_win_rate: float = 0.0
    training_loss_rate: float = 0.0
    training_draw_rate: float = 0.0
    avg_game_length: float = 0.0
    population_size: int = 1
    leader_model_name: str = "model-1"
    leader_model_elo: float = 1200.0

    def to_dict(self, include_history: bool = True) -> dict[str, Any]:
        return _sanitize({
            "is_training": self.is_training,
            "iteration": self.iteration,
            "episode": self.episode,
            "games_played": self.games_played,
            "loss_policy": self.loss_policy,
            "loss_value": self.loss_value,
            "loss_aux": self.loss_aux,
            "replay_buffer_size": self.replay_buffer_size,
            "current_temperature": self.current_temperature,
            "current_max_turns": self.current_max_turns,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "reference_checkpoint_path": self.reference_checkpoint_path,
            "loss_history": self.loss_history if include_history else [],
            "opponent_pool_size": self.opponent_pool_size,
            "entropy_warning": self.entropy_warning,
            "current_phase": self.current_phase,
            "phase_progress": self.phase_progress,
            "phase_total": self.phase_total,
            "status_message": self.status_message,
            "recent_events": self.recent_events,
            "forced_override_count": self.forced_override_count,
            "model_decision_count": self.model_decision_count,
            "forced_override_rate": self.forced_override_rate,
            "tactical_opportunity_count": self.tactical_opportunity_count,
            "tactical_blunder_count": self.tactical_blunder_count,
            "tactical_blunder_rate": self.tactical_blunder_rate,
            "win_opportunity_count": self.win_opportunity_count,
            "missed_win_count": self.missed_win_count,
            "missed_win_rate": self.missed_win_rate,
            "block_opportunity_count": self.block_opportunity_count,
            "missed_block_count": self.missed_block_count,
            "missed_block_rate": self.missed_block_rate,
            "training_wins": self.training_wins,
            "training_losses": self.training_losses,
            "training_draws": self.training_draws,
            "training_win_rate": self.training_win_rate,
            "training_loss_rate": self.training_loss_rate,
            "training_draw_rate": self.training_draw_rate,
            "avg_game_length": self.avg_game_length,
            "population_size": self.population_size,
            "leader_model_name": self.leader_model_name,
            "leader_model_elo": self.leader_model_elo,
        })


@dataclass
class EvalMetrics:
    last_eval_iteration: int = 0
    win_rate_vs_reference: float = 0.0
    win_rate_vs_random: float = 0.0
    win_rate_as_red: float = 0.0
    win_rate_as_blue: float = 0.0
    avg_game_length: float = 0.0
    loss_rate_vs_reference: float = 0.0
    pool_win_rates: list[dict[str, float | str]] = field(default_factory=list)
    promotion_history: list[dict[str, float | int | str]] = field(default_factory=list)
    was_promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _sanitize({
            "last_eval_iteration": self.last_eval_iteration,
            "win_rate_vs_reference": self.win_rate_vs_reference,
            "win_rate_vs_random": self.win_rate_vs_random,
            "win_rate_as_red": self.win_rate_as_red,
            "win_rate_as_blue": self.win_rate_as_blue,
            "avg_game_length": self.avg_game_length,
            "loss_rate_vs_reference": self.loss_rate_vs_reference,
            "pool_win_rates": self.pool_win_rates,
            "promotion_history": self.promotion_history,
            "was_promoted": self.was_promoted,
        })


@dataclass
class HumanGameSession:
    game_id: str
    human_color: str
    ai_color: Optional[str]
    opponent_model_name: str = "leader"
    opponent_model_elo: float = 1200.0
    state: GameState = field(default_factory=GameState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "human_color": self.human_color,
            "ai_color": self.ai_color,
            "opponent_model_name": self.opponent_model_name,
            "opponent_model_elo": self.opponent_model_elo,
            "state": self.state.to_dict(),
        }


@dataclass
class AppState:
    training_status: TrainingStatus = field(default_factory=TrainingStatus)
    eval_metrics: EvalMetrics = field(default_factory=EvalMetrics)
    human_games: dict[str, HumanGameSession] = field(default_factory=dict)
    current_spectate: Optional[dict[str, Any]] = None
    spectate_games: dict[str, dict[str, Any]] = field(default_factory=dict)
    trainer: Any = None
    ws_manager: Any = None
    training_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

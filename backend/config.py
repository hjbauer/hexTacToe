from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    node_feature_dim: int = 11
    hidden_dim: int = 64
    num_gnn_layers: int = 3
    num_attention_heads: int = 2
    dropout: float = 0.1
    edge_dim: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalConfig:
    eval_interval: int = 10
    eval_games: int = 12
    promotion_threshold: float = 0.60
    opponent_pool_max_size: int = 15
    snapshot_interval: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplorationConfig:
    initial_temperature: float = 1.5
    final_temperature: float = 0.3
    warmup_iterations: int = 100
    greedy_after_move: int = 20
    use_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    min_policy_entropy_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    device: str = "auto"
    self_play_device: str = "cpu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    replay_buffer_capacity: int = 100_000
    replay_buffer_historical_fraction: float = 0.2
    tactical_replay_fraction: float = 0.35
    batch_size: int = 64
    self_play_games_per_iteration: int = 24
    parallel_self_play_workers: int = 8
    local_spectate_games_per_batch: int = 1
    self_play_worker_torch_threads: int = 1
    parallel_gradient_workers: int = 1
    gradient_steps_per_iteration: int = 2
    max_models_to_train_per_iteration: int = 4
    min_buffer_size_to_train: int = 32
    checkpoint_interval: int = 10
    keep_last_checkpoints: int = 1
    auto_resume_from_latest_checkpoint: bool = True
    checkpoint_s3_bucket: str = "hex-tactoe-checkpoints-hjbauer"
    checkpoint_s3_prefix: str = "checkpoints"
    checkpoint_s3_keep_last: int = 2
    checkpoint_recent_samples: int = 1024
    checkpoint_historical_samples: int = 256
    checkpoint_member_recent_samples: int = 256
    checkpoint_member_historical_samples: int = 64
    max_turns_per_game: int = 280
    initial_max_turns_per_game: int = 180
    max_turn_growth_interval: int = 20
    max_turn_growth_step: int = 10
    population_size: int = 6
    lineage_count: int = 3
    initial_elo: float = 1200.0
    elo_k_factor: float = 24.0
    evolution_interval: int = 8
    replacements_per_evolution: int = 3
    cross_lineage_parent_rate: float = 0.35
    mutation_std: float = 0.01
    same_lineage_match_weight: float = 0.45
    cross_lineage_match_weight: float = 0.55
    heuristic_guidance_warmup_iterations: int = 40
    heuristic_guidance_initial_weight: float = 0.9
    heuristic_guidance_final_weight: float = 0.0
    use_tactical_selector: bool = True
    shaped_reward_discount: float = 0.95
    terminal_win_reward: float = 12.0
    terminal_loss_reward: float = -20.0
    draw_reward: float = -4.0
    line_extension_reward: float = 0.8
    block_threat_reward: float = 3.0
    adjacency_reward: float = 0.1
    forced_override_penalty: float = 6.0
    next_turn_threat_reward: float = 1.0
    multi_threat_bonus: float = 1.75
    trivial_win_penalty: float = 0.0
    trivial_loss_penalty: float = 9.0
    threat_exposure_penalty: float = 3.5
    compactness_reward: float = 0.4
    colony_penalty: float = 0.75
    isolated_stone_penalty: float = 0.2
    colony_seed_reward: float = 0.9
    multi_front_growth_reward: float = 1.1
    multi_front_threat_bonus: float = 1.0
    hindsight_corrections_per_loss: int = 2
    tactical_corrections_per_game: int = 4
    auxiliary_loss_weight: float = 0.25
    rotation_augmentation_samples: int = 1
    use_mixed_precision: bool = True
    reward_normalization_scale: float = 10.0
    opening_random_moves: int = 4
    opening_random_move_probability: float = 0.35
    pool_strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "recent": 0.5,
            "uniform": 0.3,
            "oldest": 0.2,
        }
    )
    opponent_sampling_weights: dict[str, float] = field(
        default_factory=lambda: {
            "population": 0.45,
            "reference": 0.25,
            "pool": 0.2,
            "random": 0.1,
        }
    )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["model"] = self.model.to_dict()
        data["eval"] = self.eval.to_dict()
        data["exploration"] = self.exploration.to_dict()
        return data

    def turn_limit_for_iteration(self, iteration: int) -> int:
        base = min(self.initial_max_turns_per_game, self.max_turns_per_game)
        if self.max_turn_growth_interval <= 0 or self.max_turn_growth_step <= 0:
            return self.max_turns_per_game
        growth_steps = max(iteration, 0) // self.max_turn_growth_interval
        return min(
            self.max_turns_per_game,
            base + (growth_steps * self.max_turn_growth_step),
        )

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, N_ACTIONS, ObservationSwitches, SensoryGridEnv
from models.shared.r2d2_interface import (
    ModelConfig,
    RecurrentFusionLSTMC51DuelingQNetwork,
    build_model_from_checkpoint,
    choose_device,
    init_runtime_context,
    obs_to_state,
    onehot_action,
    predict_action_for_gui,
    reset_runtime_context,
    update_runtime_context_after_env_step,
)
from models.model_based.dqn.future_coverage_credit_dqn.train import (
    build_default_switches,
    build_training_env as dqn_build_training_env,
    empty_eval_metrics,
    episode_diagnostic_metric_names,
    eval_metric_names,
    finalize_episode_diagnostics,
    init_episode_diagnostics,
    nanmean,
    structured_exploration_probs,
    update_episode_diagnostics,
)


TRAINER_GUI_INTERFACE_VERSION = "recurrent-distributional-dqn"
TRAINER_DISPLAY_NAME = "Recurrent distributional Double DQN"


def finite_or(value: float, fallback: float) -> float:
    value = float(value)
    return float(value) if np.isfinite(value) else float(fallback)


def positive_finite_or(value: float, fallback: float) -> float:
    value = finite_or(value, fallback)
    return float(max(value, float(fallback)))


@dataclass
class TrainConfig:
    episodes: int = 1200
    seed: int = 7
    gamma: float = 0.99
    learning_rate: float = 1.5e-4
    weight_decay: float = 1e-5
    max_gradient_norm: float = 10.0
    save_dir: str = "runs/recurrent_distributional_dqn"
    device: str = "auto"

    replay_capacity_episodes: int = 1600
    replay_priority_alpha: float = 0.80
    batch_size: int = 8
    burn_in: int = 30
    unroll_len: int = 60
    n_step_return: int = 5
    train_after_episodes: int = 20
    train_updates_per_episode: int = 4

    target_soft_tau: float = 0.02
    target_hard_sync_every_episodes: int = 40

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 90_000
    use_epsilon_pulses: bool = True
    epsilon_pulse_trigger_epsilon: float = 0.25
    epsilon_pulse_amplitude: float = 0.10
    epsilon_pulse_cycle_steps: int = 12_000
    epsilon_pulse_decay_cycles: float = 10.0
    use_state_adaptive_exploration: bool = False
    exploration_stagnation_trigger_steps: int = 6
    exploration_stagnation_ramp_steps: int = 10
    exploration_stagnation_epsilon_bonus: float = 0.12
    exploration_low_progress_window: int = 12
    exploration_low_progress_threshold: float = 0.015
    exploration_low_progress_epsilon_bonus: float = 0.08
    exploration_turn_stagnation_epsilon_bonus: float = 0.04
    exploration_no_move_epsilon_bonus: float = 0.04

    eval_every: int = 20
    eval_episodes: int = 10
    holdout_eval_episodes: int = 10
    eval_seed_start: int = 1000
    holdout_eval_seed_start: int = 2000
    save_eval_checkpoints: bool = True

    conv_channels_1: int = 16
    conv_channels_2: int = 32
    vision_embed_dim: int = 96
    thermal_embed_dim: int = 48
    sensing_embed_dim: int = 64
    scalar_state_embed_dim: int = 64
    obs_embed_dim: int = 256
    lstm_hidden_dim: int = 256
    head_hidden_dim: int = 128
    num_atoms: int = 51
    v_min: float = -600.0
    v_max: float = 50.0

    training_survival_bonus: float = 0.60
    use_exploration_reward_shaping: bool = True
    train_coverage_delta_reward_scale: float = 6.0
    train_early_progress_reward_scale: float = 4.0
    train_shaping_step_horizon: int = 150
    train_no_progress_energy_penalty_scale: float = 0.025
    train_revisit_penalty_scale: float = 0.010
    use_episodic_exploration_bonus: bool = True
    train_obs_novelty_bonus_scale: float = 0.012

    use_structured_exploration: bool = True
    exploration_softmax_temperature: float = 0.75
    exploration_forward_unvisited_bonus: float = 2.75
    exploration_forward_revisit_penalty: float = 0.85
    exploration_visible_hazard_penalty: float = 0.18
    exploration_known_hazard_penalty: float = 0.40
    exploration_post_turn_forward_bonus: float = 1.35
    exploration_repeat_turn_penalty: float = 0.45
    exploration_post_bump_forward_penalty: float = 0.35
    exploration_direction_score_scale: float = 0.85
    exploration_direction_unvisited_weight: float = 1.00
    exploration_direction_visited_penalty: float = 0.30
    exploration_direction_visible_hazard_penalty: float = 1.20
    exploration_direction_hazard_memory_penalty: float = 0.80
    exploration_frontier_score_scale: float = 0.0
    exploration_frontier_unvisited_weight: float = 0.85
    exploration_frontier_frontier_bonus: float = 0.55
    exploration_frontier_visited_penalty: float = 0.18
    exploration_frontier_visible_hazard_penalty: float = 1.15
    exploration_frontier_known_hazard_penalty: float = 0.90
    exploration_frontier_depth_decay: float = 0.72
    exploration_frontier_side_weight: float = 0.60
    exploration_frontier_turn_penalty: float = 0.05


@dataclass
class ReplayEpisode:
    transitions: List[Dict[str, object]]
    base_priority: float
    sequence_priorities: Dict[int, float]


class PrioritizedSequenceReplayBuffer:
    def __init__(self, capacity_episodes: int, priority_alpha: float = 0.80):
        self.capacity_episodes = int(capacity_episodes)
        self.priority_alpha = float(priority_alpha)
        self.episodes: List[ReplayEpisode] = []

    def add_episode(self, transitions: List[Dict[str, object]], priority: float) -> None:
        if not transitions:
            return
        if len(self.episodes) >= self.capacity_episodes:
            self.episodes.pop(0)
        base_priority = positive_finite_or(priority, 1e-3)
        self.episodes.append(
            ReplayEpisode(
                transitions=transitions,
                base_priority=base_priority,
                sequence_priorities={},
            )
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def _episode_priority(self, entry: ReplayEpisode) -> float:
        finite_sequence_priorities = [
            positive_finite_or(priority, 0.0)
            for priority in entry.sequence_priorities.values()
            if np.isfinite(float(priority))
        ]
        extra = max(finite_sequence_priorities, default=0.0)
        base_priority = positive_finite_or(entry.base_priority, 1e-3)
        return float(max(base_priority, extra, 1e-3))

    def _sample_episode_index(self) -> int:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty replay buffer.")
        weights = [
            positive_finite_or(self._episode_priority(entry) ** self.priority_alpha, 1e-3)
            for entry in self.episodes
        ]
        if not np.isfinite(np.sum(weights)) or sum(weights) <= 0.0:
            return int(random.randrange(len(self.episodes)))
        return int(random.choices(range(len(self.episodes)), weights=weights, k=1)[0])

    def _window_priority(self, entry: ReplayEpisode, start: int, learn_len: int) -> float:
        end = min(len(entry.transitions), start + learn_len)
        if end <= start:
            base = 0.05
        else:
            base = float(
                np.mean(
                    [
                        positive_finite_or(float(tr.get("sample_weight", 1.0)), 0.05)
                        for tr in entry.transitions[start:end]
                    ]
                )
            )
        override = positive_finite_or(float(entry.sequence_priorities.get(start, 0.0)), 0.0)
        return float(max(positive_finite_or(base, 0.05), override, 0.05))

    def _sample_start_index(self, entry: ReplayEpisode, learn_len: int) -> int:
        if len(entry.transitions) <= 1:
            return 0
        starts = list(range(len(entry.transitions)))
        weights = [self._window_priority(entry, start, learn_len) for start in starts]
        if not np.isfinite(np.sum(weights)) or sum(weights) <= 0.0:
            return int(random.choice(starts))
        return int(random.choices(starts, weights=weights, k=1)[0])

    def update_priorities(self, refs: List[Tuple[int, int]], priorities: List[float]) -> None:
        for (episode_idx, start), priority in zip(refs, priorities):
            if not (0 <= episode_idx < len(self.episodes)):
                continue
            entry = self.episodes[episode_idx]
            entry.sequence_priorities[int(start)] = positive_finite_or(priority, 0.05)
            if len(entry.sequence_priorities) > 256:
                worst_start = min(entry.sequence_priorities, key=entry.sequence_priorities.get)
                del entry.sequence_priorities[worst_start]

    def sample(self, batch_size: int, total_len: int, learn_len: int) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int]]]:
        batch_states: List[List[Dict[str, np.ndarray]]] = []
        batch_actions: List[np.ndarray] = []
        batch_valid: List[np.ndarray] = []
        batch_n_step_returns: List[np.ndarray] = []
        batch_n_step_discounts: List[np.ndarray] = []
        batch_n_step_dones: List[np.ndarray] = []
        batch_n_step_steps: List[np.ndarray] = []
        refs: List[Tuple[int, int]] = []

        for _ in range(batch_size):
            episode_idx = self._sample_episode_index()
            entry = self.episodes[episode_idx]
            start = self._sample_start_index(entry, learn_len)
            refs.append((episode_idx, start))
            episode = entry.transitions
            last_next_state = copy_state(episode[-1]["next_state"])

            seq_states = []
            seq_actions = np.zeros((total_len,), dtype=np.int64)
            seq_valid = np.zeros((total_len,), dtype=np.float32)
            seq_n_step_returns = np.zeros((total_len,), dtype=np.float32)
            seq_n_step_discounts = np.zeros((total_len,), dtype=np.float32)
            seq_n_step_dones = np.ones((total_len,), dtype=np.float32)
            seq_n_step_steps = np.ones((total_len,), dtype=np.int64)

            for j in range(total_len):
                idx = start + j
                if idx < len(episode):
                    tr = episode[idx]
                    seq_states.append(copy_state(tr["state"]))
                    seq_actions[j] = int(tr["action"])
                    seq_valid[j] = 1.0
                    seq_n_step_returns[j] = float(tr.get("n_step_return", 0.0))
                    seq_n_step_discounts[j] = float(tr.get("n_step_discount", 0.0))
                    seq_n_step_dones[j] = float(tr.get("n_step_done", 1.0))
                    seq_n_step_steps[j] = int(tr.get("n_step_steps", 1))
                else:
                    seq_states.append(copy_state(last_next_state))

            batch_states.append(seq_states)
            batch_actions.append(seq_actions)
            batch_valid.append(seq_valid)
            batch_n_step_returns.append(seq_n_step_returns)
            batch_n_step_discounts.append(seq_n_step_discounts)
            batch_n_step_dones.append(seq_n_step_dones)
            batch_n_step_steps.append(seq_n_step_steps)

        def collate_state_sequences(state_sequences: List[List[Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
            keys = state_sequences[0][0].keys()
            out: Dict[str, np.ndarray] = {}
            for key in keys:
                out[key] = np.stack([
                    np.stack([step[key] for step in seq]).astype(np.float32)
                    for seq in state_sequences
                ]).astype(np.float32)
            return out

        return (
            {
                "states": collate_state_sequences(batch_states),
                "actions": np.stack(batch_actions).astype(np.int64),
                "valid": np.stack(batch_valid).astype(np.float32),
                "n_step_return": np.stack(batch_n_step_returns).astype(np.float32),
                "n_step_discount": np.stack(batch_n_step_discounts).astype(np.float32),
                "n_step_done": np.stack(batch_n_step_dones).astype(np.float32),
                "n_step_steps": np.stack(batch_n_step_steps).astype(np.int64),
            },
            refs,
        )


def get_model_config(cfg: TrainConfig) -> ModelConfig:
    return ModelConfig(
        conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2,
        vision_embed_dim=cfg.vision_embed_dim,
        thermal_embed_dim=cfg.thermal_embed_dim,
        sensing_embed_dim=cfg.sensing_embed_dim,
        scalar_state_embed_dim=cfg.scalar_state_embed_dim,
        obs_embed_dim=cfg.obs_embed_dim,
        lstm_hidden_dim=cfg.lstm_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        num_atoms=cfg.num_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
    )


def build_training_env(cfg: TrainConfig | None = None) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    return dqn_build_training_env(cfg)


def build_network(env: SensoryGridEnv, switches: ObservationSwitches, cfg: TrainConfig) -> RecurrentFusionLSTMC51DuelingQNetwork:
    return RecurrentFusionLSTMC51DuelingQNetwork(
        patch_size=env.config.patch_size,
        num_actions=N_ACTIONS,
        cfg=get_model_config(cfg),
        use_vision=switches.include_vision,
        use_temperature_patch=switches.include_temperature_patch,
        use_smell_patch=switches.include_smell_patch,
        use_visited_memory=switches.include_visited_memory,
        use_hazard_memory=switches.include_hazard_memory,
        scalar_state_dim=12,
    )


def copy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in state.items()}


def state_seq_batch_to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def observation_novelty_key(obs: Dict[str, Any]) -> bytes:
    vision = np.asarray(obs.get("vision"), dtype=np.uint8)
    visited = (np.asarray(obs.get("visited_patch", np.zeros_like(vision, dtype=np.float32)), dtype=np.float32) > 0.5).astype(np.uint8)
    hazard = (np.asarray(obs.get("hazard_patch", np.zeros_like(vision, dtype=np.float32)), dtype=np.float32) >= 0.25).astype(np.uint8)
    direction = np.asarray(obs.get("direction_onehot", np.zeros((4,), dtype=np.float32)), dtype=np.float32).round().astype(np.uint8)
    no_move = min(int(obs.get("consecutive_no_move_steps", 0)), 3)
    turn_streak = min(int(obs.get("consecutive_turn_steps", 0)), 3)
    return b"".join((vision.tobytes(), visited.tobytes(), hazard.tobytes(), direction.tobytes(), bytes((no_move, turn_streak))))


def observation_novelty_bonus(obs: Dict[str, Any], novelty_counts: Dict[bytes, int]) -> float:
    key = observation_novelty_key(obs)
    count = int(novelty_counts.get(key, 0))
    novelty_counts[key] = count + 1
    return float(1.0 / np.sqrt(count + 1.0))


def transition_energy_spent(info: Dict[str, object]) -> float:
    return float(
        float(info.get("time_base_cost", 0.0))
        + float(info.get("forward_extra_cost", 0.0))
        + float(info.get("turn_extra_cost", 0.0))
        + float(info.get("thermal_extra_this_tick", 0.0))
    )


def transition_sample_weight(
    coverage_delta: float,
    obs_novelty: float,
    energy_spent: float,
    revisit: bool,
    wall_hit: bool,
    stagnating: bool,
) -> float:
    weight = 1.0
    weight += 120.0 * max(0.0, float(coverage_delta))
    weight += 0.65 * max(0.0, float(obs_novelty))
    if float(coverage_delta) > 1e-9:
        weight += 1.5
    weight -= 0.45 * float(revisit)
    weight -= 0.60 * float(wall_hit)
    weight -= 0.35 * float(stagnating)
    weight -= 0.20 * max(0.0, float(energy_spent))
    return float(max(weight, 0.05))


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / max(1, cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def epsilon_with_pulses(step: int, cfg: TrainConfig) -> float:
    base = linear_epsilon(step, cfg)
    if not cfg.use_epsilon_pulses:
        return base
    trigger = float(cfg.epsilon_pulse_trigger_epsilon)
    if base > trigger:
        return base
    denom = cfg.epsilon_end - cfg.epsilon_start
    if abs(denom) < 1e-9:
        return base
    trigger_frac = float(np.clip((trigger - cfg.epsilon_start) / denom, 0.0, 1.0))
    trigger_step = int(round(trigger_frac * cfg.epsilon_decay_steps))
    cycle_steps = max(1, int(cfg.epsilon_pulse_cycle_steps))
    elapsed = max(0, int(step) - trigger_step)
    phase = (elapsed % cycle_steps) / cycle_steps
    triangle = 1.0 - abs(2.0 * phase - 1.0)
    cycle_index = elapsed / cycle_steps
    decay_multiplier = float(np.exp(-cycle_index / max(float(cfg.epsilon_pulse_decay_cycles), 1e-6)))
    pulse = float(cfg.epsilon_pulse_amplitude) * triangle * decay_multiplier
    return float(min(cfg.epsilon_start, base + pulse))


def adaptive_exploration_epsilon(step: int, obs: Dict[str, Any], episode_diag: Dict[str, object], cfg: TrainConfig) -> float:
    base = epsilon_with_pulses(step, cfg)
    if not cfg.use_state_adaptive_exploration:
        return base
    bonus = 0.0
    steps_since_new_tile = int(episode_diag.get("steps_since_new_tile", 0))
    trigger = max(1, int(cfg.exploration_stagnation_trigger_steps))
    if steps_since_new_tile >= trigger:
        ramp_steps = max(1, int(cfg.exploration_stagnation_ramp_steps))
        streak_frac = np.clip((steps_since_new_tile - trigger + 1) / ramp_steps, 0.0, 1.0)
        bonus += float(cfg.exploration_stagnation_epsilon_bonus) * float(streak_frac)
    recent_coverages = episode_diag.get("recent_coverages")
    if isinstance(recent_coverages, list):
        history = recent_coverages
    else:
        history = list(float(x) for x in recent_coverages) if recent_coverages is not None else []
    if len(history) >= 2:
        window = min(max(1, int(cfg.exploration_low_progress_window)), len(history) - 1)
        recent_gain = max(0.0, history[-1] - history[-1 - window])
        threshold = max(float(cfg.exploration_low_progress_threshold), 1e-6)
        if recent_gain < threshold:
            bonus += float(cfg.exploration_low_progress_epsilon_bonus) * float(np.clip(1.0 - recent_gain / threshold, 0.0, 1.0))
    turn_streak = int(obs.get("consecutive_turn_steps", 0))
    if turn_streak >= 2:
        bonus += float(cfg.exploration_turn_stagnation_epsilon_bonus) * float(np.clip((turn_streak - 1) / 2.0, 0.0, 1.0))
    no_move_steps = int(obs.get("consecutive_no_move_steps", 0))
    if no_move_steps >= 2:
        bonus += float(cfg.exploration_no_move_epsilon_bonus) * float(np.clip((no_move_steps - 1) / 2.0, 0.0, 1.0))
    return float(min(cfg.epsilon_start, base + (1.0 - base) * bonus))


def composite_eval_score(eval_metrics: Dict[str, float], cfg: TrainConfig) -> float:
    def metric(name: str) -> float:
        value = float(eval_metrics.get(name, 0.0))
        return 0.0 if np.isnan(value) else value

    return (
        1.00 * metric("eval_coverage_mean")
        + 0.20 * metric("eval_coverage_at_step_50_mean")
        + 0.35 * metric("eval_coverage_at_step_100_mean")
        + 0.55 * metric("eval_coverage_at_step_150_mean")
    )


def selection_key(eval_metrics: Dict[str, float], cfg: TrainConfig) -> Tuple[float, float, float, float, float]:
    return (
        composite_eval_score(eval_metrics, cfg),
        float(eval_metrics.get("eval_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_150_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_100_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_50_mean", float("-inf"))),
    )


def project_categorical(
    next_action_logits: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(next_action_logits, dim=-1)
    num_atoms = support.numel()
    v_min = float(support[0].item())
    v_max = float(support[-1].item())
    delta_z = (v_max - v_min) / max(1, num_atoms - 1)

    tz = rewards.unsqueeze(-1) + (1.0 - dones.unsqueeze(-1)) * discounts.unsqueeze(-1) * support.view(1, 1, -1)
    tz = torch.clamp(tz, v_min, v_max)
    b = (tz - v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()

    projected = torch.zeros_like(probs)
    flat_projected = projected.view(-1, num_atoms)
    flat_probs = probs.view(-1, num_atoms)
    flat_b = b.view(-1, num_atoms)
    flat_lower = lower.view(-1, num_atoms)
    flat_upper = upper.view(-1, num_atoms)

    batch_offset = torch.arange(flat_projected.shape[0], device=probs.device).unsqueeze(1) * num_atoms
    flat_projected_flat = flat_projected.view(-1)

    lower_weight = (flat_upper.float() - flat_b)
    upper_weight = (flat_b - flat_lower.float())
    same_bucket = flat_lower == flat_upper
    lower_weight = torch.where(same_bucket, torch.ones_like(lower_weight), lower_weight)
    upper_weight = torch.where(same_bucket, torch.zeros_like(upper_weight), upper_weight)

    flat_projected_flat.index_add_(
        0,
        (flat_lower + batch_offset).reshape(-1),
        (flat_probs * lower_weight).reshape(-1),
    )
    flat_projected_flat.index_add_(
        0,
        (flat_upper + batch_offset).reshape(-1),
        (flat_probs * upper_weight).reshape(-1),
    )
    return projected


def train_step(
    online_net: RecurrentFusionLSTMC51DuelingQNetwork,
    target_net: RecurrentFusionLSTMC51DuelingQNetwork,
    optimizer: optim.Optimizer,
    buffer: PrioritizedSequenceReplayBuffer,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[float, float, float]:
    total_len = cfg.burn_in + cfg.unroll_len + cfg.n_step_return
    batch, refs = buffer.sample(cfg.batch_size, total_len, cfg.burn_in + cfg.unroll_len)
    states_t = state_seq_batch_to_torch(batch["states"], device)
    actions_t = torch.from_numpy(batch["actions"]).to(device)
    valid_t = torch.from_numpy(batch["valid"]).to(device)
    n_step_return_t = torch.from_numpy(batch["n_step_return"]).to(device)
    n_step_discount_t = torch.from_numpy(batch["n_step_discount"]).to(device)
    n_step_done_t = torch.from_numpy(batch["n_step_done"]).to(device)
    n_step_steps_t = torch.from_numpy(batch["n_step_steps"]).to(device)

    if cfg.burn_in > 0:
        burn_states = {k: v[:, :cfg.burn_in] for k, v in states_t.items()}
        with torch.no_grad():
            _, _, hidden_online = online_net.forward_sequence(burn_states, None)
            _, _, hidden_target = target_net.forward_sequence(burn_states, None)
    else:
        hidden_online = None
        hidden_target = None

    main_states = {k: v[:, cfg.burn_in:] for k, v in states_t.items()}
    logits_full, q_full, _ = online_net.forward_sequence(main_states, hidden_online)
    with torch.no_grad():
        target_logits_full, target_q_full, _ = target_net.forward_sequence(main_states, hidden_target)

    learn_len = cfg.unroll_len
    current_logits = logits_full[:, :learn_len]
    current_q = q_full[:, :learn_len]
    current_actions = actions_t[:, cfg.burn_in : cfg.burn_in + learn_len]
    current_valid = valid_t[:, cfg.burn_in : cfg.burn_in + learn_len] > 0.5
    current_returns = n_step_return_t[:, cfg.burn_in : cfg.burn_in + learn_len]
    current_discounts = n_step_discount_t[:, cfg.burn_in : cfg.burn_in + learn_len]
    current_dones = n_step_done_t[:, cfg.burn_in : cfg.burn_in + learn_len]
    current_steps = n_step_steps_t[:, cfg.burn_in : cfg.burn_in + learn_len].long()

    if not bool(current_valid.any().item()):
        return 0.0, 0.0, 0.0

    chosen_logits = current_logits.gather(
        2,
        current_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, online_net.num_atoms),
    ).squeeze(2)
    chosen_q = current_q.gather(2, current_actions.unsqueeze(-1)).squeeze(-1)

    time_offsets = torch.arange(learn_len, device=device).view(1, learn_len)
    bootstrap_indices = torch.clamp(time_offsets + current_steps, max=logits_full.shape[1] - 1)
    bootstrap_indices = bootstrap_indices.expand(current_actions.shape[0], -1)
    gather_index_q = bootstrap_indices.unsqueeze(-1).expand(-1, -1, q_full.shape[-1])
    with torch.no_grad():
        bootstrap_online_q = q_full.detach().gather(1, gather_index_q)
        bootstrap_actions = torch.argmax(bootstrap_online_q, dim=-1)
        gather_index_logits = bootstrap_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_logits_full.shape[2], target_logits_full.shape[3])
        bootstrap_target_logits_all = target_logits_full.gather(1, gather_index_logits)
        bootstrap_target_logits = bootstrap_target_logits_all.gather(
            2,
            bootstrap_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, online_net.num_atoms),
        ).squeeze(2)
        target_dist = project_categorical(
            bootstrap_target_logits,
            current_returns,
            current_discounts,
            current_dones,
            online_net.support,
        )

    log_probs = torch.log_softmax(chosen_logits, dim=-1)
    per_step_loss = -(target_dist * log_probs).sum(dim=-1)
    loss = per_step_loss[current_valid].mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_gradient_norm)
    optimizer.step()

    with torch.no_grad():
        bootstrap_target_q = target_q_full.gather(1, gather_index_q)
        bootstrap_target_q = bootstrap_target_q.gather(2, bootstrap_actions.unsqueeze(-1)).squeeze(-1)
        expected_target_q = current_returns + (1.0 - current_dones) * current_discounts * bootstrap_target_q
        td_error = torch.abs(expected_target_q - chosen_q)
        sequence_priorities: List[float] = []
        for i in range(td_error.shape[0]):
            mask = current_valid[i]
            if bool(mask.any().item()):
                sequence_priorities.append(float(td_error[i][mask].mean().item()) + 0.05)
            else:
                sequence_priorities.append(0.05)
        buffer.update_priorities(refs, sequence_priorities)
        mean_td = float(td_error[current_valid].mean().item())

    return float(loss.item()), float(loss.item()), mean_td


@torch.no_grad()
def evaluate_policy(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: RecurrentFusionLSTMC51DuelingQNetwork,
    device: torch.device,
    episodes: int = 5,
    seed_start: int = 1000,
    metric_prefix: str = "eval",
) -> Dict[str, float]:
    if episodes <= 0:
        return empty_eval_metrics(metric_prefix)

    rewards: List[float] = []
    coverages: List[float] = []
    final_healths: List[float] = []
    lengths: List[float] = []
    survived_flags: List[float] = []
    death_flags: List[float] = []
    diag_metrics: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, reset_info = env.reset(seed=seed_start + ep)
        state = obs_to_state(obs, env.config, switches, prev_action=np.zeros((N_ACTIONS,), dtype=np.float32), prev_reward=0.0, episode_step=0)
        done = False
        ep_reward = 0.0
        hidden = None
        last_info = None
        prev_action = np.zeros((N_ACTIONS,), dtype=np.float32)
        prev_reward = 0.0
        episode_step = 0
        novelty_counts: Dict[bytes, int] = {}
        diag = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))

        while not done:
            diag["obs_novelty_total"] = float(diag["obs_novelty_total"]) + observation_novelty_bonus(obs, novelty_counts)
            state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
            _, q_values, hidden = net.forward_sequence(state_t, hidden)
            action = int(torch.argmax(q_values[:, -1], dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            update_episode_diagnostics(diag, action, info)
            ep_reward += reward
            last_info = info
            done = terminated or truncated
            episode_step = int(info.get("steps", episode_step + 1))
            prev_action = onehot_action(action)
            prev_reward = float(reward)
            state = obs_to_state(next_obs, env.config, switches, prev_action=prev_action, prev_reward=prev_reward, episode_step=episode_step)
            obs = next_obs

        assert last_info is not None
        rewards.append(ep_reward)
        coverages.append(float(last_info["coverage"]))
        final_healths.append(float(last_info["health"]))
        lengths.append(float(last_info["steps"]))
        survived_flags.append(1.0 if (last_info["truncated"] and not last_info["terminated"]) else 0.0)
        death_flags.append(1.0 if last_info["terminated"] else 0.0)
        diag_metrics.append(finalize_episode_diagnostics(diag))

    metrics = {
        f"{metric_prefix}_reward_mean": float(np.mean(rewards)),
        f"{metric_prefix}_coverage_mean": float(np.mean(coverages)),
        f"{metric_prefix}_final_health_mean": float(np.mean(final_healths)),
        f"{metric_prefix}_length_mean": float(np.mean(lengths)),
        f"{metric_prefix}_survival_rate": float(np.mean(survived_flags)),
        f"{metric_prefix}_death_rate": float(np.mean(death_flags)),
    }
    for name in episode_diagnostic_metric_names():
        metrics[f"{metric_prefix}_{name}_mean"] = nanmean([item[name] for item in diag_metrics])
    return metrics


def checkpoint_payload(
    online_net: RecurrentFusionLSTMC51DuelingQNetwork,
    input_dim: int,
    switches: ObservationSwitches,
    env: SensoryGridEnv,
    cfg: TrainConfig,
    episode: int,
    global_step: int,
    eval_metrics: Dict[str, float] | None = None,
    holdout_eval_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    metrics = eval_metrics or {}
    holdout_metrics = holdout_eval_metrics or {}
    return {
        "model_state_dict": online_net.state_dict(),
        "model_arch": "patch_fusion_lstm_dueling_double_c51_recurrent_distributional_dqn",
        "model_kwargs": {
            "patch_size": env.config.patch_size,
            "num_actions": N_ACTIONS,
            "use_vision": switches.include_vision,
            "use_temperature_patch": switches.include_temperature_patch,
            "use_smell_patch": switches.include_smell_patch,
            "use_visited_memory": switches.include_visited_memory,
            "use_hazard_memory": switches.include_hazard_memory,
            "scalar_state_dim": 12,
            "conv_channels_1": cfg.conv_channels_1,
            "conv_channels_2": cfg.conv_channels_2,
            "vision_embed_dim": cfg.vision_embed_dim,
            "thermal_embed_dim": cfg.thermal_embed_dim,
            "sensing_embed_dim": cfg.sensing_embed_dim,
            "scalar_state_embed_dim": cfg.scalar_state_embed_dim,
            "obs_embed_dim": cfg.obs_embed_dim,
            "lstm_hidden_dim": cfg.lstm_hidden_dim,
            "head_hidden_dim": cfg.head_hidden_dim,
            "num_atoms": cfg.num_atoms,
            "v_min": cfg.v_min,
            "v_max": cfg.v_max,
        },
        "input_dim": input_dim,
        "num_actions": N_ACTIONS,
        "switches": asdict(switches),
        "env_config": asdict(env.config),
        "train_config": asdict(cfg),
        "episode": episode,
        "global_step": global_step,
        "eval_metrics": metrics,
        "holdout_eval_metrics": holdout_metrics,
        "selection_key": list(selection_key(metrics, cfg)) if metrics else [],
        "composite_eval_score": composite_eval_score(metrics, cfg) if metrics else None,
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
    }


def write_train_config(save_dir: Path, cfg: TrainConfig, env: SensoryGridEnv, switches: ObservationSwitches, input_dim: int) -> None:
    with open(save_dir / "train_config.txt", "w", encoding="utf-8") as handle:
        handle.write("model_arch: patch_fusion_lstm_dueling_double_c51_recurrent_distributional_dqn\n")
        for key, value in asdict(cfg).items():
            handle.write(f"{key}: {value}\n")
        handle.write(f"input_dim: {input_dim}\n")
        handle.write(f"num_actions: {N_ACTIONS}\n")
        handle.write(f"actions: forward={ACTION_FORWARD}, left={ACTION_LEFT}, right={ACTION_RIGHT}\n")
        handle.write("\n[env_config]\n")
        for key, value in asdict(env.config).items():
            handle.write(f"{key}: {value}\n")
        handle.write("\n[switches]\n")
        for key, value in asdict(switches).items():
            handle.write(f"{key}: {value}\n")


def choose_action(
    net: RecurrentFusionLSTMC51DuelingQNetwork,
    obs: Dict[str, Any],
    state: Dict[str, np.ndarray],
    prev_action: np.ndarray,
    hidden: Tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    epsilon: float,
    cfg: TrainConfig,
) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
    state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    with torch.no_grad():
        _, q_values, new_hidden = net.forward_sequence(state_t, hidden)
        q_values_np = q_values[:, -1].squeeze(0).detach().cpu().numpy()
        if random.random() < epsilon:
            if cfg.use_structured_exploration:
                probs = structured_exploration_probs(obs, prev_action, q_values_np, cfg)
                action = int(np.random.choice(np.arange(N_ACTIONS), p=probs))
            else:
                action = random.randrange(N_ACTIONS)
        else:
            action = int(np.argmax(q_values_np))
    return action, tuple(item.detach() for item in new_hidden)


def episode_priority(last_info: Dict[str, object], diag_metrics: Dict[str, float]) -> float:
    coverage = finite_or(last_info.get("coverage", 0.0), 0.0)

    def diag_metric(name: str, fallback: float = 0.0) -> float:
        return finite_or(diag_metrics.get(name, fallback), fallback)

    cov_50 = diag_metric("coverage_at_step_50", coverage)
    cov_100 = diag_metric("coverage_at_step_100", coverage)
    cov_150 = diag_metric("coverage_at_step_150", coverage)
    new_tile_rate = diag_metric("new_tile_rate", 0.0)
    coverage_gain_25 = diag_metric("coverage_gain_25", 0.0)
    coverage_gain_per_energy = diag_metric("coverage_gain_per_energy", 0.0)
    new_tiles_per_energy = diag_metric("new_tiles_per_energy", 0.0)
    wall_hit_rate = diag_metric("wall_hit_rate", 0.0)
    hazard_contact_rate = diag_metric("hazard_contact_rate", 0.0)
    stagnation_step_rate = diag_metric("stagnation_step_rate", 0.0)
    priority = (
        1.0
        + 2.0 * coverage
        + 0.40 * cov_50
        + 0.60 * cov_100
        + 0.85 * cov_150
        + 0.90 * new_tile_rate
        + 0.80 * coverage_gain_25
        + 0.90 * coverage_gain_per_energy
        + 0.55 * new_tiles_per_energy
        - 0.35 * wall_hit_rate
        - 0.30 * hazard_contact_rate
        - 0.22 * stagnation_step_rate
    )
    return positive_finite_or(priority, 1e-3)


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "sensory_grid_env_v5",
        "model_family": "patch_fusion_lstm_dueling_double_c51_r2d2",
        "default_switches": asdict(build_default_switches()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recurrent distributional C51 Double DQN")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--save_dir", type=str, default="runs/recurrent_distributional_dqn")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_eval_episodes", type=int, default=10)
    parser.add_argument("--disable_structured_exploration", action="store_true")
    parser.add_argument("--enable_state_adaptive_exploration", action="store_true")
    parser.add_argument("--disable_episodic_exploration_bonus", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        eval_episodes=args.eval_episodes,
        holdout_eval_episodes=args.holdout_eval_episodes,
        use_structured_exploration=not args.disable_structured_exploration,
        use_state_adaptive_exploration=args.enable_state_adaptive_exploration,
        use_episodic_exploration_bonus=not args.disable_episodic_exploration_bonus,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)

    env, switches = build_training_env(cfg)
    eval_env, eval_switches = build_training_env(cfg)
    env.seed(cfg.seed)
    eval_env.seed(cfg.seed + 123)

    input_dim = env.observation_dim(switches)
    device = choose_device(cfg.device)
    online_net = build_network(env, switches, cfg).to(device)
    target_net = build_network(env, switches, cfg).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    replay = PrioritizedSequenceReplayBuffer(cfg.replay_capacity_episodes, cfg.replay_priority_alpha)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_train_config(save_dir, cfg, env, switches, input_dim)

    csv_path = save_dir / "training_log.csv"
    episode_metric_names = episode_diagnostic_metric_names()
    eval_fieldnames = eval_metric_names("eval")
    holdout_eval_fieldnames = eval_metric_names("holdout_eval")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "episode",
            "global_step",
            "epsilon",
            "reward",
            "coverage",
            "final_health",
            "length",
            *episode_metric_names,
            "mean_loss",
            "mean_ce_loss",
            "mean_td_error",
            *eval_fieldnames,
            *holdout_eval_fieldnames,
            "eval_soft_score",
        ])

    best_selection = (-float("inf"),) * 5
    best_eval_coverage = -float("inf")
    best_holdout_coverage = -float("inf")
    best_eval_soft = -float("inf")
    global_step = 0

    for episode in range(1, cfg.episodes + 1):
        obs, reset_info = env.reset(seed=cfg.seed + episode)
        state = obs_to_state(
            obs,
            env.config,
            switches,
            prev_action=np.zeros((N_ACTIONS,), dtype=np.float32),
            prev_reward=0.0,
            episode_step=0,
        )
        done = False
        ep_reward = 0.0
        hidden = None
        prev_action = np.zeros((N_ACTIONS,), dtype=np.float32)
        prev_reward = 0.0
        episode_step = 0
        current_coverage = float(reset_info.get("coverage", 0.0))
        novelty_counts: Dict[bytes, int] = {}
        losses: List[float] = []
        ce_losses: List[float] = []
        td_errors: List[float] = []
        transitions: List[Dict[str, object]] = []
        last_info = None
        episode_diag = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))

        while not done:
            obs_novelty = observation_novelty_bonus(obs, novelty_counts)
            episode_diag["obs_novelty_total"] = float(episode_diag["obs_novelty_total"]) + obs_novelty
            epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
            action, hidden = choose_action(online_net, obs, state, prev_action, hidden, device, epsilon, cfg)
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            done = terminated or truncated
            update_episode_diagnostics(episode_diag, action, info)

            next_coverage = float(info.get("coverage", current_coverage))
            coverage_delta = max(0.0, next_coverage - current_coverage)
            reward_terms = info.get("reward_terms", {})
            wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
            revisit = bool(action == ACTION_FORWARD and not wall_hit and coverage_delta <= 1e-9)
            energy_spent = transition_energy_spent(info)
            step_index = int(info.get("steps", episode_step + 1))
            no_progress = float(coverage_delta <= 1e-9)
            early_frac = max(0.0, 1.0 - (step_index - 1.0) / max(1.0, float(cfg.train_shaping_step_horizon)))
            coverage_bonus = float(cfg.train_coverage_delta_reward_scale) * coverage_delta
            coverage_bonus += float(cfg.train_early_progress_reward_scale) * coverage_delta * early_frac
            novelty_bonus = float(cfg.train_obs_novelty_bonus_scale) * obs_novelty * no_progress if cfg.use_episodic_exploration_bonus else 0.0
            energy_penalty = float(cfg.train_no_progress_energy_penalty_scale) * energy_spent * no_progress
            revisit_penalty = float(cfg.train_revisit_penalty_scale) * float(revisit)
            train_reward = float(reward + coverage_bonus + novelty_bonus - energy_penalty - revisit_penalty)
            stagnating = int(episode_diag.get("steps_since_new_tile", 0)) >= int(cfg.exploration_stagnation_trigger_steps)
            sample_weight = transition_sample_weight(
                coverage_delta=coverage_delta,
                obs_novelty=obs_novelty,
                energy_spent=energy_spent,
                revisit=revisit,
                wall_hit=wall_hit,
                stagnating=stagnating,
            )

            next_state = obs_to_state(
                next_obs,
                env.config,
                switches,
                prev_action=onehot_action(action),
                prev_reward=float(reward),
                episode_step=step_index,
            )
            transitions.append(
                {
                    "state": copy_state(state),
                    "next_state": copy_state(next_state),
                    "action": int(action),
                    "reward": float(reward),
                    "train_reward": float(train_reward),
                    "done": float(done),
                    "sample_weight": float(sample_weight),
                }
            )

            obs = next_obs
            state = next_state
            prev_action = onehot_action(action)
            prev_reward = float(reward)
            episode_step = step_index
            ep_reward += reward
            last_info = info
            current_coverage = next_coverage
            global_step += 1

        assert last_info is not None

        for idx, transition in enumerate(transitions):
            n_step_return = 0.0
            discount = 1.0
            steps = 0
            done_within_n = 0.0
            for offset in range(cfg.n_step_return):
                step_idx = idx + offset
                if step_idx >= len(transitions):
                    break
                step_transition = transitions[step_idx]
                n_step_return += discount * float(step_transition["train_reward"])
                steps += 1
                if float(step_transition["done"]) > 0.5:
                    done_within_n = 1.0
                    break
                discount *= cfg.gamma
            transition["n_step_return"] = float(n_step_return)
            transition["n_step_discount"] = float(discount)
            transition["n_step_done"] = float(done_within_n)
            transition["n_step_steps"] = int(max(1, steps))

        episode_diag_metrics = finalize_episode_diagnostics(episode_diag)
        replay.add_episode(transitions, priority=episode_priority(last_info, episode_diag_metrics))

        if len(replay) >= cfg.train_after_episodes:
            for _ in range(cfg.train_updates_per_episode):
                total_loss, ce_loss, mean_td = train_step(online_net, target_net, optimizer, replay, device, cfg)
                losses.append(total_loss)
                ce_losses.append(ce_loss)
                td_errors.append(mean_td)

        with torch.no_grad():
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.mul_(1.0 - cfg.target_soft_tau).add_(online_param.data, alpha=cfg.target_soft_tau)
        if episode % cfg.target_hard_sync_every_episodes == 0:
            target_net.load_state_dict(online_net.state_dict())

        do_eval = episode == 1 or episode % cfg.eval_every == 0 or episode == cfg.episodes
        if do_eval:
            eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                online_net,
                device,
                episodes=cfg.eval_episodes,
                seed_start=cfg.eval_seed_start,
                metric_prefix="eval",
            )
            holdout_eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                online_net,
                device,
                episodes=cfg.holdout_eval_episodes,
                seed_start=cfg.holdout_eval_seed_start,
                metric_prefix="holdout_eval",
            )
        else:
            eval_metrics = empty_eval_metrics("eval")
            holdout_eval_metrics = empty_eval_metrics("holdout_eval")

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_ce_loss = float(np.mean(ce_losses)) if ce_losses else 0.0
        mean_td_error = float(np.mean(td_errors)) if td_errors else 0.0
        epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
        eval_score = composite_eval_score(eval_metrics, cfg) if do_eval else float("nan")
        print(
            f"episode={episode:04d} step={global_step:06d} eps={epsilon:.3f} "
            f"reward={ep_reward:+.3f} coverage={last_info['coverage']:.3f} "
            f"health={last_info['health']} len={last_info['steps']} "
            f"new={episode_diag_metrics['new_tile_rate']:.3f} revisit={episode_diag_metrics['revisit_rate']:.3f} "
            f"wall={episode_diag_metrics['wall_hit_rate']:.3f} hazard={episode_diag_metrics['hazard_contact_rate']:.3f} "
            f"stagnation={episode_diag_metrics['stagnation_step_rate']:.3f} c25={episode_diag_metrics['coverage_gain_25']:.3f} "
            f"nov={episode_diag_metrics['obs_novelty_mean']:.3f} "
            f"loss={mean_loss:.4f} ce={mean_ce_loss:.4f} td={mean_td_error:.4f} "
            f"eval_cov={eval_metrics['eval_coverage_mean']:.3f} holdout_cov={holdout_eval_metrics['holdout_eval_coverage_mean']:.3f} "
            f"soft={eval_score:.3f}",
            flush=True,
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                episode,
                global_step,
                epsilon,
                ep_reward,
                last_info["coverage"],
                last_info["health"],
                last_info["steps"],
                *[episode_diag_metrics[name] for name in episode_metric_names],
                mean_loss,
                mean_ce_loss,
                mean_td_error,
                *[eval_metrics[name] for name in eval_fieldnames],
                *[holdout_eval_metrics[name] for name in holdout_eval_fieldnames],
                eval_score,
            ])

        if do_eval and cfg.save_eval_checkpoints:
            payload = checkpoint_payload(
                online_net,
                input_dim,
                switches,
                env,
                cfg,
                episode,
                global_step,
                eval_metrics,
                holdout_eval_metrics,
            )
            torch.save(payload, save_dir / f"ckpt_ep{episode:04d}.pt")
            current_key = selection_key(eval_metrics, cfg)
            if current_key > best_selection:
                best_selection = current_key
                torch.save(payload, save_dir / "best_model.pt")
                torch.save(payload, save_dir / "best_exploration_model.pt")
            if eval_metrics["eval_coverage_mean"] > best_eval_coverage:
                best_eval_coverage = eval_metrics["eval_coverage_mean"]
                torch.save(payload, save_dir / "best_coverage_model.pt")
            if holdout_eval_metrics["holdout_eval_coverage_mean"] > best_holdout_coverage:
                best_holdout_coverage = holdout_eval_metrics["holdout_eval_coverage_mean"]
                torch.save(payload, save_dir / "best_holdout_coverage_model.pt")
            if eval_score > best_eval_soft:
                best_eval_soft = eval_score
                torch.save(payload, save_dir / "best_soft_model.pt")

    torch.save(
        checkpoint_payload(online_net, input_dim, switches, env, cfg, cfg.episodes, global_step, {}, {}),
        save_dir / "final_model.pt",
    )
    print(f"Training finished. Logs and checkpoints saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()

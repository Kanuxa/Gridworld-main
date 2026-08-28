from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    EnvConfig,
    N_ACTIONS,
    ObservationSwitches,
    OBJ_FIRE,
    OBJ_GLASS,
    OBJ_ICE,
    SensoryGridEnv,
)
from models.shared.recurrent_dqn_interface import (
    ModelConfig,
    RecurrentPatchFusionDuelingAuxQNetwork,
    build_model_from_checkpoint,
    choose_device,
    init_runtime_context,
    obs_to_state,
    onehot_action,
    predict_action_for_gui,
    reset_runtime_context,
    update_runtime_context_after_env_step,
)

TRAINER_GUI_INTERFACE_VERSION = "windowed-coverage-dqn"
TRAINER_DISPLAY_NAME = "Windowed-coverage recurrent Double DQN"
EXPLORATION_COVERAGE_BUDGETS = (50, 100, 150)
HAZARD_OBJECT_IDS = {OBJ_FIRE, OBJ_ICE, OBJ_GLASS}
EXPLORATION_PROGRESS_BLOCK_STEPS = 25
EXPLORATION_STAGNATION_LOG_THRESHOLD = 6


@dataclass
class TrainConfig:
    episodes: int = 1200
    seed: int = 7
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_gradient_norm: float = 10.0
    save_dir: str = "runs/windowed_coverage_dqn"
    device: str = "auto"

    replay_capacity_episodes: int = 2000
    replay_priority_alpha: float = 0.75
    batch_size: int = 32
    burn_in: int = 10
    unroll_len: int = 24
    train_after_episodes: int = 20
    train_updates_per_episode: int = 10

    target_soft_tau: float = 0.05
    target_hard_sync_every_episodes: int = 100

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
    scalar_patch_embed_dim: int = 48
    scalar_state_embed_dim: int = 32
    obs_embed_dim: int = 256
    gru_hidden_dim: int = 256
    head_hidden_dim: int = 128

    aux_health_delta_loss_weight_start: float = 0.15
    aux_health_delta_loss_weight_end: float = 0.05
    aux_weight_decay_episodes: int = 800
    nonzero_health_delta_aux_boost: float = 3.0

    # Exploration-first checkpoint scoring:
    # prioritize overall coverage and fast coverage growth, then prefer cleaner
    # exploration. Health/survival only matter once coverage is effectively perfect.
    best_model_coverage_weight: float = 1.00
    best_model_coverage_50_weight: float = 0.18
    best_model_coverage_100_weight: float = 0.32
    best_model_coverage_150_weight: float = 0.50
    best_model_new_tile_weight: float = 0.22
    best_model_revisit_penalty_weight: float = 0.10
    best_model_wall_penalty_weight: float = 0.14
    best_model_hazard_penalty_weight: float = 0.10
    best_model_reward_efficiency_weight: float = 0.08
    best_model_reward_clip_abs: float = 5.0
    best_model_survival_weight: float = 0.02
    best_model_health_weight: float = 0.01
    best_model_perfect_exploration_threshold: float = 0.999

    # v5 change 2: slight survival pull, but still weaker than v4.1.
    training_survival_bonus: float = 0.60

    # v6 change 1: reward shaping improves credit assignment toward faster,
    # cleaner coverage growth without changing the environment.
    use_exploration_reward_shaping: bool = True
    train_coverage_delta_reward_scale: float = 6.0
    train_early_progress_reward_scale: float = 4.0
    train_shaping_step_horizon: int = 150
    train_no_progress_energy_penalty_scale: float = 0.025
    train_revisit_penalty_scale: float = 0.010

    # v6 change 2: replay focuses more on productive exploration windows than
    # on stagnant late-episode segments from otherwise good episodes.
    use_window_priority_sampling: bool = True
    replay_window_new_tile_bonus: float = 1.75
    replay_window_coverage_delta_scale: float = 140.0
    replay_window_revisit_penalty: float = 0.45
    replay_window_wall_penalty: float = 0.60
    replay_window_stagnation_penalty: float = 0.35
    replay_window_energy_penalty: float = 0.20

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


class EpisodeSequenceReplayBuffer:
    def __init__(self, capacity_episodes: int, priority_alpha: float = 0.75):
        self.episodes: Deque[List[Dict[str, object]]] = deque(maxlen=capacity_episodes)
        self.priorities: Deque[float] = deque(maxlen=capacity_episodes)
        self.priority_alpha = float(priority_alpha)

    def add_episode(self, transitions: List[Dict[str, object]], priority: float) -> None:
        if transitions:
            self.episodes.append(transitions)
            self.priorities.append(float(max(priority, 1e-3)))

    def __len__(self) -> int:
        return len(self.episodes)

    def _sample_episode(self) -> List[Dict[str, object]]:
        weights = [float(p) ** self.priority_alpha for p in self.priorities]
        idx = random.choices(range(len(self.episodes)), weights=weights, k=1)[0]
        return self.episodes[idx]

    def _sample_start_index(self, ep: List[Dict[str, object]], total_len: int) -> int:
        if len(ep) <= 1:
            return 0
        prefix = [0.0]
        for transition in ep:
            prefix.append(prefix[-1] + float(transition.get("sample_weight", 1.0)))
        weights = []
        for start in range(len(ep)):
            end = min(len(ep), start + total_len)
            window_weight = prefix[end] - prefix[start]
            weights.append(max(window_weight, 1e-3))
        return random.choices(range(len(ep)), weights=weights, k=1)[0]

    def sample(self, batch_size: int, total_len: int) -> Dict[str, object]:
        batch_states: List[List[Dict[str, np.ndarray]]] = []
        batch_next_states: List[List[Dict[str, np.ndarray]]] = []
        batch_prev_actions: List[np.ndarray] = []
        batch_prev_rewards: List[np.ndarray] = []
        batch_actions: List[np.ndarray] = []
        batch_rewards: List[np.ndarray] = []
        batch_dones: List[np.ndarray] = []
        batch_next_prev_actions: List[np.ndarray] = []
        batch_next_prev_rewards: List[np.ndarray] = []
        batch_valid: List[np.ndarray] = []
        batch_health_delta: List[np.ndarray] = []
        batch_coverage_delta: List[np.ndarray] = []
        batch_energy_spent: List[np.ndarray] = []
        batch_revisit: List[np.ndarray] = []
        batch_step_index: List[np.ndarray] = []

        for _ in range(batch_size):
            ep = self._sample_episode()
            start = self._sample_start_index(ep, total_len)
            last_next_state = copy_state(ep[-1]["next_state"])

            seq_states = []
            seq_next_states = []
            seq_prev_actions = np.zeros((total_len, N_ACTIONS), dtype=np.float32)
            seq_prev_rewards = np.zeros((total_len, 1), dtype=np.float32)
            seq_actions = np.zeros((total_len,), dtype=np.int64)
            seq_rewards = np.zeros((total_len,), dtype=np.float32)
            seq_dones = np.ones((total_len,), dtype=np.float32)
            seq_next_prev_actions = np.zeros((total_len, N_ACTIONS), dtype=np.float32)
            seq_next_prev_rewards = np.zeros((total_len, 1), dtype=np.float32)
            seq_valid = np.zeros((total_len,), dtype=np.float32)
            seq_health_delta = np.zeros((total_len,), dtype=np.float32)
            seq_coverage_delta = np.zeros((total_len,), dtype=np.float32)
            seq_energy_spent = np.zeros((total_len,), dtype=np.float32)
            seq_revisit = np.zeros((total_len,), dtype=np.float32)
            seq_step_index = np.zeros((total_len,), dtype=np.float32)

            for j in range(total_len):
                idx = start + j
                if idx < len(ep):
                    tr = ep[idx]
                    seq_states.append(copy_state(tr["state"]))
                    seq_next_states.append(copy_state(tr["next_state"]))
                    seq_prev_actions[j] = np.asarray(tr["prev_action"], dtype=np.float32)
                    seq_prev_rewards[j, 0] = float(tr["prev_reward"])
                    seq_actions[j] = int(tr["action"])
                    seq_rewards[j] = float(tr["reward"])
                    seq_dones[j] = float(tr["done"])
                    seq_next_prev_actions[j] = np.asarray(tr["next_prev_action"], dtype=np.float32)
                    seq_next_prev_rewards[j, 0] = float(tr["next_prev_reward"])
                    seq_valid[j] = 1.0
                    seq_health_delta[j] = float(tr.get("health_delta", 0.0))
                    seq_coverage_delta[j] = float(tr.get("coverage_delta", 0.0))
                    seq_energy_spent[j] = float(tr.get("energy_spent", 0.0))
                    seq_revisit[j] = float(tr.get("revisit", 0.0))
                    seq_step_index[j] = float(tr.get("step_index", 0.0))
                else:
                    seq_states.append(copy_state(last_next_state))
                    seq_next_states.append(copy_state(last_next_state))

            batch_states.append(seq_states)
            batch_next_states.append(seq_next_states)
            batch_prev_actions.append(seq_prev_actions)
            batch_prev_rewards.append(seq_prev_rewards)
            batch_actions.append(seq_actions)
            batch_rewards.append(seq_rewards)
            batch_dones.append(seq_dones)
            batch_next_prev_actions.append(seq_next_prev_actions)
            batch_next_prev_rewards.append(seq_next_prev_rewards)
            batch_valid.append(seq_valid)
            batch_health_delta.append(seq_health_delta)
            batch_coverage_delta.append(seq_coverage_delta)
            batch_energy_spent.append(seq_energy_spent)
            batch_revisit.append(seq_revisit)
            batch_step_index.append(seq_step_index)

        def collate_state_sequences(state_sequences: List[List[Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
            keys = state_sequences[0][0].keys()
            out: Dict[str, np.ndarray] = {}
            for k in keys:
                out[k] = np.stack([
                    np.stack([step[k] for step in seq]).astype(np.float32)
                    for seq in state_sequences
                ]).astype(np.float32)
            return out

        return {
            "states": collate_state_sequences(batch_states),
            "next_states": collate_state_sequences(batch_next_states),
            "prev_actions": np.stack(batch_prev_actions).astype(np.float32),
            "prev_rewards": np.stack(batch_prev_rewards).astype(np.float32),
            "actions": np.stack(batch_actions).astype(np.int64),
            "rewards": np.stack(batch_rewards).astype(np.float32),
            "dones": np.stack(batch_dones).astype(np.float32),
            "next_prev_actions": np.stack(batch_next_prev_actions).astype(np.float32),
            "next_prev_rewards": np.stack(batch_next_prev_rewards).astype(np.float32),
            "valid": np.stack(batch_valid).astype(np.float32),
            "health_delta": np.stack(batch_health_delta).astype(np.float32),
            "coverage_delta": np.stack(batch_coverage_delta).astype(np.float32),
            "energy_spent": np.stack(batch_energy_spent).astype(np.float32),
            "revisit": np.stack(batch_revisit).astype(np.float32),
            "step_index": np.stack(batch_step_index).astype(np.float32),
        }


def get_model_config(cfg: TrainConfig) -> ModelConfig:
    return ModelConfig(
        conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2,
        vision_embed_dim=cfg.vision_embed_dim,
        scalar_patch_embed_dim=cfg.scalar_patch_embed_dim,
        scalar_state_embed_dim=cfg.scalar_state_embed_dim,
        obs_embed_dim=cfg.obs_embed_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
    )


def copy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in state.items()}


def state_seq_batch_to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def transition_energy_spent(info: Dict[str, object]) -> float:
    return float(
        float(info.get("time_base_cost", 0.0))
        + float(info.get("forward_extra_cost", 0.0))
        + float(info.get("turn_extra_cost", 0.0))
        + float(info.get("thermal_extra_this_tick", 0.0))
    )


def transition_sample_weight(
    coverage_delta: float,
    energy_spent: float,
    revisit: bool,
    wall_hit: bool,
    stagnating: bool,
    cfg: TrainConfig,
) -> float:
    weight = 1.0
    weight += float(cfg.replay_window_coverage_delta_scale) * max(0.0, float(coverage_delta))
    if float(coverage_delta) > 1e-9:
        weight += float(cfg.replay_window_new_tile_bonus)
    weight -= float(cfg.replay_window_revisit_penalty) * float(revisit)
    weight -= float(cfg.replay_window_wall_penalty) * float(wall_hit)
    weight -= float(cfg.replay_window_stagnation_penalty) * float(stagnating)
    weight -= float(cfg.replay_window_energy_penalty) * max(0.0, float(energy_spent))
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
    trigger_frac = (trigger - cfg.epsilon_start) / denom
    trigger_frac = float(np.clip(trigger_frac, 0.0, 1.0))
    trigger_step = int(round(trigger_frac * cfg.epsilon_decay_steps))
    cycle_steps = max(1, int(cfg.epsilon_pulse_cycle_steps))
    elapsed = max(0, int(step) - trigger_step)
    phase = (elapsed % cycle_steps) / cycle_steps
    triangle = 1.0 - abs(2.0 * phase - 1.0)
    cycle_index = elapsed / cycle_steps
    decay_cycles = max(float(cfg.epsilon_pulse_decay_cycles), 1e-6)
    decay_multiplier = float(np.exp(-cycle_index / decay_cycles))
    pulse = float(cfg.epsilon_pulse_amplitude) * triangle * decay_multiplier
    return float(min(cfg.epsilon_start, base + pulse))


def adaptive_exploration_epsilon(
    step: int,
    obs: Dict[str, Any],
    episode_diag: Dict[str, object],
    cfg: TrainConfig,
) -> float:
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
    if isinstance(recent_coverages, deque) and len(recent_coverages) >= 2:
        history = list(float(x) for x in recent_coverages)
        window = min(max(1, int(cfg.exploration_low_progress_window)), len(history) - 1)
        recent_gain = max(0.0, history[-1] - history[-1 - window])
        threshold = max(float(cfg.exploration_low_progress_threshold), 1e-6)
        if recent_gain < threshold:
            gain_frac = 1.0 - (recent_gain / threshold)
            bonus += float(cfg.exploration_low_progress_epsilon_bonus) * float(np.clip(gain_frac, 0.0, 1.0))

    turn_streak = int(obs.get("consecutive_turn_steps", 0))
    if turn_streak >= 2:
        turn_frac = np.clip((turn_streak - 1) / 2.0, 0.0, 1.0)
        bonus += float(cfg.exploration_turn_stagnation_epsilon_bonus) * float(turn_frac)

    no_move_steps = int(obs.get("consecutive_no_move_steps", 0))
    if no_move_steps >= 2:
        no_move_frac = np.clip((no_move_steps - 1) / 2.0, 0.0, 1.0)
        bonus += float(cfg.exploration_no_move_epsilon_bonus) * float(no_move_frac)

    return float(min(cfg.epsilon_start, base + (1.0 - base) * bonus))


def episode_diagnostic_metric_names() -> List[str]:
    metrics = [
        "new_tile_rate",
        "revisit_rate",
        "wall_hit_rate",
        "turn_streak_rate",
        "hazard_contact_rate",
        "action_forward_rate",
        "action_left_rate",
        "action_right_rate",
        "max_no_new_tile_streak",
        "stagnation_step_rate",
        "coverage_gain_25",
        "energy_spent",
        "coverage_gain_per_energy",
        "new_tiles_per_energy",
        "turn_oscillation_rate",
        "forward_after_turn_success_rate",
    ]
    metrics.extend([f"coverage_at_step_{budget}" for budget in EXPLORATION_COVERAGE_BUDGETS])
    return metrics


def eval_metric_names(prefix: str) -> List[str]:
    base_metrics = [
        f"{prefix}_reward_mean",
        f"{prefix}_coverage_mean",
        f"{prefix}_final_health_mean",
        f"{prefix}_length_mean",
        f"{prefix}_survival_rate",
        f"{prefix}_death_rate",
    ]
    diag_metrics = [f"{prefix}_{name}_mean" for name in episode_diagnostic_metric_names()]
    return base_metrics + diag_metrics


def empty_eval_metrics(prefix: str) -> Dict[str, float]:
    return {name: float("nan") for name in eval_metric_names(prefix)}


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def init_episode_diagnostics(initial_coverage: float = 0.0) -> Dict[str, object]:
    return {
        "initial_coverage": float(initial_coverage),
        "steps": 0,
        "new_tile_steps": 0,
        "revisit_steps": 0,
        "wall_hits": 0,
        "turn_streak_steps": 0,
        "hazard_contacts": 0,
        "energy_spent": 0.0,
        "action_counts": {
            ACTION_FORWARD: 0,
            ACTION_LEFT: 0,
            ACTION_RIGHT: 0,
        },
        "coverage_by_budget": {
            budget: float("nan") for budget in EXPLORATION_COVERAGE_BUDGETS
        },
        "steps_since_new_tile": 0,
        "max_no_new_tile_streak": 0,
        "stagnation_steps": 0,
        "recent_actions": deque(maxlen=3),
        "turn_oscillation_events": 0,
        "forward_after_turn_attempts": 0,
        "forward_after_turn_successes": 0,
        "block_start_step": 0,
        "block_start_coverage": float(initial_coverage),
        "coverage_gain_blocks": [],
        "recent_coverages": deque([float(initial_coverage)], maxlen=256),
    }


def update_episode_diagnostics(diag: Dict[str, object], action: int, info: Dict[str, object]) -> None:
    diag["steps"] = int(diag["steps"]) + 1
    action_counts = diag["action_counts"]
    assert isinstance(action_counts, dict)
    action_counts[action] = int(action_counts.get(action, 0)) + 1

    reward_terms = info.get("reward_terms", {})
    wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
    new_tile = float(reward_terms.get("explore_reward", 0.0)) > 0.0
    revisit = action == ACTION_FORWARD and not wall_hit and not new_tile
    turn_streak = float(reward_terms.get("turn_streak_penalty", 0.0)) < 0.0
    hazard_contact = int(info.get("contacted", -1)) in HAZARD_OBJECT_IDS

    diag["new_tile_steps"] = int(diag["new_tile_steps"]) + int(new_tile)
    diag["revisit_steps"] = int(diag["revisit_steps"]) + int(revisit)
    diag["wall_hits"] = int(diag["wall_hits"]) + int(wall_hit)
    diag["turn_streak_steps"] = int(diag["turn_streak_steps"]) + int(turn_streak)
    diag["hazard_contacts"] = int(diag["hazard_contacts"]) + int(hazard_contact)
    diag["energy_spent"] = float(diag["energy_spent"]) + transition_energy_spent(info)

    if new_tile:
        diag["steps_since_new_tile"] = 0
    else:
        diag["steps_since_new_tile"] = int(diag["steps_since_new_tile"]) + 1
    diag["max_no_new_tile_streak"] = max(int(diag["max_no_new_tile_streak"]), int(diag["steps_since_new_tile"]))
    if int(diag["steps_since_new_tile"]) >= EXPLORATION_STAGNATION_LOG_THRESHOLD:
        diag["stagnation_steps"] = int(diag["stagnation_steps"]) + 1

    recent_actions = diag["recent_actions"]
    assert isinstance(recent_actions, deque)
    if len(recent_actions) > 0 and recent_actions[-1] in (ACTION_LEFT, ACTION_RIGHT) and action == ACTION_FORWARD:
        diag["forward_after_turn_attempts"] = int(diag["forward_after_turn_attempts"]) + 1
        diag["forward_after_turn_successes"] = int(diag["forward_after_turn_successes"]) + int(new_tile)
    recent_actions.append(action)
    if len(recent_actions) == 3:
        pattern = tuple(int(x) for x in recent_actions)
        if pattern in (
            (ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT),
            (ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT),
        ):
            diag["turn_oscillation_events"] = int(diag["turn_oscillation_events"]) + 1

    step_index = int(info.get("steps", diag["steps"]))
    coverage = float(info.get("coverage", float("nan")))
    coverage_by_budget = diag["coverage_by_budget"]
    assert isinstance(coverage_by_budget, dict)
    for budget in EXPLORATION_COVERAGE_BUDGETS:
        if step_index >= budget and np.isnan(float(coverage_by_budget[budget])):
            coverage_by_budget[budget] = coverage

    recent_coverages = diag["recent_coverages"]
    assert isinstance(recent_coverages, deque)
    if np.isfinite(coverage):
        recent_coverages.append(float(coverage))

    block_start_step = int(diag["block_start_step"])
    block_start_coverage = float(diag["block_start_coverage"])
    coverage_gain_blocks = diag["coverage_gain_blocks"]
    assert isinstance(coverage_gain_blocks, list)
    if np.isfinite(coverage) and step_index - block_start_step >= EXPLORATION_PROGRESS_BLOCK_STEPS:
        if np.isfinite(block_start_coverage):
            coverage_gain_blocks.append(float(max(0.0, coverage - block_start_coverage)))
        diag["block_start_step"] = step_index
        diag["block_start_coverage"] = float(coverage)


def finalize_episode_diagnostics(diag: Dict[str, object]) -> Dict[str, float]:
    steps = max(1, int(diag["steps"]))
    action_counts = diag["action_counts"]
    coverage_by_budget = diag["coverage_by_budget"]
    assert isinstance(action_counts, dict)
    assert isinstance(coverage_by_budget, dict)
    recent_coverages = diag["recent_coverages"]
    coverage_gain_blocks = list(diag["coverage_gain_blocks"])
    assert isinstance(recent_coverages, deque)

    if len(recent_coverages) > 0:
        final_coverage = float(recent_coverages[-1])
        block_start_coverage = float(diag["block_start_coverage"])
        block_start_step = int(diag["block_start_step"])
        tail_steps = max(0, steps - block_start_step)
        if (
            tail_steps > 0
            and np.isfinite(final_coverage)
            and np.isfinite(block_start_coverage)
        ):
            tail_gain = max(0.0, final_coverage - block_start_coverage)
            scaled_tail_gain = tail_gain * (EXPLORATION_PROGRESS_BLOCK_STEPS / max(1, tail_steps))
            coverage_gain_blocks.append(float(scaled_tail_gain))
    else:
        final_coverage = float(diag["initial_coverage"])

    total_energy_spent = float(diag["energy_spent"])
    effective_energy = max(total_energy_spent, 1e-6)
    coverage_gain_total = max(0.0, final_coverage - float(diag["initial_coverage"]))

    metrics = {
        "new_tile_rate": float(diag["new_tile_steps"]) / steps,
        "revisit_rate": float(diag["revisit_steps"]) / steps,
        "wall_hit_rate": float(diag["wall_hits"]) / steps,
        "turn_streak_rate": float(diag["turn_streak_steps"]) / steps,
        "hazard_contact_rate": float(diag["hazard_contacts"]) / steps,
        "action_forward_rate": float(action_counts.get(ACTION_FORWARD, 0)) / steps,
        "action_left_rate": float(action_counts.get(ACTION_LEFT, 0)) / steps,
        "action_right_rate": float(action_counts.get(ACTION_RIGHT, 0)) / steps,
        "max_no_new_tile_streak": float(diag["max_no_new_tile_streak"]),
        "stagnation_step_rate": float(diag["stagnation_steps"]) / steps,
        "coverage_gain_25": nanmean(coverage_gain_blocks),
        "energy_spent": total_energy_spent,
        "coverage_gain_per_energy": coverage_gain_total / effective_energy,
        "new_tiles_per_energy": float(diag["new_tile_steps"]) / effective_energy,
        "turn_oscillation_rate": float(diag["turn_oscillation_events"]) / steps,
        "forward_after_turn_success_rate": (
            float(diag["forward_after_turn_successes"]) / max(1, int(diag["forward_after_turn_attempts"]))
        ),
    }
    for budget in EXPLORATION_COVERAGE_BUDGETS:
        metrics[f"coverage_at_step_{budget}"] = float(coverage_by_budget[budget])
    return metrics


def current_aux_weight(episode: int, cfg: TrainConfig) -> float:
    if episode >= cfg.aux_weight_decay_episodes:
        return cfg.aux_health_delta_loss_weight_end
    frac = episode / max(1, cfg.aux_weight_decay_episodes)
    return cfg.aux_health_delta_loss_weight_start + frac * (
        cfg.aux_health_delta_loss_weight_end - cfg.aux_health_delta_loss_weight_start
    )


def extract_front_cell_features(obs: Dict[str, Any]) -> Dict[str, object]:
    vision_patch = np.asarray(obs.get("vision"), dtype=np.int64)
    centre = vision_patch.shape[0] // 2
    front_r = max(centre - 1, 0)
    front_c = centre
    front_obj = int(vision_patch[front_r, front_c])

    visited_patch = np.asarray(
        obs.get("visited_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )
    hazard_patch = np.asarray(
        obs.get("hazard_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )

    front_visited = float(visited_patch[front_r, front_c]) > 0.5
    front_known_hazard = float(hazard_patch[front_r, front_c]) >= 0.25
    return {
        "front_obj": front_obj,
        "front_unvisited": not front_visited,
        "front_visible_hazard": front_obj in HAZARD_OBJECT_IDS,
        "front_known_hazard": front_known_hazard,
    }


def directional_novelty_scores(obs: Dict[str, Any], cfg: TrainConfig) -> np.ndarray:
    vision_patch = np.asarray(obs.get("vision"), dtype=np.int64)
    visited_patch = np.asarray(
        obs.get("visited_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )
    hazard_patch = np.asarray(
        obs.get("hazard_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )
    centre = vision_patch.shape[0] // 2

    def cell_score(r: int, c: int) -> float:
        if not (0 <= r < vision_patch.shape[0] and 0 <= c < vision_patch.shape[1]):
            return 0.0
        obj = int(vision_patch[r, c])
        visited = float(visited_patch[r, c])
        hazard_mem = float(hazard_patch[r, c])
        score = (
            cfg.exploration_direction_unvisited_weight
            if visited < 0.5
            else -cfg.exploration_direction_visited_penalty
        )
        if obj in HAZARD_OBJECT_IDS:
            score -= cfg.exploration_direction_visible_hazard_penalty
        score -= cfg.exploration_direction_hazard_memory_penalty * hazard_mem
        return float(score)

    direction_cells = {
        ACTION_FORWARD: [
            ((centre - 1, centre), 1.00),
            ((centre - 1, centre - 1), 0.55),
            ((centre - 1, centre + 1), 0.55),
            ((centre - 2, centre), 0.80),
        ],
        ACTION_LEFT: [
            ((centre, centre - 1), 1.00),
            ((centre - 1, centre - 1), 0.70),
            ((centre + 1, centre - 1), 0.45),
            ((centre - 1, centre - 2), 0.35),
        ],
        ACTION_RIGHT: [
            ((centre, centre + 1), 1.00),
            ((centre - 1, centre + 1), 0.70),
            ((centre + 1, centre + 1), 0.45),
            ((centre - 1, centre + 2), 0.35),
        ],
    }

    scores = np.zeros((N_ACTIONS,), dtype=np.float64)
    for action, items in direction_cells.items():
        for (r, c), weight in items:
            scores[action] += weight * cell_score(r, c)
    frontier_specs = {
        ACTION_FORWARD: ((-1, 0), (0, 1)),
        ACTION_LEFT: ((0, -1), (-1, 0)),
        ACTION_RIGHT: ((0, 1), (-1, 0)),
    }
    max_depth = max(1, min(centre, 3))
    for action, (main_vec, side_vec) in frontier_specs.items():
        frontier_score = 0.0
        for depth in range(1, max_depth + 1):
            depth_weight = float(cfg.exploration_frontier_depth_decay) ** (depth - 1)
            base_r = centre + main_vec[0] * depth
            base_c = centre + main_vec[1] * depth
            for lateral in (0, -1, 1):
                r = base_r + side_vec[0] * lateral
                c = base_c + side_vec[1] * lateral
                if not (0 <= r < vision_patch.shape[0] and 0 <= c < vision_patch.shape[1]):
                    continue
                lateral_weight = 1.0 if lateral == 0 else float(cfg.exploration_frontier_side_weight)
                obj = int(vision_patch[r, c])
                visited = float(visited_patch[r, c])
                hazard_mem = float(hazard_patch[r, c])
                cell_frontier_score = (
                    float(cfg.exploration_frontier_unvisited_weight)
                    if visited < 0.5
                    else -float(cfg.exploration_frontier_visited_penalty)
                )
                if visited < 0.5:
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < visited_patch.shape[0] and 0 <= nc < visited_patch.shape[1]:
                            if float(visited_patch[nr, nc]) > 0.5:
                                cell_frontier_score += float(cfg.exploration_frontier_frontier_bonus)
                                break
                if obj in HAZARD_OBJECT_IDS:
                    cell_frontier_score -= float(cfg.exploration_frontier_visible_hazard_penalty)
                cell_frontier_score -= float(cfg.exploration_frontier_known_hazard_penalty) * hazard_mem
                frontier_score += depth_weight * lateral_weight * cell_frontier_score
        if action in (ACTION_LEFT, ACTION_RIGHT):
            frontier_score -= float(cfg.exploration_frontier_turn_penalty)
        scores[action] += float(cfg.exploration_frontier_score_scale) * frontier_score
    return scores


def structured_exploration_probs(
    obs: Dict[str, Any],
    prev_action: np.ndarray,
    q_values: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
    temperature = max(float(cfg.exploration_softmax_temperature), 1e-3)
    logits = np.asarray(q_values, dtype=np.float64) / temperature
    logits = logits - np.max(logits)
    base_probs = np.exp(np.clip(logits, -60.0, 60.0))
    if not np.isfinite(base_probs).all() or base_probs.sum() <= 0.0:
        base_probs = np.ones((N_ACTIONS,), dtype=np.float64)

    weights = np.ones((N_ACTIONS,), dtype=np.float64)
    features = extract_front_cell_features(obs)
    directional_scores = directional_novelty_scores(obs, cfg)
    directional_scores = directional_scores - np.max(directional_scores)
    weights *= np.exp(cfg.exploration_direction_score_scale * directional_scores)

    if bool(features["front_unvisited"]):
        weights[ACTION_FORWARD] *= cfg.exploration_forward_unvisited_bonus
    else:
        weights[ACTION_FORWARD] *= cfg.exploration_forward_revisit_penalty

    if bool(features["front_visible_hazard"]):
        weights[ACTION_FORWARD] *= cfg.exploration_visible_hazard_penalty
    elif bool(features["front_known_hazard"]):
        weights[ACTION_FORWARD] *= cfg.exploration_known_hazard_penalty

    prev_action_arr = np.asarray(prev_action, dtype=np.float32)
    prev_action_idx = int(np.argmax(prev_action_arr)) if float(prev_action_arr.sum()) > 0.0 else None

    if prev_action_idx in (ACTION_LEFT, ACTION_RIGHT):
        weights[ACTION_LEFT] *= cfg.exploration_repeat_turn_penalty
        weights[ACTION_RIGHT] *= cfg.exploration_repeat_turn_penalty
        weights[ACTION_FORWARD] *= cfg.exploration_post_turn_forward_bonus

    no_move_steps = int(obs.get("consecutive_no_move_steps", 0))
    turn_streak_steps = int(obs.get("consecutive_turn_steps", 0))
    if prev_action_idx == ACTION_FORWARD and no_move_steps > 0 and turn_streak_steps == 0:
        weights[ACTION_FORWARD] *= cfg.exploration_post_bump_forward_penalty
        weights[ACTION_LEFT] *= 1.20
        weights[ACTION_RIGHT] *= 1.20

    if turn_streak_steps >= 2:
        weights[ACTION_LEFT] *= cfg.exploration_repeat_turn_penalty
        weights[ACTION_RIGHT] *= cfg.exploration_repeat_turn_penalty
        weights[ACTION_FORWARD] *= cfg.exploration_post_turn_forward_bonus

    probs = base_probs * weights
    if not np.isfinite(probs).all() or probs.sum() <= 0.0:
        return np.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=np.float64)
    return probs / probs.sum()


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)


def build_default_switches() -> ObservationSwitches:
    return ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )


def build_training_env(cfg: TrainConfig | None = None) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    train_cfg = cfg or TrainConfig()
    env_cfg = EnvConfig(
        grid_size=15,
        patch_size=5,
        init_health=10,
        max_health=10,
        init_energy=10.0,
        max_energy=10.0,
        max_steps=250,
        n_fire=2,
        n_ice=1,
        n_meat=3,
        n_flower=2,
        n_glass=2,
        glass_damage=2,
        meat_heal=2,
        fire_damage=3,
        ice_damage=3,
        ambient_temperature_c=22.0,
        comfort_low_c=18.0,
        comfort_high_c=24.0,
        time_energy_cost=0.20,
        turn_energy_cost=0.10,
        forward_energy_cost=0.60,
        thermal_extra_energy_max=0.50,
        energy_reward_scale=0.025,
        discomfort_reward_scale=0.04,
        fire_temp_delta_amp=13.0,
        ice_temp_delta_amp=13.0,
        temp_sigma=2.1,
        meat_smell_amp=0.75,
        flower_smell_amp=1.00,
        smell_sigma_meat=2.2,
        smell_sigma_flower=2.8,
        no_move_penalty=0.01,
        turn_streak_penalty=0.005,
        survival_bonus=train_cfg.training_survival_bonus,
    )
    switches = build_default_switches()
    return SensoryGridEnv(env_cfg), switches


def build_network(env: SensoryGridEnv, switches: ObservationSwitches, cfg: TrainConfig) -> RecurrentPatchFusionDuelingAuxQNetwork:
    return RecurrentPatchFusionDuelingAuxQNetwork(
        patch_size=env.config.patch_size,
        num_actions=N_ACTIONS,
        cfg=get_model_config(cfg),
        use_vision=switches.include_vision,
        use_temperature_patch=switches.include_temperature_patch,
        use_smell_patch=switches.include_smell_patch,
        use_visited_memory=switches.include_visited_memory,
        use_hazard_memory=switches.include_hazard_memory,
        scalar_state_dim=6,
    )


def train_step(
    online_net: RecurrentPatchFusionDuelingAuxQNetwork,
    target_net: RecurrentPatchFusionDuelingAuxQNetwork,
    optimizer: optim.Optimizer,
    buffer: EpisodeSequenceReplayBuffer,
    device: torch.device,
    cfg: TrainConfig,
    aux_weight: float,
) -> Tuple[float, float, float]:
    total_len = cfg.burn_in + cfg.unroll_len
    batch = buffer.sample(cfg.batch_size, total_len)
    states_t = state_seq_batch_to_torch(batch["states"], device)
    next_states_t = state_seq_batch_to_torch(batch["next_states"], device)
    prev_actions_t = torch.from_numpy(batch["prev_actions"]).to(device)
    prev_rewards_t = torch.from_numpy(batch["prev_rewards"]).to(device)
    actions_t = torch.from_numpy(batch["actions"]).to(device)
    rewards_t = torch.from_numpy(batch["rewards"]).to(device)
    dones_t = torch.from_numpy(batch["dones"]).to(device)
    next_prev_actions_t = torch.from_numpy(batch["next_prev_actions"]).to(device)
    next_prev_rewards_t = torch.from_numpy(batch["next_prev_rewards"]).to(device)
    valid_t = torch.from_numpy(batch["valid"]).to(device)
    coverage_delta_t = torch.from_numpy(batch["coverage_delta"]).to(device)
    energy_spent_t = torch.from_numpy(batch["energy_spent"]).to(device)
    revisit_t = torch.from_numpy(batch["revisit"]).to(device)
    step_index_t = torch.from_numpy(batch["step_index"]).to(device)

    if cfg.burn_in > 0:
        with torch.no_grad():
            _, h_online, _ = online_net.forward_sequence(
                {k: v[:, :cfg.burn_in] for k, v in states_t.items()},
                prev_actions_t[:, :cfg.burn_in],
                prev_rewards_t[:, :cfg.burn_in],
                None,
            )
            _, h_target, _ = target_net.forward_sequence(
                {k: v[:, :cfg.burn_in] for k, v in states_t.items()},
                prev_actions_t[:, :cfg.burn_in],
                prev_rewards_t[:, :cfg.burn_in],
                None,
            )
        h_online = h_online.detach()
        h_target = h_target.detach()
    else:
        h_online = None
        h_target = None

    states_main = {k: v[:, cfg.burn_in:] for k, v in states_t.items()}
    next_states_main = {k: v[:, cfg.burn_in:] for k, v in next_states_t.items()}
    prev_actions_main = prev_actions_t[:, cfg.burn_in:]
    prev_rewards_main = prev_rewards_t[:, cfg.burn_in:]
    actions_main = actions_t[:, cfg.burn_in:]
    rewards_main = rewards_t[:, cfg.burn_in:]
    dones_main = dones_t[:, cfg.burn_in:]
    next_prev_actions_main = next_prev_actions_t[:, cfg.burn_in:]
    next_prev_rewards_main = next_prev_rewards_t[:, cfg.burn_in:]
    valid_main = valid_t[:, cfg.burn_in:]
    coverage_delta_main = coverage_delta_t[:, cfg.burn_in:]
    energy_spent_main = energy_spent_t[:, cfg.burn_in:]
    revisit_main = revisit_t[:, cfg.burn_in:]
    step_index_main = step_index_t[:, cfg.burn_in:]

    q_seq, _, health_delta_seq = online_net.forward_sequence(states_main, prev_actions_main, prev_rewards_main, h_online)
    chosen_q = q_seq.gather(-1, actions_main.unsqueeze(-1)).squeeze(-1)

    shaped_rewards_main = rewards_main
    if cfg.use_exploration_reward_shaping:
        coverage_bonus = float(cfg.train_coverage_delta_reward_scale) * coverage_delta_main
        horizon = max(1.0, float(cfg.train_shaping_step_horizon))
        early_frac = torch.clamp(1.0 - (step_index_main - 1.0) / horizon, 0.0, 1.0)
        coverage_bonus = coverage_bonus + float(cfg.train_early_progress_reward_scale) * coverage_delta_main * early_frac
        no_progress = (coverage_delta_main <= 1e-9).to(rewards_main.dtype)
        energy_penalty = float(cfg.train_no_progress_energy_penalty_scale) * energy_spent_main * no_progress
        revisit_penalty = float(cfg.train_revisit_penalty_scale) * revisit_main
        shaped_rewards_main = rewards_main + coverage_bonus - energy_penalty - revisit_penalty

    with torch.no_grad():
        next_online_q, _, _ = online_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_online)
        next_actions = torch.argmax(next_online_q, dim=-1, keepdim=True)
        next_target_q, _, _ = target_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_target)
        next_q = next_target_q.gather(-1, next_actions).squeeze(-1)
        targets = shaped_rewards_main + cfg.gamma * (1.0 - dones_main) * next_q

    mask = valid_main > 0.5
    if not bool(mask.any().item()):
        return 0.0, 0.0, 0.0

    q_loss = nn.functional.smooth_l1_loss(chosen_q[mask], targets[mask])

    health_delta_main = torch.from_numpy(batch["health_delta"]).to(device)[:, cfg.burn_in:]
    forward_mask = mask & (actions_main == ACTION_FORWARD)
    if bool(forward_mask.any().item()):
        pred = health_delta_seq.squeeze(-1)[forward_mask]
        target = health_delta_main[forward_mask]
        aux_per_item = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.ones_like(aux_per_item)
        weights = torch.where(target.abs() > 1e-6, weights * cfg.nonzero_health_delta_aux_boost, weights)
        aux_loss = (aux_per_item * weights).sum() / weights.sum().clamp_min(1.0)
    else:
        aux_loss = torch.tensor(0.0, device=device)

    total_loss = q_loss + aux_weight * aux_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_gradient_norm)
    optimizer.step()
    return float(total_loss.item()), float(q_loss.item()), float(aux_loss.item())


@torch.no_grad()
def evaluate_policy(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: RecurrentPatchFusionDuelingAuxQNetwork,
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
        state = obs_to_state(obs, env.config, switches)
        done = False
        ep_reward = 0.0
        hidden = None
        prev_action = np.zeros((1, 1, N_ACTIONS), dtype=np.float32)
        prev_reward = np.zeros((1, 1, 1), dtype=np.float32)
        last_info = None
        diag = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))

        while not done:
            state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
            prev_action_t = torch.from_numpy(prev_action).to(device)
            prev_reward_t = torch.from_numpy(prev_reward).to(device)
            q, hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
            action = int(torch.argmax(q[:, -1], dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            update_episode_diagnostics(diag, action, info)
            state = obs_to_state(next_obs, env.config, switches)
            prev_action[0, 0] = onehot_action(action)
            prev_reward[0, 0, 0] = float(reward)
            ep_reward += reward
            last_info = info
            done = terminated or truncated

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


def composite_eval_score(eval_metrics: Dict[str, float], cfg: TrainConfig) -> float:
    def metric(name: str, default: float = 0.0) -> float:
        value = float(eval_metrics.get(name, default))
        return default if np.isnan(value) else value

    coverage = metric("eval_coverage_mean")
    coverage_50 = metric("eval_coverage_at_step_50_mean")
    coverage_100 = metric("eval_coverage_at_step_100_mean")
    coverage_150 = metric("eval_coverage_at_step_150_mean")
    new_tile_rate = metric("eval_new_tile_rate_mean")
    revisit_rate = metric("eval_revisit_rate_mean")
    wall_hit_rate = metric("eval_wall_hit_rate_mean")
    hazard_contact_rate = metric("eval_hazard_contact_rate_mean")

    reward_clip = max(float(cfg.best_model_reward_clip_abs), 1e-6)
    reward_efficiency = float(np.clip(metric("eval_reward_mean") / reward_clip, -1.0, 1.0))

    perfect_exploration_gate = 1.0 if coverage >= cfg.best_model_perfect_exploration_threshold else 0.0
    return (
        cfg.best_model_coverage_weight * coverage
        + cfg.best_model_coverage_50_weight * coverage_50
        + cfg.best_model_coverage_100_weight * coverage_100
        + cfg.best_model_coverage_150_weight * coverage_150
        + cfg.best_model_new_tile_weight * new_tile_rate
        - cfg.best_model_revisit_penalty_weight * revisit_rate
        - cfg.best_model_wall_penalty_weight * wall_hit_rate
        - cfg.best_model_hazard_penalty_weight * hazard_contact_rate
        + cfg.best_model_reward_efficiency_weight * reward_efficiency
        + perfect_exploration_gate * cfg.best_model_survival_weight * metric("eval_survival_rate")
        + perfect_exploration_gate * cfg.best_model_health_weight * max(metric("eval_final_health_mean"), 0.0)
    )


def selection_key(eval_metrics: Dict[str, float], cfg: TrainConfig) -> Tuple[float, float, float, float, float, float]:
    return (
        composite_eval_score(eval_metrics, cfg),
        float(eval_metrics.get("eval_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_150_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_100_mean", float("-inf"))),
        float(eval_metrics.get("eval_reward_mean", float("-inf"))),
        -float(eval_metrics.get("eval_wall_hit_rate_mean", float("inf"))),
    )


def checkpoint_payload(
    online_net: RecurrentPatchFusionDuelingAuxQNetwork,
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
        "model_arch": "patch_fusion_gru_dueling_double_dqn_windowed_coverage",
        "model_kwargs": {
            "patch_size": env.config.patch_size,
            "num_actions": N_ACTIONS,
            "use_vision": switches.include_vision,
            "use_temperature_patch": switches.include_temperature_patch,
            "use_smell_patch": switches.include_smell_patch,
            "use_visited_memory": switches.include_visited_memory,
            "use_hazard_memory": switches.include_hazard_memory,
            "scalar_state_dim": 6,
            "conv_channels_1": cfg.conv_channels_1,
            "conv_channels_2": cfg.conv_channels_2,
            "vision_embed_dim": cfg.vision_embed_dim,
            "scalar_patch_embed_dim": cfg.scalar_patch_embed_dim,
            "scalar_state_embed_dim": cfg.scalar_state_embed_dim,
            "obs_embed_dim": cfg.obs_embed_dim,
            "gru_hidden_dim": cfg.gru_hidden_dim,
            "head_hidden_dim": cfg.head_hidden_dim,
            "uses_prev_action": True,
            "uses_prev_reward": True,
            "predicts_forward_health_delta": True,
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
    with open(save_dir / "train_config.txt", "w", encoding="utf-8") as f:
        f.write("model_arch: patch_fusion_gru_dueling_double_dqn_windowed_coverage\n")
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")
        f.write(f"input_dim: {input_dim}\n")
        f.write(f"num_actions: {N_ACTIONS}\n")
        f.write(f"actions: forward={ACTION_FORWARD}, left={ACTION_LEFT}, right={ACTION_RIGHT}\n")
        f.write("\n[env_config]\n")
        for k, v in asdict(env.config).items():
            f.write(f"{k}: {v}\n")
        f.write("\n[switches]\n")
        for k, v in asdict(switches).items():
            f.write(f"{k}: {v}\n")


def choose_action(
    net: RecurrentPatchFusionDuelingAuxQNetwork,
    obs: Dict[str, Any],
    state: Dict[str, np.ndarray],
    prev_action: np.ndarray,
    prev_reward: float,
    hidden: torch.Tensor | None,
    device: torch.device,
    epsilon: float,
    cfg: TrainConfig,
) -> Tuple[int, torch.Tensor]:
    state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action_t = torch.from_numpy(prev_action).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[prev_reward]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, new_hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        q_values = q[:, -1].squeeze(0).detach().cpu().numpy()
        if random.random() < epsilon:
            if cfg.use_structured_exploration:
                probs = structured_exploration_probs(obs, prev_action, q_values, cfg)
                action = int(np.random.choice(np.arange(N_ACTIONS), p=probs))
            else:
                action = random.randrange(N_ACTIONS)
        else:
            action = int(np.argmax(q_values))
    return action, new_hidden.detach()


def episode_priority(last_info: Dict[str, object], ep_reward: float, diag_metrics: Dict[str, float]) -> float:
    coverage = float(last_info.get("coverage", 0.0))
    reward_pos = max(float(ep_reward), 0.0)
    cov_50 = float(diag_metrics.get("coverage_at_step_50", 0.0))
    cov_100 = float(diag_metrics.get("coverage_at_step_100", 0.0))
    cov_150 = float(diag_metrics.get("coverage_at_step_150", 0.0))
    if np.isnan(cov_50):
        cov_50 = coverage
    if np.isnan(cov_100):
        cov_100 = coverage
    if np.isnan(cov_150):
        cov_150 = coverage
    new_tile_rate = float(diag_metrics.get("new_tile_rate", 0.0))
    revisit_rate = float(diag_metrics.get("revisit_rate", 0.0))
    wall_hit_rate = float(diag_metrics.get("wall_hit_rate", 0.0))
    hazard_contact_rate = float(diag_metrics.get("hazard_contact_rate", 0.0))
    stagnation_step_rate = float(diag_metrics.get("stagnation_step_rate", 0.0))
    coverage_gain_25 = float(diag_metrics.get("coverage_gain_25", 0.0))
    turn_oscillation_rate = float(diag_metrics.get("turn_oscillation_rate", 0.0))
    forward_after_turn_success_rate = float(diag_metrics.get("forward_after_turn_success_rate", 0.0))
    return (
        1.0
        + 1.8 * coverage
        + 0.35 * cov_50
        + 0.55 * cov_100
        + 0.75 * cov_150
        + 0.90 * new_tile_rate
        + 0.80 * coverage_gain_25
        + 0.18 * forward_after_turn_success_rate
        - 0.20 * revisit_rate
        - 0.35 * wall_hit_rate
        - 0.30 * hazard_contact_rate
        - 0.22 * stagnation_step_rate
        - 0.10 * turn_oscillation_rate
        + 0.02 * reward_pos
    )


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "sensory_grid_env_v4_1",
        "model_family": "patch_fusion_gru_dueling_double_dqn_aux",
        "default_switches": asdict(build_default_switches()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Windowed-coverage recurrent Double DQN")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--save_dir", type=str, default="runs/windowed_coverage_dqn")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_eval_episodes", type=int, default=10)
    parser.add_argument("--disable_structured_exploration", action="store_true")
    parser.add_argument("--enable_state_adaptive_exploration", action="store_true")
    parser.add_argument("--disable_window_priority_sampling", action="store_true")
    parser.add_argument("--disable_exploration_reward_shaping", action="store_true")
    parser.add_argument("--frontier_score_scale", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        eval_episodes=args.eval_episodes,
        holdout_eval_episodes=args.holdout_eval_episodes,
        use_state_adaptive_exploration=args.enable_state_adaptive_exploration,
        use_window_priority_sampling=not args.disable_window_priority_sampling,
        use_exploration_reward_shaping=not args.disable_exploration_reward_shaping,
        exploration_frontier_score_scale=(
            TrainConfig.exploration_frontier_score_scale
            if args.frontier_score_scale is None
            else args.frontier_score_scale
        ),
        use_structured_exploration=not args.disable_structured_exploration,
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
    replay = EpisodeSequenceReplayBuffer(cfg.replay_capacity_episodes, cfg.replay_priority_alpha)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_train_config(save_dir, cfg, env, switches, input_dim)

    csv_path = save_dir / "training_log.csv"
    episode_metric_names = episode_diagnostic_metric_names()
    eval_fieldnames = eval_metric_names("eval")
    holdout_eval_fieldnames = eval_metric_names("holdout_eval")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
            "mean_q_loss",
            "mean_aux_loss",
            "aux_weight",
            *eval_fieldnames,
            *holdout_eval_fieldnames,
            "eval_soft_score",
        ])

    best_selection = (-float("inf"),) * 6
    best_eval_reward = -float("inf")
    best_eval_coverage = -float("inf")
    best_eval_survival = -float("inf")
    best_eval_soft = -float("inf")
    global_step = 0

    for episode in range(1, cfg.episodes + 1):
        obs, reset_info = env.reset(seed=cfg.seed + episode)
        state = obs_to_state(obs, env.config, switches)
        done = False
        ep_reward = 0.0
        losses: List[float] = []
        q_losses: List[float] = []
        aux_losses: List[float] = []
        last_info = None
        hidden = None
        prev_action = np.zeros(N_ACTIONS, dtype=np.float32)
        prev_reward = 0.0
        episode_transitions: List[Dict[str, object]] = []
        episode_diag = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))
        current_coverage = float(reset_info.get("coverage", 0.0))

        while not done:
            epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
            action, hidden = choose_action(online_net, obs, state, prev_action, prev_reward, hidden, device, epsilon, cfg)
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            next_state = obs_to_state(next_obs, env.config, switches)
            done = terminated or truncated
            update_episode_diagnostics(episode_diag, action, info)
            next_coverage = float(info.get("coverage", current_coverage))
            coverage_delta = max(0.0, next_coverage - current_coverage)
            reward_terms = info.get("reward_terms", {})
            wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
            revisit = bool(action == ACTION_FORWARD and not wall_hit and coverage_delta <= 1e-9)
            energy_spent = transition_energy_spent(info)
            stagnating = int(episode_diag.get("steps_since_new_tile", 0)) >= int(cfg.exploration_stagnation_trigger_steps)
            sample_weight = 1.0
            if cfg.use_window_priority_sampling:
                sample_weight = transition_sample_weight(
                    coverage_delta,
                    energy_spent,
                    revisit,
                    wall_hit,
                    stagnating,
                    cfg,
                )

            episode_transitions.append({
                "state": copy_state(state),
                "prev_action": prev_action.copy(),
                "prev_reward": float(prev_reward),
                "action": int(action),
                "reward": float(reward),
                "next_state": copy_state(next_state),
                "done": float(done),
                "next_prev_action": onehot_action(action),
                "next_prev_reward": float(reward),
                "health_delta": float(info.get("health_delta", 0.0)),
                "coverage_delta": float(coverage_delta),
                "energy_spent": float(energy_spent),
                "revisit": float(revisit),
                "step_index": float(info.get("steps", 0)),
                "sample_weight": float(sample_weight),
            })

            obs = next_obs
            state = next_state
            prev_action = onehot_action(action)
            prev_reward = float(reward)
            ep_reward += reward
            last_info = info
            current_coverage = next_coverage
            global_step += 1

        assert last_info is not None
        episode_diag_metrics = finalize_episode_diagnostics(episode_diag)
        replay.add_episode(episode_transitions, priority=episode_priority(last_info, ep_reward, episode_diag_metrics))

        aux_weight = current_aux_weight(episode, cfg)
        if len(replay) >= cfg.train_after_episodes:
            for _ in range(cfg.train_updates_per_episode):
                total_loss, q_loss, aux_loss = train_step(online_net, target_net, optimizer, replay, device, cfg, aux_weight)
                losses.append(total_loss)
                q_losses.append(q_loss)
                aux_losses.append(aux_loss)

        soft_update(target_net, online_net, cfg.target_soft_tau)
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
        mean_q_loss = float(np.mean(q_losses)) if q_losses else 0.0
        mean_aux_loss = float(np.mean(aux_losses)) if aux_losses else 0.0
        epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
        eval_score = composite_eval_score(eval_metrics, cfg) if do_eval else float("nan")
        print(
            f"episode={episode:04d} step={global_step:06d} eps={epsilon:.3f} "
            f"reward={ep_reward:+.3f} coverage={last_info['coverage']:.3f} "
            f"health={last_info['health']} len={last_info['steps']} "
            f"new={episode_diag_metrics['new_tile_rate']:.3f} revisit={episode_diag_metrics['revisit_rate']:.3f} "
            f"wall={episode_diag_metrics['wall_hit_rate']:.3f} hazard={episode_diag_metrics['hazard_contact_rate']:.3f} "
            f"stagnation={episode_diag_metrics['stagnation_step_rate']:.3f} c25={episode_diag_metrics['coverage_gain_25']:.3f} "
            f"loss={mean_loss:.4f} q={mean_q_loss:.4f} aux={mean_aux_loss:.4f} aw={aux_weight:.3f} "
            f"eval_surv={eval_metrics['eval_survival_rate']:.3f} "
            f"eval_cov={eval_metrics['eval_coverage_mean']:.3f} "
            f"eval_reward={eval_metrics['eval_reward_mean']:.3f} "
            f"holdout_cov={holdout_eval_metrics['holdout_eval_coverage_mean']:.3f} "
            f"holdout_reward={holdout_eval_metrics['holdout_eval_reward_mean']:.3f} "
            f"soft={eval_score:.3f}",
            flush=True,
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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
                mean_q_loss,
                mean_aux_loss,
                aux_weight,
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

            if eval_metrics["eval_reward_mean"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_reward_mean"]
                torch.save(payload, save_dir / "best_reward_model.pt")

            if eval_metrics["eval_coverage_mean"] > best_eval_coverage:
                best_eval_coverage = eval_metrics["eval_coverage_mean"]
                torch.save(payload, save_dir / "best_coverage_model.pt")

            if eval_metrics["eval_survival_rate"] > best_eval_survival:
                best_eval_survival = eval_metrics["eval_survival_rate"]
                torch.save(payload, save_dir / "best_survival_model.pt")

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

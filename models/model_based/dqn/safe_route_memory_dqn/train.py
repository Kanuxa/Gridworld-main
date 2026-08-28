from __future__ import annotations

import argparse
import csv
import heapq
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import random
import resource
import sys
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
    reset_runtime_context,
    update_runtime_context_after_env_step,
)

TRAINER_GUI_INTERFACE_VERSION = "safe-route-memory-dqn"
TRAINER_DISPLAY_NAME = "Safe-route memory recurrent Double DQN"
EXPLORATION_COVERAGE_BUDGETS = (50, 100, 150)
HAZARD_OBJECT_IDS = {OBJ_FIRE, OBJ_ICE, OBJ_GLASS}
# Fire and ice create a local exclusion zone. Glass is excluded only at its own
# cell, while thermal gradients outside the local zone remain explorable.
THERMAL_EXCLUSION_OBJECT_IDS = {OBJ_FIRE, OBJ_ICE}
EXPLORATION_PROGRESS_BLOCK_STEPS = 25
EXPLORATION_STAGNATION_LOG_THRESHOLD = 6
CENTER_REGION_MARGIN = 1


@dataclass
class TrainConfig:
    episodes: int = 1200
    seed: int = 7
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_gradient_norm: float = 10.0
    save_dir: str = "runs/safe_route_memory_dqn"
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
    # A single, atomically replaced recovery checkpoint.  It keeps an interrupted
    # run recoverable without producing one checkpoint file per episode.
    last_checkpoint_every_episodes: int = 5

    conv_channels_1: int = 16
    conv_channels_2: int = 32
    vision_embed_dim: int = 96
    scalar_patch_embed_dim: int = 48
    scalar_state_embed_dim: int = 32
    obs_embed_dim: int = 256
    gru_hidden_dim: int = 256
    head_hidden_dim: int = 128

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
    best_model_seen_weight: float = 0.90
    best_model_seen_100_weight: float = 0.35
    best_model_seen_150_weight: float = 0.45
    best_model_center_seen_weight: float = 0.20

    # v5 change 2: slight survival pull, but still weaker than v4.1.
    training_survival_bonus: float = 0.60

    # v6 change 1: reward shaping improves credit assignment toward faster,
    # cleaner coverage growth without changing the environment.
    use_exploration_reward_shaping: bool = True
    train_coverage_delta_reward_scale: float = 6.0
    train_early_progress_reward_scale: float = 4.0
    train_post_seen_coverage_delta_reward_scale: float = 20.0
    train_shaping_step_horizon: int = 150
    # Energy use and direct hazards should not dominate the exploration objective.
    # The game still applies its original costs; these values only shape Q targets.
    train_no_progress_energy_penalty_scale: float = 0.005
    train_revisit_penalty_scale: float = 0.008
    train_seen_delta_reward_scale: float = 3.0
    train_center_seen_delta_reward_scale: float = 3.8
    train_planner_alignment_bonus_scale: float = 0.18

    # v6 change 2: replay focuses more on productive exploration windows than
    # on stagnant late-episode segments from otherwise good episodes.
    use_window_priority_sampling: bool = True
    replay_window_new_tile_bonus: float = 1.75
    replay_window_coverage_delta_scale: float = 140.0
    replay_future_coverage_weight: float = 60.0
    replay_obs_novelty_weight: float = 0.50
    replay_window_revisit_penalty: float = 0.45
    replay_window_wall_penalty: float = 0.08
    replay_window_stagnation_penalty: float = 0.35
    replay_window_energy_penalty: float = 0.05
    replay_episode_wall_penalty: float = 0.05
    replay_episode_hazard_penalty: float = 0.0

    # v8 change 1: explicitly reward setup actions that open exploration a few
    # steps later, not only the final forward move that collects the new tile.
    future_coverage_horizon_steps: int = 10
    train_future_coverage_bonus_scale: float = 2.0
    train_post_seen_future_coverage_bonus_scale: float = 3.0
    train_future_turn_bonus_multiplier: float = 1.15
    train_future_forward_bonus_multiplier: float = 0.30

    # v8 change 2: add light episodic novelty so stagnant local loops get
    # nudged away without overpowering the structured exploration prior.
    use_observation_novelty_bonus: bool = True
    train_obs_novelty_bonus_scale: float = 0.012
    exploration_target_seen_fraction: float = 0.80
    exploration_center_margin: int = CENTER_REGION_MARGIN
    thermal_exclusion_radius: int = 1
    planner_distance_penalty: float = 0.60
    planner_outer_ring_target_penalty: float = 0.30
    planner_route_unvisited_bonus: float = 0.55
    planner_safe_zone_bonus_early: float = 0.28
    planner_safe_zone_bonus_late: float = 1.30
    planner_reveal_bonus: float = 0.90
    planner_center_reveal_bonus: float = 1.75
    planner_exclusion_travel_cost: float = 4.0
    planner_exclusion_target_penalty: float = 4.0
    planner_action_score_scale: float = 0.75
    coverage_goal_commitment_steps: int = 15
    coverage_goal_component_weight: float = 1.00
    coverage_goal_route_unvisited_weight: float = 1.25

    # Multi-step returns propagate delayed coverage rewards through turns and
    # routes rather than crediting only the final forward action.
    n_step_return: int = 3

    use_structured_exploration: bool = True
    exploration_softmax_temperature: float = 0.75
    exploration_forward_unvisited_bonus: float = 2.75
    exploration_forward_revisit_penalty: float = 0.85
    exploration_exclusion_forward_penalty: float = 0.10
    exploration_post_turn_forward_bonus: float = 1.35
    exploration_repeat_turn_penalty: float = 0.45
    exploration_post_bump_forward_penalty: float = 0.75
    exploration_direction_score_scale: float = 0.85
    exploration_direction_unvisited_weight: float = 1.00
    exploration_direction_visited_penalty: float = 0.30
    exploration_direction_exclusion_penalty: float = 2.50
    exploration_frontier_score_scale: float = 0.0
    exploration_frontier_unvisited_weight: float = 0.85
    exploration_frontier_frontier_bonus: float = 0.55
    exploration_frontier_visited_penalty: float = 0.18
    exploration_frontier_exclusion_penalty: float = 2.50
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
            self.priorities.append(max(finite_float(priority, 1.0), 1e-3))

    def __len__(self) -> int:
        return len(self.episodes)

    def _sample_episode(self) -> List[Dict[str, object]]:
        weights = [max(finite_float(p, 1.0), 1e-3) ** self.priority_alpha for p in self.priorities]
        if not weights or not np.isfinite(np.sum(weights)):
            idx = random.randrange(len(self.episodes))
            return self.episodes[idx]
        idx = random.choices(range(len(self.episodes)), weights=weights, k=1)[0]
        return self.episodes[idx]

    def _sample_start_index(self, ep: List[Dict[str, object]], total_len: int) -> int:
        if len(ep) <= 1:
            return 0
        prefix = [0.0]
        for transition in ep:
            prefix.append(prefix[-1] + max(finite_float(transition.get("sample_weight", 1.0), 1.0), 1e-3))
        weights = []
        for start in range(len(ep)):
            end = min(len(ep), start + total_len)
            window_weight = prefix[end] - prefix[start]
            weights.append(max(finite_float(window_weight, 1.0), 1e-3))
        if not np.isfinite(np.sum(weights)):
            return random.randrange(len(ep))
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
        batch_coverage_delta: List[np.ndarray] = []
        batch_seen_delta: List[np.ndarray] = []
        batch_center_seen_delta: List[np.ndarray] = []
        batch_energy_spent: List[np.ndarray] = []
        batch_obs_novelty: List[np.ndarray] = []
        batch_future_coverage_gain: List[np.ndarray] = []
        batch_revisit: List[np.ndarray] = []
        batch_post_seen_phase: List[np.ndarray] = []
        batch_planner_alignment: List[np.ndarray] = []
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
            seq_coverage_delta = np.zeros((total_len,), dtype=np.float32)
            seq_seen_delta = np.zeros((total_len,), dtype=np.float32)
            seq_center_seen_delta = np.zeros((total_len,), dtype=np.float32)
            seq_energy_spent = np.zeros((total_len,), dtype=np.float32)
            seq_obs_novelty = np.zeros((total_len,), dtype=np.float32)
            seq_future_coverage_gain = np.zeros((total_len,), dtype=np.float32)
            seq_revisit = np.zeros((total_len,), dtype=np.float32)
            seq_post_seen_phase = np.zeros((total_len,), dtype=np.float32)
            seq_planner_alignment = np.zeros((total_len,), dtype=np.float32)
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
                    seq_coverage_delta[j] = float(tr.get("coverage_delta", 0.0))
                    seq_seen_delta[j] = float(tr.get("seen_delta", 0.0))
                    seq_center_seen_delta[j] = float(tr.get("center_seen_delta", 0.0))
                    seq_energy_spent[j] = float(tr.get("energy_spent", 0.0))
                    seq_obs_novelty[j] = float(tr.get("obs_novelty", 0.0))
                    seq_future_coverage_gain[j] = float(tr.get("future_coverage_gain", 0.0))
                    seq_revisit[j] = float(tr.get("revisit", 0.0))
                    seq_post_seen_phase[j] = float(tr.get("post_seen_phase", 0.0))
                    seq_planner_alignment[j] = float(tr.get("planner_alignment", 0.0))
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
            batch_coverage_delta.append(seq_coverage_delta)
            batch_seen_delta.append(seq_seen_delta)
            batch_center_seen_delta.append(seq_center_seen_delta)
            batch_energy_spent.append(seq_energy_spent)
            batch_obs_novelty.append(seq_obs_novelty)
            batch_future_coverage_gain.append(seq_future_coverage_gain)
            batch_revisit.append(seq_revisit)
            batch_post_seen_phase.append(seq_post_seen_phase)
            batch_planner_alignment.append(seq_planner_alignment)
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
            "coverage_delta": np.stack(batch_coverage_delta).astype(np.float32),
            "seen_delta": np.stack(batch_seen_delta).astype(np.float32),
            "center_seen_delta": np.stack(batch_center_seen_delta).astype(np.float32),
            "energy_spent": np.stack(batch_energy_spent).astype(np.float32),
            "obs_novelty": np.stack(batch_obs_novelty).astype(np.float32),
            "future_coverage_gain": np.stack(batch_future_coverage_gain).astype(np.float32),
            "revisit": np.stack(batch_revisit).astype(np.float32),
            "post_seen_phase": np.stack(batch_post_seen_phase).astype(np.float32),
            "planner_alignment": np.stack(batch_planner_alignment).astype(np.float32),
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


def obs_to_exploration_state(
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
) -> Dict[str, np.ndarray]:
    """Build the v9 policy state while withholding current health from the agent."""
    state = obs_to_state(obs, env_cfg, switches)
    scalars = np.asarray(state["scalars"], dtype=np.float32).copy()
    # Keep the six-value checkpoint-compatible shape, but health carries no signal.
    scalars[0] = 0.0
    state["scalars"] = scalars
    return state


def state_seq_batch_to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def observation_novelty_key(obs: Dict[str, Any]) -> bytes:
    vision = np.asarray(obs.get("vision"), dtype=np.uint8)
    visited = (
        np.asarray(obs.get("visited_patch", np.zeros_like(vision, dtype=np.float32)), dtype=np.float32) > 0.5
    ).astype(np.uint8)
    hazard = (
        np.asarray(obs.get("hazard_patch", np.zeros_like(vision, dtype=np.float32)), dtype=np.float32) >= 0.25
    ).astype(np.uint8)
    direction = np.asarray(obs.get("direction_onehot", np.zeros((4,), dtype=np.float32)), dtype=np.float32)
    direction = direction.round().astype(np.uint8)
    no_move = min(int(obs.get("consecutive_no_move_steps", 0)), 3)
    turn_streak = min(int(obs.get("consecutive_turn_steps", 0)), 3)
    return b"".join(
        (
            vision.tobytes(),
            visited.tobytes(),
            hazard.tobytes(),
            direction.tobytes(),
            bytes((no_move, turn_streak)),
        )
    )


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
    seen_delta: float,
    center_seen_delta: float,
    energy_spent: float,
    future_coverage_gain: float,
    obs_novelty: float,
    revisit: bool,
    wall_hit: bool,
    stagnating: bool,
    planner_alignment: float,
    cfg: TrainConfig,
) -> float:
    weight = 1.0
    weight += float(cfg.replay_window_coverage_delta_scale) * max(0.0, float(coverage_delta))
    weight += 85.0 * max(0.0, float(seen_delta))
    weight += 110.0 * max(0.0, float(center_seen_delta))
    weight += float(cfg.replay_future_coverage_weight) * max(0.0, float(future_coverage_gain))
    weight += float(cfg.replay_obs_novelty_weight) * max(0.0, float(obs_novelty))
    weight += 0.30 * max(0.0, float(planner_alignment))
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
        "seen_coverage",
        "center_seen_coverage",
        "action_forward_rate",
        "action_left_rate",
        "action_right_rate",
        "max_no_new_tile_streak",
        "stagnation_step_rate",
        "coverage_gain_25",
        "energy_spent",
        "coverage_gain_per_energy",
        "new_tiles_per_energy",
        "obs_novelty_mean",
        "turn_oscillation_rate",
        "forward_after_turn_success_rate",
    ]
    metrics.extend([f"coverage_at_step_{budget}" for budget in EXPLORATION_COVERAGE_BUDGETS])
    metrics.extend([f"seen_at_step_{budget}" for budget in EXPLORATION_COVERAGE_BUDGETS])
    metrics.extend([f"center_seen_at_step_{budget}" for budget in EXPLORATION_COVERAGE_BUDGETS])
    return metrics


def eval_metric_names(prefix: str) -> List[str]:
    base_metrics = [
        f"{prefix}_reward_mean",
        f"{prefix}_coverage_mean",
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


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def center_region_mask(grid_size: int, margin: int = CENTER_REGION_MARGIN) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    inner_margin = int(max(0, min(margin, grid_size // 2)))
    if inner_margin == 0:
        mask[:] = True
    else:
        mask[inner_margin:grid_size - inner_margin, inner_margin:grid_size - inner_margin] = True
    return mask


def reveal_visible_cells(
    seen_map: np.ndarray,
    agent_pos: Tuple[int, int],
    patch_size: int,
    center_mask: np.ndarray,
) -> Tuple[float, float]:
    half = patch_size // 2
    r0 = max(0, int(agent_pos[0]) - half)
    r1 = min(seen_map.shape[0], int(agent_pos[0]) + half + 1)
    c0 = max(0, int(agent_pos[1]) - half)
    c1 = min(seen_map.shape[1], int(agent_pos[1]) + half + 1)

    visible_seen = seen_map[r0:r1, c0:c1]
    new_mask = visible_seen < 0.5
    if not bool(np.any(new_mask)):
        return 0.0, 0.0

    center_visible = center_mask[r0:r1, c0:c1]
    new_seen = float(np.count_nonzero(new_mask)) / max(1.0, float(seen_map.size))
    new_center_seen = float(np.count_nonzero(new_mask & center_visible)) / max(1.0, float(np.count_nonzero(center_mask)))
    visible_seen[new_mask] = 1.0
    seen_map[r0:r1, c0:c1] = visible_seen
    return new_seen, new_center_seen


def init_episode_diagnostics(env: SensoryGridEnv, cfg: TrainConfig, initial_coverage: float = 0.0) -> Dict[str, object]:
    center_mask = center_region_mask(int(env.config.grid_size), int(cfg.exploration_center_margin))
    seen_map = np.zeros((env.config.grid_size, env.config.grid_size), dtype=np.float32)
    reveal_visible_cells(seen_map, env.agent_pos, env.config.patch_size, center_mask)
    return {
        "initial_coverage": float(initial_coverage),
        "steps": 0,
        "new_tile_steps": 0,
        "revisit_steps": 0,
        "wall_hits": 0,
        "turn_streak_steps": 0,
        "hazard_contacts": 0,
        "energy_spent": 0.0,
        "obs_novelty_total": 0.0,
        "action_counts": {
            ACTION_FORWARD: 0,
            ACTION_LEFT: 0,
            ACTION_RIGHT: 0,
        },
        "coverage_by_budget": {
            budget: float("nan") for budget in EXPLORATION_COVERAGE_BUDGETS
        },
        "seen_by_budget": {
            budget: float("nan") for budget in EXPLORATION_COVERAGE_BUDGETS
        },
        "center_seen_by_budget": {
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
        "coverage_goal_target": None,
        "coverage_goal_steps_remaining": 0,
        "center_mask": center_mask,
        "seen_map": seen_map,
        "seen_fraction": float(np.mean(seen_map > 0.5)),
        "center_seen_fraction": float(np.mean(seen_map[center_mask] > 0.5)),
    }


def update_episode_diagnostics(
    diag: Dict[str, object],
    env: SensoryGridEnv,
    action: int,
    info: Dict[str, object],
) -> Tuple[float, float]:
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
    seen_by_budget = diag["seen_by_budget"]
    center_seen_by_budget = diag["center_seen_by_budget"]
    center_mask = np.asarray(diag["center_mask"], dtype=bool)
    seen_map = np.asarray(diag["seen_map"], dtype=np.float32)
    assert isinstance(coverage_by_budget, dict)
    assert isinstance(seen_by_budget, dict)
    assert isinstance(center_seen_by_budget, dict)

    seen_delta, center_seen_delta = reveal_visible_cells(seen_map, env.agent_pos, env.config.patch_size, center_mask)
    diag["seen_fraction"] = float(np.mean(seen_map > 0.5))
    diag["center_seen_fraction"] = float(np.mean(seen_map[center_mask] > 0.5))

    for budget in EXPLORATION_COVERAGE_BUDGETS:
        if step_index >= budget and np.isnan(float(coverage_by_budget[budget])):
            coverage_by_budget[budget] = coverage
        if step_index >= budget and np.isnan(float(seen_by_budget[budget])):
            seen_by_budget[budget] = float(diag["seen_fraction"])
        if step_index >= budget and np.isnan(float(center_seen_by_budget[budget])):
            center_seen_by_budget[budget] = float(diag["center_seen_fraction"])

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
    return seen_delta, center_seen_delta


def finalize_episode_diagnostics(diag: Dict[str, object]) -> Dict[str, float]:
    steps = max(1, int(diag["steps"]))
    action_counts = diag["action_counts"]
    coverage_by_budget = diag["coverage_by_budget"]
    seen_by_budget = diag["seen_by_budget"]
    center_seen_by_budget = diag["center_seen_by_budget"]
    assert isinstance(action_counts, dict)
    assert isinstance(coverage_by_budget, dict)
    assert isinstance(seen_by_budget, dict)
    assert isinstance(center_seen_by_budget, dict)
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
        "seen_coverage": float(diag["seen_fraction"]),
        "center_seen_coverage": float(diag["center_seen_fraction"]),
        "action_forward_rate": float(action_counts.get(ACTION_FORWARD, 0)) / steps,
        "action_left_rate": float(action_counts.get(ACTION_LEFT, 0)) / steps,
        "action_right_rate": float(action_counts.get(ACTION_RIGHT, 0)) / steps,
        "max_no_new_tile_streak": float(diag["max_no_new_tile_streak"]),
        "stagnation_step_rate": float(diag["stagnation_steps"]) / steps,
        "coverage_gain_25": nanmean(coverage_gain_blocks),
        "energy_spent": total_energy_spent,
        "coverage_gain_per_energy": coverage_gain_total / effective_energy,
        "new_tiles_per_energy": float(diag["new_tile_steps"]) / effective_energy,
        "obs_novelty_mean": float(diag["obs_novelty_total"]) / steps,
        "turn_oscillation_rate": float(diag["turn_oscillation_events"]) / steps,
        "forward_after_turn_success_rate": (
            float(diag["forward_after_turn_successes"]) / max(1, int(diag["forward_after_turn_attempts"]))
        ),
    }
    for budget in EXPLORATION_COVERAGE_BUDGETS:
        metrics[f"coverage_at_step_{budget}"] = float(coverage_by_budget[budget])
        metrics[f"seen_at_step_{budget}"] = float(seen_by_budget[budget])
        metrics[f"center_seen_at_step_{budget}"] = float(center_seen_by_budget[budget])
    return metrics


def discomfort_map(env: SensoryGridEnv) -> np.ndarray:
    temp = np.asarray(env.temperature_field_c, dtype=np.float32)
    low = float(env.config.comfort_low_c)
    high = float(env.config.comfort_high_c)
    scale = max(1e-6, float(env.config.discomfort_temp_scale_c))
    below = np.maximum(low - temp, 0.0)
    above = np.maximum(temp - high, 0.0)
    return np.clip((below + above) / scale, 0.0, 1.0).astype(np.float32)


def thermal_exclusion_mask(grid: np.ndarray, radius: int) -> np.ndarray:
    """Mark the local square around fire and ice."""
    object_grid = np.asarray(grid, dtype=np.int64)
    mask = np.zeros(object_grid.shape, dtype=bool)
    margin = max(0, int(radius))
    for row, col in np.argwhere(np.isin(object_grid, tuple(THERMAL_EXCLUSION_OBJECT_IDS))):
        r0 = max(0, int(row) - margin)
        r1 = min(mask.shape[0], int(row) + margin + 1)
        c0 = max(0, int(col) - margin)
        c1 = min(mask.shape[1], int(col) + margin + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def exploration_exclusion_mask(grid: np.ndarray, thermal_radius: int) -> np.ndarray:
    """Exclude fire/ice neighbourhoods and the exact glass cells only."""
    object_grid = np.asarray(grid, dtype=np.int64)
    mask = thermal_exclusion_mask(object_grid, thermal_radius)
    mask |= object_grid == OBJ_GLASS
    return mask


def safe_zone_size_map(env: SensoryGridEnv, cfg: TrainConfig) -> Tuple[np.ndarray, float]:
    grid = np.asarray(env.grid, dtype=np.int64)
    safe_mask = ~exploration_exclusion_mask(grid, cfg.thermal_exclusion_radius)
    zone_sizes = np.zeros(grid.shape, dtype=np.float32)
    visited = np.zeros(grid.shape, dtype=bool)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not safe_mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            component: List[Tuple[int, int]] = []
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                component.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and safe_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            size = float(len(component))
            for cr, cc in component:
                zone_sizes[cr, cc] = size
    return zone_sizes, float(np.count_nonzero(safe_mask))


def reveal_scores_for_position(
    seen_map: np.ndarray,
    center_mask: np.ndarray,
    pos: Tuple[int, int],
    patch_size: int,
) -> Tuple[float, float]:
    half = patch_size // 2
    r0 = max(0, int(pos[0]) - half)
    r1 = min(seen_map.shape[0], int(pos[0]) + half + 1)
    c0 = max(0, int(pos[1]) - half)
    c1 = min(seen_map.shape[1], int(pos[1]) + half + 1)
    unseen = seen_map[r0:r1, c0:c1] < 0.5
    center_visible = center_mask[r0:r1, c0:c1]
    patch_area = float(max(1, patch_size * patch_size))
    return float(np.count_nonzero(unseen)) / patch_area, float(np.count_nonzero(unseen & center_visible)) / patch_area


def route_action_candidates(current_pos: Tuple[int, int], current_dir: int, next_pos: Tuple[int, int]) -> List[Tuple[int, float]]:
    dr = int(next_pos[0]) - int(current_pos[0])
    dc = int(next_pos[1]) - int(current_pos[1])
    desired_dir = None
    if dr == -1 and dc == 0:
        desired_dir = 0
    elif dr == 0 and dc == 1:
        desired_dir = 1
    elif dr == 1 and dc == 0:
        desired_dir = 2
    elif dr == 0 and dc == -1:
        desired_dir = 3
    if desired_dir is None:
        return []
    turn_delta = (desired_dir - int(current_dir)) % 4
    if turn_delta == 0:
        return [(ACTION_FORWARD, 1.0)]
    if turn_delta == 1:
        return [(ACTION_RIGHT, 1.0)]
    if turn_delta == 3:
        return [(ACTION_LEFT, 1.0)]
    return [(ACTION_LEFT, 0.85), (ACTION_RIGHT, 0.85)]


def dijkstra_route_data(cost_map: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int]]]:
    dist = np.full(cost_map.shape, np.inf, dtype=np.float32)
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heap: List[Tuple[float, Tuple[int, int]]] = []
    dist[start] = 0.0
    heapq.heappush(heap, (0.0, start))

    while heap:
        cur_cost, (r, c) = heapq.heappop(heap)
        if cur_cost > float(dist[r, c]) + 1e-9:
            continue
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < cost_map.shape[0] and 0 <= nc < cost_map.shape[1]):
                continue
            step_cost = float(cost_map[nr, nc])
            next_cost = float(cur_cost) + step_cost
            if next_cost + 1e-9 < float(dist[nr, nc]):
                dist[nr, nc] = next_cost
                parents[(nr, nc)] = (r, c)
                heapq.heappush(heap, (next_cost, (nr, nc)))
    return dist, parents


def reconstruct_path(
    parents: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if goal == start:
        return [start]
    if goal not in parents:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = parents[cur]
        path.append(cur)
    path.reverse()
    return path


def unvisited_components(unvisited_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Return 4-connected, currently visitable unvisited regions."""
    mask = np.asarray(unvisited_mask, dtype=bool)
    explored = np.zeros(mask.shape, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or explored[row, col]:
                continue
            queue = deque([(row, col)])
            explored[row, col] = True
            component: List[Tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < mask.shape[0]
                        and 0 <= next_col < mask.shape[1]
                        and mask[next_row, next_col]
                        and not explored[next_row, next_col]
                    ):
                        explored[next_row, next_col] = True
                        queue.append((next_row, next_col))
            components.append(component)
    return components


def path_action_scores(
    current_pos: Tuple[int, int],
    current_dir: int,
    path: List[Tuple[int, int]],
) -> np.ndarray | None:
    if len(path) < 2:
        return None
    scores = np.full((N_ACTIONS,), -3.0, dtype=np.float64)
    candidates = route_action_candidates(current_pos, current_dir, path[1])
    if not candidates:
        return None
    for action, weight in candidates:
        scores[action] = max(scores[action], float(np.log(max(weight, 1e-6))))
    return scores - np.max(scores)


def coverage_goal_action_scores(
    env: SensoryGridEnv,
    diag: Dict[str, object],
    cfg: TrainConfig,
    visited_map: np.ndarray,
    exclusion_mask: np.ndarray,
    dist: np.ndarray,
    parents: Dict[Tuple[int, int], Tuple[int, int]],
) -> np.ndarray | None:
    """Commit briefly to the densest reachable unvisited component and route."""
    current_pos = tuple(env.agent_pos)
    unvisited_mask = (visited_map < 0.5) & ~exclusion_mask
    target_value = diag.get("coverage_goal_target")
    remaining = int(diag.get("coverage_goal_steps_remaining", 0))
    target: Tuple[int, int] | None = None
    if (
        isinstance(target_value, tuple)
        and len(target_value) == 2
        and remaining > 0
        and 0 <= int(target_value[0]) < unvisited_mask.shape[0]
        and 0 <= int(target_value[1]) < unvisited_mask.shape[1]
        and bool(unvisited_mask[int(target_value[0]), int(target_value[1])])
    ):
        target = (int(target_value[0]), int(target_value[1]))
        path = reconstruct_path(parents, current_pos, target)
        scores = path_action_scores(current_pos, int(env.direction), path)
        if scores is not None:
            diag["coverage_goal_steps_remaining"] = remaining - 1
            return scores

    best_target: Tuple[int, int] | None = None
    best_path: List[Tuple[int, int]] = []
    best_score = -float("inf")
    for component in unvisited_components(unvisited_mask):
        component_target: Tuple[int, int] | None = None
        component_path: List[Tuple[int, int]] = []
        component_route_unvisited = -1
        for candidate in component:
            if not np.isfinite(float(dist[candidate])):
                continue
            path = reconstruct_path(parents, current_pos, candidate)
            if len(path) < 2:
                continue
            route_unvisited = int(sum(bool(unvisited_mask[row, col]) for row, col in path[1:]))
            # Prefer a target that carries the agent through a substantial route,
            # not merely the nearest edge of an otherwise useful component.
            if route_unvisited > component_route_unvisited or (
                route_unvisited == component_route_unvisited and len(path) > len(component_path)
            ):
                component_target = candidate
                component_path = path
                component_route_unvisited = route_unvisited
        if component_target is None:
            continue
        route_cost = max(1.0, float(dist[component_target]))
        score = (
            float(cfg.coverage_goal_component_weight) * float(len(component))
            + float(cfg.coverage_goal_route_unvisited_weight) * float(component_route_unvisited)
        ) / route_cost
        if score > best_score:
            best_score = score
            best_target = component_target
            best_path = component_path

    if best_target is None:
        diag["coverage_goal_target"] = None
        diag["coverage_goal_steps_remaining"] = 0
        return None
    scores = path_action_scores(current_pos, int(env.direction), best_path)
    if scores is None:
        return None
    diag["coverage_goal_target"] = best_target
    diag["coverage_goal_steps_remaining"] = max(0, int(cfg.coverage_goal_commitment_steps) - 1)
    return scores


def planner_action_scores(env: SensoryGridEnv, diag: Dict[str, object], cfg: TrainConfig) -> np.ndarray:
    seen_map = np.asarray(diag["seen_map"], dtype=np.float32)
    center_mask = np.asarray(diag["center_mask"], dtype=bool)
    seen_fraction = float(diag.get("seen_fraction", np.mean(seen_map > 0.5)))
    visited_map = np.asarray(env.visited_map, dtype=np.float32)
    grid = np.asarray(env.grid, dtype=np.int64)
    exclusion_mask = exploration_exclusion_mask(grid, cfg.thermal_exclusion_radius)
    zone_sizes, safe_zone_total = safe_zone_size_map(env, cfg)
    route_cost = (
        1.0
        + float(cfg.planner_exclusion_travel_cost) * exclusion_mask.astype(np.float32)
    ).astype(np.float32)
    if seen_fraction < float(cfg.exploration_target_seen_fraction):
        route_cost = route_cost + float(cfg.planner_outer_ring_target_penalty) * (~center_mask).astype(np.float32)

    dist, parents = dijkstra_route_data(route_cost, tuple(env.agent_pos))
    if seen_fraction >= float(cfg.exploration_target_seen_fraction):
        goal_scores = coverage_goal_action_scores(
            env,
            diag,
            cfg,
            visited_map,
            exclusion_mask,
            dist,
            parents,
        )
        if goal_scores is not None:
            return goal_scores

    action_scores = np.full((N_ACTIONS,), -np.inf, dtype=np.float64)
    grid_size = float(env.config.grid_size)

    for r in range(env.config.grid_size):
        for c in range(env.config.grid_size):
            if (r, c) == tuple(env.agent_pos) or not np.isfinite(float(dist[r, c])):
                continue
            path = reconstruct_path(parents, tuple(env.agent_pos), (r, c))
            if len(path) < 2:
                continue
            reveal_frac, center_reveal_frac = reveal_scores_for_position(seen_map, center_mask, (r, c), env.config.patch_size)
            path_unvisited_frac = float(
                np.mean([float(visited_map[pr, pc] < 0.5) for pr, pc in path[1:]])
            ) if len(path) > 1 else 0.0
            zone_bonus = float(zone_sizes[r, c]) / max(1.0, safe_zone_total)
            visit_bonus = 1.0 if float(visited_map[r, c]) < 0.5 else 0.0
            norm_dist = float(dist[r, c]) / max(1.0, grid_size)
            target_in_exclusion = float(exclusion_mask[r, c])
            outer_ring = 0.0 if bool(center_mask[r, c]) else 1.0

            if seen_fraction < float(cfg.exploration_target_seen_fraction):
                score = (
                    float(cfg.planner_center_reveal_bonus) * center_reveal_frac
                    + float(cfg.planner_reveal_bonus) * reveal_frac
                    + float(cfg.planner_route_unvisited_bonus) * path_unvisited_frac
                    + float(cfg.planner_safe_zone_bonus_early) * zone_bonus
                    + 0.15 * visit_bonus
                    - float(cfg.planner_distance_penalty) * norm_dist
                    - float(cfg.planner_outer_ring_target_penalty) * outer_ring
                    - float(cfg.planner_exclusion_target_penalty) * target_in_exclusion
                )
            else:
                score = (
                    float(cfg.planner_safe_zone_bonus_late) * zone_bonus
                    + 0.65 * reveal_frac
                    + 0.45 * path_unvisited_frac
                    + 0.20 * visit_bonus
                    - 0.85 * float(cfg.planner_distance_penalty) * norm_dist
                    - 0.50 * float(cfg.planner_exclusion_target_penalty) * target_in_exclusion
                )

            for action, weight in route_action_candidates(tuple(env.agent_pos), int(env.direction), path[1]):
                adjusted_score = score + float(np.log(max(weight, 1e-6)))
                if np.isneginf(action_scores[action]):
                    action_scores[action] = adjusted_score
                else:
                    action_scores[action] = np.logaddexp(action_scores[action], adjusted_score)

    if not np.isfinite(action_scores).any():
        return np.zeros((N_ACTIONS,), dtype=np.float64)
    action_scores = action_scores - np.max(action_scores)
    return action_scores


def extract_front_cell_features(obs: Dict[str, Any], cfg: TrainConfig) -> Dict[str, object]:
    vision_patch = np.asarray(obs.get("vision"), dtype=np.int64)
    centre = vision_patch.shape[0] // 2
    front_r = max(centre - 1, 0)
    front_c = centre

    visited_patch = np.asarray(
        obs.get("visited_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )
    exclusion_patch = exploration_exclusion_mask(vision_patch, cfg.thermal_exclusion_radius)

    front_visited = float(visited_patch[front_r, front_c]) > 0.5
    return {
        "front_unvisited": not front_visited,
        "front_exclusion": bool(exclusion_patch[front_r, front_c]),
    }


def directional_novelty_scores(obs: Dict[str, Any], cfg: TrainConfig) -> np.ndarray:
    vision_patch = np.asarray(obs.get("vision"), dtype=np.int64)
    visited_patch = np.asarray(
        obs.get("visited_patch", np.zeros_like(vision_patch, dtype=np.float32)),
        dtype=np.float32,
    )
    exclusion_patch = exploration_exclusion_mask(vision_patch, cfg.thermal_exclusion_radius)
    centre = vision_patch.shape[0] // 2

    def cell_score(r: int, c: int) -> float:
        if not (0 <= r < vision_patch.shape[0] and 0 <= c < vision_patch.shape[1]):
            return 0.0
        visited = float(visited_patch[r, c])
        score = (
            cfg.exploration_direction_unvisited_weight
            if visited < 0.5
            else -cfg.exploration_direction_visited_penalty
        )
        if bool(exclusion_patch[r, c]):
            score -= cfg.exploration_direction_exclusion_penalty
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
                visited = float(visited_patch[r, c])
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
                if bool(exclusion_patch[r, c]):
                    cell_frontier_score -= float(cfg.exploration_frontier_exclusion_penalty)
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
    planner_scores: np.ndarray | None = None,
) -> np.ndarray:
    temperature = max(float(cfg.exploration_softmax_temperature), 1e-3)
    logits = np.asarray(q_values, dtype=np.float64) / temperature
    logits = logits - np.max(logits)
    base_probs = np.exp(np.clip(logits, -60.0, 60.0))
    if not np.isfinite(base_probs).all() or base_probs.sum() <= 0.0:
        base_probs = np.ones((N_ACTIONS,), dtype=np.float64)

    weights = np.ones((N_ACTIONS,), dtype=np.float64)
    features = extract_front_cell_features(obs, cfg)
    directional_scores = directional_novelty_scores(obs, cfg)
    directional_scores = directional_scores - np.max(directional_scores)
    weights *= np.exp(cfg.exploration_direction_score_scale * directional_scores)
    if planner_scores is not None:
        planner_scores = np.asarray(planner_scores, dtype=np.float64)
        if planner_scores.shape == (N_ACTIONS,) and np.isfinite(planner_scores).any():
            normalized_planner_scores = planner_scores - np.max(planner_scores)
            weights *= np.exp(float(cfg.planner_action_score_scale) * normalized_planner_scores)

    if bool(features["front_unvisited"]):
        weights[ACTION_FORWARD] *= cfg.exploration_forward_unvisited_bonus
    else:
        weights[ACTION_FORWARD] *= cfg.exploration_forward_revisit_penalty

    if bool(features["front_exclusion"]):
        weights[ACTION_FORWARD] *= cfg.exploration_exclusion_forward_penalty

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


def n_step_double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    valid: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    n_step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build terminal-aware n-step targets and a mask for complete targets."""
    steps = max(1, int(n_step))
    batch_size, sequence_len = rewards.shape
    returns = torch.zeros_like(rewards)
    alive = valid > 0.5
    terminal_reached = torch.zeros_like(alive)

    for offset in range(steps):
        shifted_rewards = torch.zeros_like(rewards)
        shifted_dones = torch.ones_like(dones)
        shifted_valid = torch.zeros_like(valid)
        if offset < sequence_len:
            remaining = sequence_len - offset
            shifted_rewards[:, :remaining] = rewards[:, offset:]
            shifted_dones[:, :remaining] = dones[:, offset:]
            shifted_valid[:, :remaining] = valid[:, offset:]
        active = alive & (shifted_valid > 0.5)
        returns = returns + (float(gamma) ** offset) * shifted_rewards * active.to(rewards.dtype)
        terminal_now = active & (shifted_dones > 0.5)
        terminal_reached = terminal_reached | terminal_now
        alive = active & ~terminal_now

    bootstrap_q = torch.zeros_like(next_q)
    if steps <= sequence_len:
        bootstrap_q[:, :sequence_len - steps + 1] = next_q[:, steps - 1:]
    targets = returns + (float(gamma) ** steps) * alive.to(rewards.dtype) * bootstrap_q
    target_available = terminal_reached | alive
    return targets, target_available


def train_step(
    online_net: RecurrentPatchFusionDuelingAuxQNetwork,
    target_net: RecurrentPatchFusionDuelingAuxQNetwork,
    optimizer: optim.Optimizer,
    buffer: EpisodeSequenceReplayBuffer,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[float, float]:
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
    seen_delta_t = torch.from_numpy(batch["seen_delta"]).to(device)
    center_seen_delta_t = torch.from_numpy(batch["center_seen_delta"]).to(device)
    energy_spent_t = torch.from_numpy(batch["energy_spent"]).to(device)
    obs_novelty_t = torch.from_numpy(batch["obs_novelty"]).to(device)
    future_coverage_gain_t = torch.from_numpy(batch["future_coverage_gain"]).to(device)
    revisit_t = torch.from_numpy(batch["revisit"]).to(device)
    post_seen_phase_t = torch.from_numpy(batch["post_seen_phase"]).to(device)
    planner_alignment_t = torch.from_numpy(batch["planner_alignment"]).to(device)
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
    seen_delta_main = seen_delta_t[:, cfg.burn_in:]
    center_seen_delta_main = center_seen_delta_t[:, cfg.burn_in:]
    energy_spent_main = energy_spent_t[:, cfg.burn_in:]
    obs_novelty_main = obs_novelty_t[:, cfg.burn_in:]
    future_coverage_gain_main = future_coverage_gain_t[:, cfg.burn_in:]
    revisit_main = revisit_t[:, cfg.burn_in:]
    post_seen_phase_main = post_seen_phase_t[:, cfg.burn_in:]
    planner_alignment_main = planner_alignment_t[:, cfg.burn_in:]
    step_index_main = step_index_t[:, cfg.burn_in:]

    q_seq, _, _ = online_net.forward_sequence(states_main, prev_actions_main, prev_rewards_main, h_online)
    chosen_q = q_seq.gather(-1, actions_main.unsqueeze(-1)).squeeze(-1)

    shaped_rewards_main = rewards_main
    if cfg.use_exploration_reward_shaping:
        post_seen_phase_main = torch.clamp(post_seen_phase_main, 0.0, 1.0)
        reveal_phase_main = 1.0 - post_seen_phase_main
        horizon = max(1.0, float(cfg.train_shaping_step_horizon))
        early_frac = torch.clamp(1.0 - (step_index_main - 1.0) / horizon, 0.0, 1.0)
        reveal_coverage_bonus = (
            float(cfg.train_coverage_delta_reward_scale)
            + float(cfg.train_early_progress_reward_scale) * early_frac
        ) * coverage_delta_main
        visit_coverage_bonus = float(cfg.train_post_seen_coverage_delta_reward_scale) * coverage_delta_main
        coverage_bonus = reveal_phase_main * reveal_coverage_bonus + post_seen_phase_main * visit_coverage_bonus
        seen_bonus = reveal_phase_main * float(cfg.train_seen_delta_reward_scale) * seen_delta_main
        center_seen_bonus = (
            reveal_phase_main
            * float(cfg.train_center_seen_delta_reward_scale)
            * center_seen_delta_main
            * early_frac
        )
        no_progress = (coverage_delta_main <= 1e-9).to(rewards_main.dtype)
        turn_setup = no_progress * (actions_main != ACTION_FORWARD).to(rewards_main.dtype)
        forward_setup = no_progress * (actions_main == ACTION_FORWARD).to(rewards_main.dtype)
        future_coverage_scale = (
            reveal_phase_main * float(cfg.train_future_coverage_bonus_scale)
            + post_seen_phase_main * float(cfg.train_post_seen_future_coverage_bonus_scale)
        )
        future_bonus = future_coverage_scale * future_coverage_gain_main * (
            float(cfg.train_future_turn_bonus_multiplier) * turn_setup
            + float(cfg.train_future_forward_bonus_multiplier) * forward_setup
        )
        novelty_bonus = torch.zeros_like(rewards_main)
        if cfg.use_observation_novelty_bonus:
            novelty_bonus = float(cfg.train_obs_novelty_bonus_scale) * obs_novelty_main * no_progress
        planner_bonus = reveal_phase_main * float(cfg.train_planner_alignment_bonus_scale) * planner_alignment_main
        energy_penalty = reveal_phase_main * float(cfg.train_no_progress_energy_penalty_scale) * energy_spent_main * no_progress
        revisit_penalty = reveal_phase_main * float(cfg.train_revisit_penalty_scale) * revisit_main
        shaped_rewards_main = (
            rewards_main
            + coverage_bonus
            + seen_bonus
            + center_seen_bonus
            + future_bonus
            + reveal_phase_main * novelty_bonus
            + planner_bonus
            - energy_penalty
            - revisit_penalty
        )

    with torch.no_grad():
        next_online_q, _, _ = online_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_online)
        next_actions = torch.argmax(next_online_q, dim=-1, keepdim=True)
        next_target_q, _, _ = target_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_target)
        next_q = next_target_q.gather(-1, next_actions).squeeze(-1)
        targets, target_available = n_step_double_dqn_targets(
            shaped_rewards_main,
            dones_main,
            valid_main,
            next_q,
            cfg.gamma,
            cfg.n_step_return,
        )

    mask = (valid_main > 0.5) & target_available
    if not bool(mask.any().item()):
        return 0.0, 0.0

    if not torch.isfinite(chosen_q[mask]).all() or not torch.isfinite(targets[mask]).all():
        raise FloatingPointError("Non-finite Q values or TD targets in train_step")
    q_loss = nn.functional.smooth_l1_loss(chosen_q[mask], targets[mask])

    total_loss = q_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite total loss in train_step")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_gradient_norm)
    optimizer.step()
    return float(total_loss.item()), float(q_loss.item())


@torch.no_grad()
def evaluate_policy(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: RecurrentPatchFusionDuelingAuxQNetwork,
    device: torch.device,
    cfg: TrainConfig,
    episodes: int = 5,
    seed_start: int = 1000,
    metric_prefix: str = "eval",
) -> Dict[str, float]:
    if episodes <= 0:
        return empty_eval_metrics(metric_prefix)

    rewards: List[float] = []
    coverages: List[float] = []
    lengths: List[float] = []
    survived_flags: List[float] = []
    death_flags: List[float] = []
    diag_metrics: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, reset_info = env.reset(seed=seed_start + ep)
        state = obs_to_exploration_state(obs, env.config, switches)
        done = False
        ep_reward = 0.0
        hidden = None
        prev_action = np.zeros((1, 1, N_ACTIONS), dtype=np.float32)
        prev_reward = np.zeros((1, 1, 1), dtype=np.float32)
        last_info = None
        diag = init_episode_diagnostics(env, cfg, initial_coverage=float(reset_info.get("coverage", 0.0)))
        novelty_counts: Dict[bytes, int] = {}

        while not done:
            diag["obs_novelty_total"] = float(diag["obs_novelty_total"]) + observation_novelty_bonus(obs, novelty_counts)
            state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
            prev_action_t = torch.from_numpy(prev_action).to(device)
            prev_reward_t = torch.from_numpy(prev_reward).to(device)
            q, hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
            action = int(torch.argmax(q[:, -1], dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            update_episode_diagnostics(diag, env, action, info)
            obs = next_obs
            state = obs_to_exploration_state(next_obs, env.config, switches)
            prev_action[0, 0] = onehot_action(action)
            prev_reward[0, 0, 0] = float(reward)
            ep_reward += reward
            last_info = info
            done = terminated or truncated

        assert last_info is not None
        rewards.append(ep_reward)
        coverages.append(float(last_info["coverage"]))
        lengths.append(float(last_info["steps"]))
        survived_flags.append(1.0 if (last_info["truncated"] and not last_info["terminated"]) else 0.0)
        death_flags.append(1.0 if last_info["terminated"] else 0.0)
        diag_metrics.append(finalize_episode_diagnostics(diag))

    metrics = {
        f"{metric_prefix}_reward_mean": float(np.mean(rewards)),
        f"{metric_prefix}_coverage_mean": float(np.mean(coverages)),
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
    seen = metric("eval_seen_coverage_mean")
    seen_100 = metric("eval_seen_at_step_100_mean")
    seen_150 = metric("eval_seen_at_step_150_mean")
    center_seen = metric("eval_center_seen_coverage_mean")
    return (
        cfg.best_model_seen_weight * seen
        + cfg.best_model_seen_100_weight * seen_100
        + cfg.best_model_seen_150_weight * seen_150
        + cfg.best_model_center_seen_weight * center_seen
        +
        cfg.best_model_coverage_weight * coverage
        + 0.20 * coverage_50
        + 0.35 * coverage_100
        + 0.55 * coverage_150
    )


def selection_key(eval_metrics: Dict[str, float], cfg: TrainConfig) -> Tuple[float, float, float, float, float, float]:
    return (
        composite_eval_score(eval_metrics, cfg),
        float(eval_metrics.get("eval_seen_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_seen_at_step_150_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_150_mean", float("-inf"))),
        float(eval_metrics.get("eval_center_seen_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_new_tile_rate_mean", float("-inf"))),
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
        "model_arch": "patch_fusion_gru_dueling_double_dqn_safe_route_memory",
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
            "uses_health_input": False,
            "predicts_forward_health_delta": False,
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


def training_state_payload(
    online_net: RecurrentPatchFusionDuelingAuxQNetwork,
    target_net: RecurrentPatchFusionDuelingAuxQNetwork,
    optimizer: optim.Optimizer,
    input_dim: int,
    switches: ObservationSwitches,
    env: SensoryGridEnv,
    cfg: TrainConfig,
    episode: int,
    global_step: int,
    eval_metrics: Dict[str, float] | None = None,
    holdout_eval_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Return a recovery checkpoint including the optimizer and target network."""
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
    payload["target_model_state_dict"] = target_net.state_dict()
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["python_random_state"] = random.getstate()
    payload["numpy_random_state"] = np.random.get_state()
    payload["torch_random_state"] = torch.get_rng_state()
    payload["resume_state_version"] = 1
    return payload


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    """Keep the previous recovery checkpoint intact if saving is interrupted."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def peak_rss_mb() -> float:
    """Return the process peak RSS in MiB on macOS and Linux."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def append_run_status(status_path: Path, episode: int, global_step: int, phase: str, **details: object) -> None:
    """Persist the current phase so an external process termination is visible."""
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    line = (
        f"{datetime.now().isoformat(timespec='seconds')} episode={episode:04d} "
        f"step={global_step:06d} phase={phase} peak_rss_mb={peak_rss_mb():.1f}"
    )
    if detail_text:
        line = f"{line} {detail_text}"
    with open(status_path, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")
        f.flush()


def write_train_config(save_dir: Path, cfg: TrainConfig, env: SensoryGridEnv, switches: ObservationSwitches, input_dim: int) -> None:
    with open(save_dir / "train_config.txt", "w", encoding="utf-8") as f:
        f.write("model_arch: patch_fusion_gru_dueling_double_dqn_safe_route_memory\n")
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
    planner_scores: np.ndarray | None = None,
) -> Tuple[int, torch.Tensor]:
    state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action_t = torch.from_numpy(prev_action).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[prev_reward]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, new_hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        q_values = q[:, -1].squeeze(0).detach().cpu().numpy()
        if random.random() < epsilon:
            if cfg.use_structured_exploration:
                probs = structured_exploration_probs(obs, prev_action, q_values, cfg, planner_scores)
                action = int(np.random.choice(np.arange(N_ACTIONS), p=probs))
            else:
                action = random.randrange(N_ACTIONS)
        else:
            action = int(np.argmax(q_values))
    return action, new_hidden.detach()


def predict_action_for_gui(
    net: nn.Module,
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    """Run GUI inference with the same health-masked state used in training."""
    ctx = runtime_context or init_runtime_context()
    device = choose_device(str(ctx.get("device", "cpu")))
    state = obs_to_exploration_state(obs, env_cfg, switches)
    state_t = {k: torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action_t = torch.from_numpy(
        np.asarray(ctx.get("prev_action", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32)
    ).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[float(ctx.get("prev_reward", 0.0))]]], dtype=torch.float32, device=device)
    hidden = ctx.get("hidden", None)
    with torch.no_grad():
        q, hidden_out, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        action = int(torch.argmax(q[:, -1], dim=-1).item())
    next_ctx = dict(ctx)
    next_ctx["hidden"] = hidden_out.detach()
    next_ctx["device"] = str(device)
    return action, next_ctx


def episode_priority(
    last_info: Dict[str, object],
    ep_reward: float,
    diag_metrics: Dict[str, float],
    cfg: TrainConfig,
) -> float:
    coverage = finite_float(last_info.get("coverage", 0.0), 0.0)
    cov_50 = finite_float(diag_metrics.get("coverage_at_step_50", coverage), coverage)
    cov_100 = finite_float(diag_metrics.get("coverage_at_step_100", coverage), coverage)
    cov_150 = finite_float(diag_metrics.get("coverage_at_step_150", coverage), coverage)
    seen = finite_float(diag_metrics.get("seen_coverage", 0.0), 0.0)
    seen_100 = finite_float(diag_metrics.get("seen_at_step_100", seen), seen)
    seen_150 = finite_float(diag_metrics.get("seen_at_step_150", seen), seen)
    center_seen = finite_float(diag_metrics.get("center_seen_coverage", 0.0), 0.0)
    if np.isnan(cov_50):
        cov_50 = coverage
    if np.isnan(cov_100):
        cov_100 = coverage
    if np.isnan(cov_150):
        cov_150 = coverage
    new_tile_rate = finite_float(diag_metrics.get("new_tile_rate", 0.0), 0.0)
    revisit_rate = finite_float(diag_metrics.get("revisit_rate", 0.0), 0.0)
    wall_hit_rate = finite_float(diag_metrics.get("wall_hit_rate", 0.0), 0.0)
    hazard_contact_rate = finite_float(diag_metrics.get("hazard_contact_rate", 0.0), 0.0)
    stagnation_step_rate = finite_float(diag_metrics.get("stagnation_step_rate", 0.0), 0.0)
    coverage_gain_25 = finite_float(diag_metrics.get("coverage_gain_25", 0.0), 0.0)
    coverage_gain_per_energy = finite_float(diag_metrics.get("coverage_gain_per_energy", 0.0), 0.0)
    new_tiles_per_energy = finite_float(diag_metrics.get("new_tiles_per_energy", 0.0), 0.0)
    turn_oscillation_rate = finite_float(diag_metrics.get("turn_oscillation_rate", 0.0), 0.0)
    forward_after_turn_success_rate = finite_float(diag_metrics.get("forward_after_turn_success_rate", 0.0), 0.0)
    priority = (
        1.0
        + 1.80 * seen
        + 0.55 * seen_100
        + 0.75 * seen_150
        + 0.25 * center_seen
        + 2.0 * coverage
        + 0.40 * cov_50
        + 0.60 * cov_100
        + 0.85 * cov_150
        + 0.90 * new_tile_rate
        + 0.80 * coverage_gain_25
        + 0.90 * coverage_gain_per_energy
        + 0.55 * new_tiles_per_energy
        + 0.18 * forward_after_turn_success_rate
        - 0.20 * revisit_rate
        - float(cfg.replay_episode_wall_penalty) * wall_hit_rate
        - float(cfg.replay_episode_hazard_penalty) * hazard_contact_rate
        - 0.22 * stagnation_step_rate
        - 0.10 * turn_oscillation_rate
    )
    return max(finite_float(priority, 1.0), 1e-3)


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "sensory_grid_env_v5",
        "model_family": "patch_fusion_gru_dueling_double_dqn_aux",
        "default_switches": asdict(build_default_switches()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe-route memory recurrent Double DQN")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--save_dir", type=str, default="runs/safe_route_memory_dqn")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--last_checkpoint_every", type=int, default=TrainConfig.last_checkpoint_every_episodes)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_eval_episodes", type=int, default=10)
    parser.add_argument("--disable_structured_exploration", action="store_true")
    parser.add_argument("--enable_state_adaptive_exploration", action="store_true")
    parser.add_argument("--disable_window_priority_sampling", action="store_true")
    parser.add_argument("--disable_exploration_reward_shaping", action="store_true")
    parser.add_argument("--disable_observation_novelty_bonus", action="store_true")
    parser.add_argument("--frontier_score_scale", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        last_checkpoint_every_episodes=max(0, args.last_checkpoint_every),
        eval_episodes=args.eval_episodes,
        holdout_eval_episodes=args.holdout_eval_episodes,
        use_state_adaptive_exploration=args.enable_state_adaptive_exploration,
        use_window_priority_sampling=not args.disable_window_priority_sampling,
        use_exploration_reward_shaping=not args.disable_exploration_reward_shaping,
        use_observation_novelty_bonus=not args.disable_observation_novelty_bonus,
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
    resume_payload: Dict[str, object] | None = None
    start_episode = 1
    global_step = 0

    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        loaded_payload = torch.load(resume_path, map_location=device)
        if not isinstance(loaded_payload, dict) or "model_state_dict" not in loaded_payload:
            raise ValueError(f"Invalid resume checkpoint: {resume_path}")
        model_arch = loaded_payload.get("model_arch")
        if model_arch not in (None, "patch_fusion_gru_dueling_double_dqn_safe_route_memory"):
            raise ValueError(f"Checkpoint architecture is incompatible with the safe-route memory model: {model_arch}")
        online_net.load_state_dict(loaded_payload["model_state_dict"])
        target_state = loaded_payload.get("target_model_state_dict", loaded_payload["model_state_dict"])
        target_net.load_state_dict(target_state)
        optimizer_state = loaded_payload.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        python_random_state = loaded_payload.get("python_random_state")
        numpy_random_state = loaded_payload.get("numpy_random_state")
        torch_random_state = loaded_payload.get("torch_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)
        if numpy_random_state is not None:
            np.random.set_state(numpy_random_state)
        if isinstance(torch_random_state, torch.Tensor):
            torch.set_rng_state(torch_random_state.cpu())
        resume_payload = loaded_payload
        start_episode = int(loaded_payload.get("episode", 0)) + 1
        global_step = int(loaded_payload.get("global_step", 0))
        if start_episode > cfg.episodes:
            raise ValueError(
                f"Checkpoint is at episode {start_episode - 1}, but --episodes is only {cfg.episodes}."
            )

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
            "length",
            *episode_metric_names,
            "mean_loss",
            "mean_q_loss",
            *eval_fieldnames,
            *holdout_eval_fieldnames,
            "eval_soft_score",
        ])

    status_path = save_dir / "run_status.log"
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("# Persistent progress log. The final phase identifies where an external stop occurred.\n")
    if resume_payload is not None:
        append_run_status(
            status_path,
            start_episode - 1,
            global_step,
            "resume_loaded",
            checkpoint=Path(args.resume_checkpoint).resolve(),
            replay="reset",
        )
        print(
            f"Resuming model from episode {start_episode - 1}; replay is rebuilt from new experience.",
            flush=True,
        )
    else:
        append_run_status(status_path, 0, global_step, "run_started", device=device)

    best_selection = (-float("inf"),) * 6
    best_eval_reward = -float("inf")
    best_eval_coverage = -float("inf")
    best_holdout_eval_coverage = -float("inf")
    best_eval_survival = -float("inf")
    best_eval_soft = -float("inf")
    if resume_payload is not None:
        resume_eval_metrics = resume_payload.get("eval_metrics", {})
        resume_holdout_metrics = resume_payload.get("holdout_eval_metrics", {})
        if isinstance(resume_eval_metrics, dict) and resume_eval_metrics:
            best_selection = selection_key(resume_eval_metrics, cfg)
            best_eval_reward = finite_float(resume_eval_metrics.get("eval_reward_mean"), -float("inf"))
            best_eval_coverage = finite_float(resume_eval_metrics.get("eval_coverage_mean"), -float("inf"))
            best_eval_survival = finite_float(resume_eval_metrics.get("eval_survival_rate"), -float("inf"))
            best_eval_soft = composite_eval_score(resume_eval_metrics, cfg)
        if isinstance(resume_holdout_metrics, dict) and resume_holdout_metrics:
            best_holdout_eval_coverage = finite_float(
                resume_holdout_metrics.get("holdout_eval_coverage_mean"), -float("inf")
            )

    for episode in range(start_episode, cfg.episodes + 1):
        append_run_status(status_path, episode, global_step, "rollout_started")
        obs, reset_info = env.reset(seed=cfg.seed + episode)
        state = obs_to_exploration_state(obs, env.config, switches)
        done = False
        ep_reward = 0.0
        losses: List[float] = []
        q_losses: List[float] = []
        last_info = None
        hidden = None
        prev_action = np.zeros(N_ACTIONS, dtype=np.float32)
        prev_reward = 0.0
        episode_transitions: List[Dict[str, object]] = []
        episode_diag = init_episode_diagnostics(env, cfg, float(reset_info.get("coverage", 0.0)))
        current_coverage = float(reset_info.get("coverage", 0.0))
        novelty_counts: Dict[bytes, int] = {}

        while not done:
            obs_novelty = observation_novelty_bonus(obs, novelty_counts)
            episode_diag["obs_novelty_total"] = float(episode_diag["obs_novelty_total"]) + obs_novelty
            epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
            planner_scores = planner_action_scores(env, episode_diag, cfg) if cfg.use_structured_exploration else None
            planner_best_action = int(np.argmax(planner_scores)) if planner_scores is not None and np.isfinite(planner_scores).any() else None
            seen_before = float(episode_diag.get("seen_fraction", 0.0))
            center_seen_before = float(episode_diag.get("center_seen_fraction", 0.0))
            post_seen_phase = float(seen_before >= float(cfg.exploration_target_seen_fraction))
            action, hidden = choose_action(
                online_net,
                obs,
                state,
                prev_action,
                prev_reward,
                hidden,
                device,
                epsilon,
                cfg,
                planner_scores,
            )
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            next_state = obs_to_exploration_state(next_obs, env.config, switches)
            done = terminated or truncated
            seen_delta, center_seen_delta = update_episode_diagnostics(episode_diag, env, action, info)
            next_coverage = float(info.get("coverage", current_coverage))
            coverage_delta = max(0.0, next_coverage - current_coverage)
            reward_terms = info.get("reward_terms", {})
            wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
            revisit = bool(action == ACTION_FORWARD and not wall_hit and coverage_delta <= 1e-9)
            energy_spent = transition_energy_spent(info)
            stagnating = int(episode_diag.get("steps_since_new_tile", 0)) >= int(cfg.exploration_stagnation_trigger_steps)
            planner_alignment = 1.0 if planner_best_action is not None and action == planner_best_action else 0.0

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
                "coverage_before": float(current_coverage),
                "next_coverage": float(next_coverage),
                "coverage_delta": float(coverage_delta),
                "seen_before": float(seen_before),
                "center_seen_before": float(center_seen_before),
                "seen_delta": float(seen_delta),
                "center_seen_delta": float(center_seen_delta),
                "energy_spent": float(energy_spent),
                "obs_novelty": float(obs_novelty),
                "future_coverage_gain": 0.0,
                "revisit": float(revisit),
                "wall_hit": float(wall_hit),
                "post_seen_phase": post_seen_phase,
                "stagnating": float(stagnating),
                "planner_alignment": float(planner_alignment),
                "step_index": float(info.get("steps", 0)),
                "sample_weight": 1.0,
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
        horizon_steps = max(1, int(cfg.future_coverage_horizon_steps))
        for idx, transition in enumerate(episode_transitions):
            end = min(len(episode_transitions), idx + horizon_steps)
            future_coverage = max(float(item["next_coverage"]) for item in episode_transitions[idx:end])
            future_gain = max(0.0, future_coverage - float(transition["coverage_before"]))
            transition["future_coverage_gain"] = float(future_gain)
            if cfg.use_window_priority_sampling:
                transition["sample_weight"] = float(
                    transition_sample_weight(
                        float(transition["coverage_delta"]),
                        float(transition["seen_delta"]),
                        float(transition["center_seen_delta"]),
                        float(transition["energy_spent"]),
                        future_gain,
                        float(transition["obs_novelty"]),
                        bool(transition["revisit"]),
                        bool(transition["wall_hit"]),
                        bool(transition["stagnating"]),
                        float(transition["planner_alignment"]),
                        cfg,
                    )
                )

        episode_diag_metrics = finalize_episode_diagnostics(episode_diag)
        replay.add_episode(episode_transitions, priority=episode_priority(last_info, ep_reward, episode_diag_metrics, cfg))
        append_run_status(
            status_path,
            episode,
            global_step,
            "rollout_complete",
            transitions=len(episode_transitions),
            replay_episodes=len(replay),
        )

        if len(replay) >= cfg.train_after_episodes:
            for update_index in range(cfg.train_updates_per_episode):
                append_run_status(
                    status_path,
                    episode,
                    global_step,
                    "optimizer_update_started",
                    update=f"{update_index + 1}/{cfg.train_updates_per_episode}",
                )
                total_loss, q_loss = train_step(online_net, target_net, optimizer, replay, device, cfg)
                losses.append(total_loss)
                q_losses.append(q_loss)
                append_run_status(
                    status_path,
                    episode,
                    global_step,
                    "optimizer_update_complete",
                    update=f"{update_index + 1}/{cfg.train_updates_per_episode}",
                    loss=f"{total_loss:.6f}",
                )

        soft_update(target_net, online_net, cfg.target_soft_tau)
        if episode % cfg.target_hard_sync_every_episodes == 0:
            target_net.load_state_dict(online_net.state_dict())

        do_eval = episode == 1 or episode % cfg.eval_every == 0 or episode == cfg.episodes
        if do_eval:
            append_run_status(status_path, episode, global_step, "evaluation_started")
            eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                online_net,
                device,
                cfg,
                episodes=cfg.eval_episodes,
                seed_start=cfg.eval_seed_start,
                metric_prefix="eval",
            )
            holdout_eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                online_net,
                device,
                cfg,
                episodes=cfg.holdout_eval_episodes,
                seed_start=cfg.holdout_eval_seed_start,
                metric_prefix="holdout_eval",
            )
            append_run_status(status_path, episode, global_step, "evaluation_complete")
        else:
            eval_metrics = empty_eval_metrics("eval")
            holdout_eval_metrics = empty_eval_metrics("holdout_eval")

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_q_loss = float(np.mean(q_losses)) if q_losses else 0.0
        epsilon = adaptive_exploration_epsilon(global_step, obs, episode_diag, cfg)
        eval_score = composite_eval_score(eval_metrics, cfg) if do_eval else float("nan")
        print(
            f"episode={episode:04d} step={global_step:06d} eps={epsilon:.3f} "
            f"reward={ep_reward:+.3f} coverage={last_info['coverage']:.3f} "
            f"seen={episode_diag_metrics['seen_coverage']:.3f} center_seen={episode_diag_metrics['center_seen_coverage']:.3f} "
            f"len={last_info['steps']} "
            f"new={episode_diag_metrics['new_tile_rate']:.3f} revisit={episode_diag_metrics['revisit_rate']:.3f} "
            f"wall={episode_diag_metrics['wall_hit_rate']:.3f} hazard={episode_diag_metrics['hazard_contact_rate']:.3f} "
            f"stagnation={episode_diag_metrics['stagnation_step_rate']:.3f} c25={episode_diag_metrics['coverage_gain_25']:.3f} "
            f"nov={episode_diag_metrics['obs_novelty_mean']:.3f} "
            f"loss={mean_loss:.4f} q={mean_q_loss:.4f} "
            f"eval_surv={eval_metrics['eval_survival_rate']:.3f} "
            f"eval_cov={eval_metrics['eval_coverage_mean']:.3f} "
            f"eval_seen={eval_metrics['eval_seen_coverage_mean']:.3f} "
            f"eval_reward={eval_metrics['eval_reward_mean']:.3f} "
            f"holdout_cov={holdout_eval_metrics['holdout_eval_coverage_mean']:.3f} "
            f"holdout_seen={holdout_eval_metrics['holdout_eval_seen_coverage_mean']:.3f} "
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
                last_info["steps"],
                *[episode_diag_metrics[name] for name in episode_metric_names],
                mean_loss,
                mean_q_loss,
                *[eval_metrics[name] for name in eval_fieldnames],
                *[holdout_eval_metrics[name] for name in holdout_eval_fieldnames],
                eval_score,
            ])
        append_run_status(status_path, episode, global_step, "episode_complete")

        if (
            cfg.last_checkpoint_every_episodes > 0
            and (episode % cfg.last_checkpoint_every_episodes == 0 or episode == cfg.episodes)
        ):
            append_run_status(status_path, episode, global_step, "recovery_checkpoint_started")
            atomic_torch_save(
                training_state_payload(
                    online_net,
                    target_net,
                    optimizer,
                    input_dim,
                    switches,
                    env,
                    cfg,
                    episode,
                    global_step,
                    eval_metrics,
                    holdout_eval_metrics,
                ),
                save_dir / "last_train_state.pt",
            )
            append_run_status(status_path, episode, global_step, "recovery_checkpoint_complete")

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

            if holdout_eval_metrics["holdout_eval_coverage_mean"] > best_holdout_eval_coverage:
                best_holdout_eval_coverage = holdout_eval_metrics["holdout_eval_coverage_mean"]
                torch.save(payload, save_dir / "best_holdout_coverage_model.pt")

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

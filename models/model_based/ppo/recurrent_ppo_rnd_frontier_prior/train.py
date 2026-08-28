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

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    EnvConfig,
    N_ACTIONS,
    ObservationSwitches,
    SensoryGridEnv,
)
from models.shared.ppo_rnd_interface import (
    ModelConfig,
    RecurrentPPORNDNetwork,
    build_model_from_checkpoint as base_build_model_from_checkpoint,
    choose_device,
    init_runtime_context,
    obs_to_state,
    onehot_action,
    reset_runtime_context,
    update_runtime_context_after_env_step,
)
from models.model_based.dqn.recurrent_patch_fusion_initial_heuristics.train import (
    HAZARD_OBJECT_IDS,
    build_default_switches,
    build_training_env as dqn_build_training_env,
    composite_eval_score,
    empty_eval_metrics,
    episode_diagnostic_metric_names,
    eval_metric_names,
    finalize_episode_diagnostics,
    init_episode_diagnostics,
    nanmean,
    update_episode_diagnostics,
)


TRAINER_GUI_INTERFACE_VERSION = "recurrent-ppo-rnd-frontier-prior"
TRAINER_DISPLAY_NAME = "Recurrent PPO/RND with frontier prior"


@dataclass
class TrainConfig:
    updates: int = 300
    seed: int = 7
    save_dir: str = "runs/recurrent_ppo_rnd_frontier_prior"
    device: str = "auto"

    n_envs: int = 8
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_envs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma_ext: float = 0.99
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.15
    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    intrinsic_value_loss_coef: float = 0.5
    rnd_loss_coef: float = 1.0
    intrinsic_advantage_coef: float = 1.0
    max_gradient_norm: float = 0.5

    intrinsic_reward_scale: float = 0.20
    intrinsic_reward_clip: float = 5.0
    rnd_turn_intrinsic_scale: float = 0.10
    rnd_wall_intrinsic_scale: float = 0.0
    rnd_new_tile_intrinsic_scale: float = 1.15
    rnd_revisit_intrinsic_scale: float = 0.35
    rnd_turn_training_scale: float = 0.25
    rnd_wall_training_scale: float = 0.10
    rnd_new_tile_training_scale: float = 0.85
    rnd_revisit_training_scale: float = 1.80
    rnd_min_training_weight: float = 0.05

    policy_prior_score_scale: float = 0.18
    policy_prior_unvisited_weight: float = 1.00
    policy_prior_visited_penalty: float = 0.60
    policy_prior_visible_hazard_penalty: float = 1.10
    policy_prior_known_hazard_penalty: float = 0.85
    policy_prior_forward_unvisited_bonus: float = 1.20
    policy_prior_forward_revisit_penalty: float = 0.88
    policy_prior_front_visible_hazard_penalty: float = 0.35
    policy_prior_front_known_hazard_penalty: float = 0.70
    policy_prior_repeat_turn_penalty: float = 0.72
    policy_prior_post_turn_forward_bonus: float = 1.25
    policy_prior_turn_penalty: float = 0.10

    turn_balance_loss_coef: float = 0.03
    turn_floor_loss_coef: float = 0.02
    min_turn_probability: float = 0.06

    eval_every: int = 10
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
    rnd_hidden_dim: int = 256
    rnd_output_dim: int = 128

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

    training_survival_bonus: float = 0.60


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = max(m2 / total_count, 1e-12)
        self.count = total_count


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
        rnd_hidden_dim=cfg.rnd_hidden_dim,
        rnd_output_dim=cfg.rnd_output_dim,
    )


def build_training_env(cfg: TrainConfig | None = None) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    return dqn_build_training_env(cfg)


def build_network(env: SensoryGridEnv, switches: ObservationSwitches, cfg: TrainConfig) -> RecurrentPPORNDNetwork:
    return RecurrentPPORNDNetwork(
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


def copy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in state.items()}


def stack_state_dicts(states: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = states[0].keys()
    return {k: np.stack([state[k] for state in states], axis=0).astype(np.float32) for k in keys}


def transpose_state_rollout(states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.swapaxes(v, 0, 1).astype(np.float32) for k, v in states.items()}


def states_to_torch_sequence(states: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in states.items()}


def selection_key(eval_metrics: Dict[str, float], cfg: TrainConfig) -> Tuple[float, float, float, float, float, float]:
    return (
        composite_eval_score(eval_metrics, cfg),
        float(eval_metrics.get("eval_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_150_mean", float("-inf"))),
        float(eval_metrics.get("eval_coverage_at_step_100_mean", float("-inf"))),
        float(eval_metrics.get("eval_reward_mean", float("-inf"))),
        -float(eval_metrics.get("eval_wall_hit_rate_mean", float("inf"))),
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros((rewards.shape[1],), dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        if t == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def vision_ids_from_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    return np.argmax(np.asarray(state["vision"], dtype=np.float32), axis=0).astype(np.int64)


def extract_front_cell_features_from_state(state: Dict[str, np.ndarray]) -> Dict[str, bool]:
    vision_ids = vision_ids_from_state(state)
    visited_patch = np.asarray(state["visited_patch"][0], dtype=np.float32)
    hazard_patch = np.asarray(state["hazard_patch"][0], dtype=np.float32)
    centre = vision_ids.shape[0] // 2
    front_r = max(centre - 1, 0)
    front_c = centre
    front_obj = int(vision_ids[front_r, front_c])
    front_visited = float(visited_patch[front_r, front_c]) > 0.5
    front_known_hazard = float(hazard_patch[front_r, front_c]) >= 0.25
    return {
        "front_unvisited": not front_visited,
        "front_visible_hazard": front_obj in HAZARD_OBJECT_IDS,
        "front_known_hazard": front_known_hazard,
    }


def directional_novelty_scores_from_state(state: Dict[str, np.ndarray], cfg: TrainConfig) -> np.ndarray:
    vision_ids = vision_ids_from_state(state)
    visited_patch = np.asarray(state["visited_patch"][0], dtype=np.float32)
    hazard_patch = np.asarray(state["hazard_patch"][0], dtype=np.float32)
    centre = vision_ids.shape[0] // 2

    def cell_score(r: int, c: int) -> float:
        if not (0 <= r < vision_ids.shape[0] and 0 <= c < vision_ids.shape[1]):
            return 0.0
        obj = int(vision_ids[r, c])
        visited = float(visited_patch[r, c])
        hazard_mem = float(hazard_patch[r, c])
        score = (
            float(cfg.policy_prior_unvisited_weight)
            if visited < 0.5
            else -float(cfg.policy_prior_visited_penalty)
        )
        if obj in HAZARD_OBJECT_IDS:
            score -= float(cfg.policy_prior_visible_hazard_penalty)
        score -= float(cfg.policy_prior_known_hazard_penalty) * hazard_mem
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
        if action in (ACTION_LEFT, ACTION_RIGHT):
            scores[action] -= float(cfg.policy_prior_turn_penalty)
    return scores


def policy_prior_logit_bias_from_state(
    state: Dict[str, np.ndarray],
    prev_action: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
    weights = np.ones((N_ACTIONS,), dtype=np.float64)
    features = extract_front_cell_features_from_state(state)
    novelty_scores = directional_novelty_scores_from_state(state, cfg)
    novelty_scores -= float(np.max(novelty_scores))
    weights *= np.exp(float(cfg.policy_prior_score_scale) * novelty_scores)

    if features["front_unvisited"]:
        weights[ACTION_FORWARD] *= float(cfg.policy_prior_forward_unvisited_bonus)
    else:
        weights[ACTION_FORWARD] *= float(cfg.policy_prior_forward_revisit_penalty)

    if features["front_visible_hazard"]:
        weights[ACTION_FORWARD] *= float(cfg.policy_prior_front_visible_hazard_penalty)
    elif features["front_known_hazard"]:
        weights[ACTION_FORWARD] *= float(cfg.policy_prior_front_known_hazard_penalty)

    prev_action_arr = np.asarray(prev_action, dtype=np.float32)
    prev_action_idx = int(np.argmax(prev_action_arr)) if float(prev_action_arr.sum()) > 0.0 else None
    if prev_action_idx in (ACTION_LEFT, ACTION_RIGHT):
        weights[ACTION_LEFT] *= float(cfg.policy_prior_repeat_turn_penalty)
        weights[ACTION_RIGHT] *= float(cfg.policy_prior_repeat_turn_penalty)
        weights[ACTION_FORWARD] *= float(cfg.policy_prior_post_turn_forward_bonus)

    weights = np.clip(weights, 1e-6, None)
    logit_bias = np.log(weights)
    logit_bias -= float(np.mean(logit_bias))
    return logit_bias.astype(np.float32)


def batch_policy_prior_logit_bias(
    states: List[Dict[str, np.ndarray]],
    prev_actions: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
    return np.stack(
        [
            policy_prior_logit_bias_from_state(state, prev_actions[idx], cfg)
            for idx, state in enumerate(states)
        ],
        axis=0,
    ).astype(np.float32)


def intrinsic_reward_scale(action: int, info: Dict[str, Any], cfg: TrainConfig) -> float:
    reward_terms = info.get("reward_terms", {})
    wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
    new_tile = float(reward_terms.get("explore_reward", 0.0)) > 0.0
    revisit = action == ACTION_FORWARD and not wall_hit and not new_tile
    scale = float(cfg.intrinsic_reward_scale)
    if action != ACTION_FORWARD:
        scale *= float(cfg.rnd_turn_intrinsic_scale)
    if wall_hit:
        scale *= float(cfg.rnd_wall_intrinsic_scale)
    if new_tile:
        scale *= float(cfg.rnd_new_tile_intrinsic_scale)
    if revisit:
        scale *= float(cfg.rnd_revisit_intrinsic_scale)
    return float(scale)


def rnd_training_weight(action: int, info: Dict[str, Any], cfg: TrainConfig) -> float:
    reward_terms = info.get("reward_terms", {})
    wall_hit = float(reward_terms.get("wall_penalty", 0.0)) < 0.0
    new_tile = float(reward_terms.get("explore_reward", 0.0)) > 0.0
    revisit = action == ACTION_FORWARD and not wall_hit and not new_tile
    weight = 1.0
    if action != ACTION_FORWARD:
        weight *= float(cfg.rnd_turn_training_scale)
    if wall_hit:
        weight *= float(cfg.rnd_wall_training_scale)
    if new_tile:
        weight *= float(cfg.rnd_new_tile_training_scale)
    if revisit:
        weight *= float(cfg.rnd_revisit_training_scale)
    return float(max(weight, cfg.rnd_min_training_weight))


def aggregate_completed_episode_stats(
    rewards: List[float],
    infos: List[Dict[str, Any]],
    diagnostics: List[Dict[str, float]],
) -> Dict[str, float]:
    stats: Dict[str, float] = {
        "train_reward_mean": float("nan"),
        "train_coverage_mean": float("nan"),
        "train_final_health_mean": float("nan"),
        "train_length_mean": float("nan"),
    }
    if rewards:
        stats["train_reward_mean"] = float(np.mean(rewards))
        stats["train_coverage_mean"] = float(np.mean([float(info["coverage"]) for info in infos]))
        stats["train_final_health_mean"] = float(np.mean([float(info["health"]) for info in infos]))
        stats["train_length_mean"] = float(np.mean([float(info["steps"]) for info in infos]))
    for name in episode_diagnostic_metric_names():
        values = [item[name] for item in diagnostics] if diagnostics else []
        stats[f"train_{name}_mean"] = nanmean(values)
    return stats


def rollout_batch_to_training_format(batch: Dict[str, Any], cfg: TrainConfig) -> Dict[str, np.ndarray]:
    advantages = batch["ext_advantages"] + float(cfg.intrinsic_advantage_coef) * batch["int_advantages"]
    advantages = advantages.astype(np.float32)
    adv_mean = float(np.mean(advantages))
    adv_std = float(np.std(advantages))
    advantages = (advantages - adv_mean) / max(adv_std, 1e-6)
    return {
        "states": transpose_state_rollout(batch["states"]),
        "next_states": transpose_state_rollout(batch["next_states"]),
        "prev_actions": np.swapaxes(batch["prev_actions"], 0, 1).astype(np.float32),
        "prev_rewards": np.swapaxes(batch["prev_rewards"], 0, 1).astype(np.float32),
        "actions": np.swapaxes(batch["actions"], 0, 1).astype(np.int64),
        "old_logprobs": np.swapaxes(batch["logprobs"], 0, 1).astype(np.float32),
        "ext_returns": np.swapaxes(batch["ext_returns"], 0, 1).astype(np.float32),
        "int_returns": np.swapaxes(batch["int_returns"], 0, 1).astype(np.float32),
        "advantages": np.swapaxes(advantages, 0, 1).astype(np.float32),
        "episode_starts": np.swapaxes(batch["episode_starts"], 0, 1).astype(np.float32),
        "rnd_weights": np.swapaxes(batch["rnd_weights"], 0, 1).astype(np.float32),
        "action_prior_bias": np.swapaxes(batch["action_prior_bias"], 0, 1).astype(np.float32),
        "hidden_start": batch["hidden_start"].astype(np.float32),
    }


def collect_rollout(
    envs: List[SensoryGridEnv],
    switches: ObservationSwitches,
    net: RecurrentPPORNDNetwork,
    device: torch.device,
    cfg: TrainConfig,
    current_obs: List[Dict[str, Any]],
    current_states: List[Dict[str, np.ndarray]],
    prev_actions: np.ndarray,
    prev_rewards: np.ndarray,
    hidden: torch.Tensor,
    episode_starts: np.ndarray,
    diag_trackers: List[Dict[str, object]],
    episode_returns: np.ndarray,
    intrinsic_rms: RunningMeanStd,
    reset_seed_cursor: int,
    episodes_seen: int,
    global_step: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, torch.Tensor, np.ndarray, List[Dict[str, object]], np.ndarray, int, int, int]:
    rollout_steps = int(cfg.rollout_steps)
    n_envs = int(cfg.n_envs)
    state_template = current_states[0]
    rollout_states = {
        k: np.zeros((rollout_steps, n_envs, *v.shape), dtype=np.float32) for k, v in state_template.items()
    }
    rollout_next_states = {
        k: np.zeros((rollout_steps, n_envs, *v.shape), dtype=np.float32) for k, v in state_template.items()
    }
    rollout_prev_actions = np.zeros((rollout_steps, n_envs, N_ACTIONS), dtype=np.float32)
    rollout_prev_rewards = np.zeros((rollout_steps, n_envs, 1), dtype=np.float32)
    rollout_actions = np.zeros((rollout_steps, n_envs), dtype=np.int64)
    rollout_logprobs = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_ext_values = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_int_values = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_ext_rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_int_rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_episode_starts = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_rnd_weights = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    rollout_action_prior_bias = np.zeros((rollout_steps, n_envs, N_ACTIONS), dtype=np.float32)

    hidden_start = hidden.detach().cpu().numpy()
    completed_rewards: List[float] = []
    completed_infos: List[Dict[str, Any]] = []
    completed_diags: List[Dict[str, float]] = []

    for t in range(rollout_steps):
        current_state_batch_np = stack_state_dicts(current_states)
        for key in rollout_states:
            rollout_states[key][t] = current_state_batch_np[key]
        rollout_prev_actions[t] = prev_actions
        rollout_prev_rewards[t, :, 0] = prev_rewards
        rollout_episode_starts[t] = episode_starts.astype(np.float32)

        state_batch = {k: torch.from_numpy(v).unsqueeze(1).to(device) for k, v in current_state_batch_np.items()}
        action_prior_bias_np = batch_policy_prior_logit_bias(current_states, prev_actions, cfg)
        rollout_action_prior_bias[t] = action_prior_bias_np
        action_prior_bias_t = torch.from_numpy(action_prior_bias_np).to(device)
        prev_action_t = torch.from_numpy(prev_actions).unsqueeze(1).to(device)
        prev_reward_t = torch.from_numpy(prev_rewards).view(n_envs, 1, 1).to(device)
        reset_mask_t = torch.from_numpy(episode_starts.astype(np.float32)).view(n_envs, 1).to(device)

        with torch.no_grad():
            logits, ext_value, int_value, hidden = net.forward_sequence(
                state_batch,
                prev_action_t,
                prev_reward_t,
                hidden,
                reset_mask=reset_mask_t,
            )
            dist = torch.distributions.Categorical(logits=logits[:, 0] + action_prior_bias_t)
            actions_t = dist.sample()
            logprobs_t = dist.log_prob(actions_t)

        actions_np = actions_t.cpu().numpy().astype(np.int64)
        rollout_actions[t] = actions_np
        rollout_logprobs[t] = logprobs_t.cpu().numpy().astype(np.float32)
        rollout_ext_values[t] = ext_value[:, 0].detach().cpu().numpy().astype(np.float32)
        rollout_int_values[t] = int_value[:, 0].detach().cpu().numpy().astype(np.float32)

        next_obs_list: List[Dict[str, Any]] = []
        next_state_list: List[Dict[str, np.ndarray]] = []
        next_prev_actions = np.zeros_like(prev_actions)
        next_prev_rewards = np.zeros_like(prev_rewards)
        next_episode_starts = np.zeros_like(episode_starts)
        intrinsic_scales = np.zeros((n_envs,), dtype=np.float32)

        for env_idx, env in enumerate(envs):
            action = int(actions_np[env_idx])
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            next_state = obs_to_state(next_obs, env.config, switches)
            done = terminated or truncated
            update_episode_diagnostics(diag_trackers[env_idx], action, info)
            episode_returns[env_idx] += float(reward)

            rollout_ext_rewards[t, env_idx] = float(reward)
            rollout_dones[t, env_idx] = float(done)
            rollout_rnd_weights[t, env_idx] = rnd_training_weight(action, info, cfg)
            intrinsic_scales[env_idx] = intrinsic_reward_scale(action, info, cfg)

            if done:
                completed_rewards.append(float(episode_returns[env_idx]))
                completed_infos.append(info)
                completed_diags.append(finalize_episode_diagnostics(diag_trackers[env_idx]))
                episode_returns[env_idx] = 0.0
                episodes_seen += 1
                reset_seed_cursor += 1
                next_obs, reset_info = env.reset(seed=reset_seed_cursor)
                next_state = obs_to_state(next_obs, env.config, switches)
                diag_trackers[env_idx] = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))
                next_prev_actions[env_idx] = 0.0
                next_prev_rewards[env_idx] = 0.0
                next_episode_starts[env_idx] = True
            else:
                next_prev_actions[env_idx] = onehot_action(action)
                next_prev_rewards[env_idx] = float(reward)
                next_episode_starts[env_idx] = False

            next_obs_list.append(next_obs)
            next_state_list.append(next_state)

        next_state_batch_np = stack_state_dicts(next_state_list)
        for key in rollout_next_states:
            rollout_next_states[key][t] = next_state_batch_np[key]

        next_state_batch = {k: torch.from_numpy(v).unsqueeze(1).to(device) for k, v in next_state_batch_np.items()}
        with torch.no_grad():
            rnd_error, _, _ = net.rnd_prediction_error(next_state_batch)
        intrinsic_error = rnd_error[:, 0].cpu().numpy().astype(np.float32)
        intrinsic_rms.update(intrinsic_error)
        normalized_intrinsic = intrinsic_error / float(np.sqrt(intrinsic_rms.var + 1e-8))
        normalized_intrinsic = np.clip(normalized_intrinsic, 0.0, float(cfg.intrinsic_reward_clip))
        rollout_int_rewards[t] = normalized_intrinsic * intrinsic_scales

        current_obs = next_obs_list
        current_states = next_state_list
        prev_actions = next_prev_actions
        prev_rewards = next_prev_rewards
        episode_starts = next_episode_starts
        global_step += n_envs

    last_state_batch = stack_state_dicts(current_states)
    last_state_t = {k: torch.from_numpy(v).unsqueeze(1).to(device) for k, v in last_state_batch.items()}
    last_prev_action_t = torch.from_numpy(prev_actions).unsqueeze(1).to(device)
    last_prev_reward_t = torch.from_numpy(prev_rewards).view(n_envs, 1, 1).to(device)
    last_reset_mask_t = torch.from_numpy(episode_starts.astype(np.float32)).view(n_envs, 1).to(device)
    with torch.no_grad():
        _, last_ext_value, last_int_value, _ = net.forward_sequence(
            last_state_t,
            last_prev_action_t,
            last_prev_reward_t,
            hidden,
            reset_mask=last_reset_mask_t,
        )

    ext_advantages, ext_returns = compute_gae(
        rollout_ext_rewards,
        rollout_ext_values,
        rollout_dones,
        last_ext_value[:, 0].detach().cpu().numpy().astype(np.float32),
        cfg.gamma_ext,
        cfg.gae_lambda,
    )
    int_advantages, int_returns = compute_gae(
        rollout_int_rewards,
        rollout_int_values,
        rollout_dones,
        last_int_value[:, 0].detach().cpu().numpy().astype(np.float32),
        cfg.gamma_int,
        cfg.gae_lambda,
    )

    batch = {
        "states": rollout_states,
        "next_states": rollout_next_states,
        "prev_actions": rollout_prev_actions,
        "prev_rewards": rollout_prev_rewards,
        "actions": rollout_actions,
        "logprobs": rollout_logprobs,
        "ext_values": rollout_ext_values,
        "int_values": rollout_int_values,
        "ext_rewards": rollout_ext_rewards,
        "int_rewards": rollout_int_rewards,
        "dones": rollout_dones,
        "episode_starts": rollout_episode_starts,
        "rnd_weights": rollout_rnd_weights,
        "action_prior_bias": rollout_action_prior_bias,
        "hidden_start": hidden_start,
        "ext_advantages": ext_advantages,
        "ext_returns": ext_returns,
        "int_advantages": int_advantages,
        "int_returns": int_returns,
        "rollout_ext_reward_mean": float(np.mean(rollout_ext_rewards)),
        "rollout_int_reward_mean": float(np.mean(rollout_int_rewards)),
        "completed_episode_stats": aggregate_completed_episode_stats(completed_rewards, completed_infos, completed_diags),
    }
    return (
        batch,
        current_obs,
        current_states,
        prev_actions,
        prev_rewards,
        hidden,
        episode_starts,
        diag_trackers,
        episode_returns,
        reset_seed_cursor,
        episodes_seen,
        global_step,
    )


def ppo_update(
    net: RecurrentPPORNDNetwork,
    optimizer: optim.Optimizer,
    batch: Dict[str, np.ndarray],
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[float, float, float, float, float, float, float, float]:
    states_t = states_to_torch_sequence(batch["states"], device)
    next_states_t = states_to_torch_sequence(batch["next_states"], device)
    prev_actions_t = torch.from_numpy(batch["prev_actions"]).to(device)
    prev_rewards_t = torch.from_numpy(batch["prev_rewards"]).to(device)
    actions_t = torch.from_numpy(batch["actions"]).to(device)
    old_logprobs_t = torch.from_numpy(batch["old_logprobs"]).to(device)
    ext_returns_t = torch.from_numpy(batch["ext_returns"]).to(device)
    int_returns_t = torch.from_numpy(batch["int_returns"]).to(device)
    advantages_t = torch.from_numpy(batch["advantages"]).to(device)
    episode_starts_t = torch.from_numpy(batch["episode_starts"]).to(device)
    rnd_weights_t = torch.from_numpy(batch["rnd_weights"]).to(device)
    action_prior_bias_t = torch.from_numpy(batch["action_prior_bias"]).to(device)
    hidden_start_t = torch.from_numpy(batch["hidden_start"]).to(device)

    n_envs = actions_t.shape[0]
    minibatch_envs = max(1, min(int(cfg.minibatch_envs), n_envs))
    policy_losses: List[float] = []
    ext_value_losses: List[float] = []
    int_value_losses: List[float] = []
    rnd_losses: List[float] = []
    entropies: List[float] = []
    turn_balance_losses: List[float] = []
    turn_floor_losses: List[float] = []
    total_losses: List[float] = []

    env_indices = list(range(n_envs))
    for _ in range(int(cfg.ppo_epochs)):
        random.shuffle(env_indices)
        for start in range(0, n_envs, minibatch_envs):
            idx = env_indices[start : start + minibatch_envs]
            logits, ext_values, int_values, _ = net.forward_sequence(
                {k: v[idx] for k, v in states_t.items()},
                prev_actions_t[idx],
                prev_rewards_t[idx],
                hidden_start_t[:, idx],
                reset_mask=episode_starts_t[idx],
            )
            dist = torch.distributions.Categorical(logits=logits + action_prior_bias_t[idx])
            new_logprobs = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            mean_action_probs = dist.probs.mean(dim=(0, 1))
            left_prob = mean_action_probs[ACTION_LEFT]
            right_prob = mean_action_probs[ACTION_RIGHT]
            turn_balance_loss = (left_prob - right_prob).pow(2)
            min_turn_prob = torch.tensor(float(cfg.min_turn_probability), dtype=mean_action_probs.dtype, device=device)
            turn_floor_loss = torch.relu(min_turn_prob - left_prob) + torch.relu(min_turn_prob - right_prob)

            ratio = torch.exp(new_logprobs - old_logprobs_t[idx])
            adv = advantages_t[idx]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            ext_value_loss = 0.5 * (ext_returns_t[idx] - ext_values).pow(2).mean()
            int_value_loss = 0.5 * (int_returns_t[idx] - int_values).pow(2).mean()

            rnd_error, _, _ = net.rnd_prediction_error({k: v[idx] for k, v in next_states_t.items()})
            rnd_loss = (rnd_error * rnd_weights_t[idx]).sum() / rnd_weights_t[idx].sum().clamp_min(1.0)

            total_loss = (
                policy_loss
                + cfg.value_loss_coef * ext_value_loss
                + cfg.intrinsic_value_loss_coef * int_value_loss
                + cfg.rnd_loss_coef * rnd_loss
                + cfg.turn_balance_loss_coef * turn_balance_loss
                + cfg.turn_floor_loss_coef * turn_floor_loss
                - cfg.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_gradient_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            ext_value_losses.append(float(ext_value_loss.item()))
            int_value_losses.append(float(int_value_loss.item()))
            rnd_losses.append(float(rnd_loss.item()))
            entropies.append(float(entropy.item()))
            turn_balance_losses.append(float(turn_balance_loss.item()))
            turn_floor_losses.append(float(turn_floor_loss.item()))
            total_losses.append(float(total_loss.item()))

    return (
        float(np.mean(total_losses)) if total_losses else 0.0,
        float(np.mean(policy_losses)) if policy_losses else 0.0,
        float(np.mean(ext_value_losses)) if ext_value_losses else 0.0,
        float(np.mean(int_value_losses)) if int_value_losses else 0.0,
        float(np.mean(rnd_losses)) if rnd_losses else 0.0,
        float(np.mean(entropies)) if entropies else 0.0,
        float(np.mean(turn_balance_losses)) if turn_balance_losses else 0.0,
        float(np.mean(turn_floor_losses)) if turn_floor_losses else 0.0,
    )


@torch.no_grad()
def evaluate_policy(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: RecurrentPPORNDNetwork,
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
    final_healths: List[float] = []
    lengths: List[float] = []
    survived_flags: List[float] = []
    death_flags: List[float] = []
    diag_metrics: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, reset_info = env.reset(seed=seed_start + ep)
        state = obs_to_state(obs, env.config, switches)
        hidden = net.initial_hidden(1, device)
        prev_action = np.zeros((1, N_ACTIONS), dtype=np.float32)
        prev_reward = np.zeros((1,), dtype=np.float32)
        episode_start = np.ones((1,), dtype=np.float32)
        done = False
        ep_reward = 0.0
        last_info = None
        diag = init_episode_diagnostics(float(reset_info.get("coverage", 0.0)))

        while not done:
            state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
            prev_action_t = torch.from_numpy(prev_action).unsqueeze(1).to(device)
            prev_reward_t = torch.from_numpy(prev_reward).view(1, 1, 1).to(device)
            reset_mask_t = torch.from_numpy(episode_start).view(1, 1).to(device)
            logits, _, _, hidden = net.forward_sequence(
                state_t,
                prev_action_t,
                prev_reward_t,
                hidden,
                reset_mask=reset_mask_t,
            )
            action_prior_bias = policy_prior_logit_bias_from_state(state, prev_action[0], cfg)
            biased_logits = logits[:, -1] + torch.from_numpy(action_prior_bias).to(device).view(1, -1)
            action = int(torch.argmax(biased_logits, dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            update_episode_diagnostics(diag, action, info)
            state = obs_to_state(next_obs, env.config, switches)
            prev_action[0] = onehot_action(action)
            prev_reward[0] = float(reward)
            episode_start[0] = 0.0
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


def build_model_from_checkpoint(payload: Dict[str, Any], device: str = "cpu") -> nn.Module:
    net = base_build_model_from_checkpoint(payload, device=device)
    cfg = TrainConfig()
    train_cfg_payload = payload.get("train_config", {})
    if isinstance(train_cfg_payload, dict):
        cfg_kwargs = {
            key: value
            for key, value in train_cfg_payload.items()
            if key in TrainConfig.__dataclass_fields__
        }
        cfg = TrainConfig(**cfg_kwargs)
    setattr(net, "_train_cfg", cfg)
    return net


def predict_action_for_gui(
    net: nn.Module,
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    ctx = runtime_context or init_runtime_context()
    device = choose_device(str(ctx.get("device", "cpu")))
    state = obs_to_state(obs, env_cfg, switches)
    cfg = getattr(net, "_train_cfg", TrainConfig())
    state_t = {k: torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action = np.asarray(ctx.get("prev_action", np.zeros((N_ACTIONS,), dtype=np.float32)), dtype=np.float32)
    prev_action_t = torch.from_numpy(prev_action).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[float(ctx.get("prev_reward", 0.0))]]], dtype=torch.float32, device=device)
    hidden = ctx.get("hidden", None)
    with torch.no_grad():
        logits, _, _, hidden_out = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        action_prior_bias = policy_prior_logit_bias_from_state(state, prev_action, cfg)
        biased_logits = logits[:, -1] + torch.from_numpy(action_prior_bias).to(device).view(1, -1)
        action = int(torch.argmax(biased_logits, dim=-1).item())
    next_ctx = dict(ctx)
    next_ctx["hidden"] = hidden_out.detach()
    next_ctx["device"] = str(device)
    return action, next_ctx


def checkpoint_payload(
    net: RecurrentPPORNDNetwork,
    input_dim: int,
    switches: ObservationSwitches,
    env: SensoryGridEnv,
    cfg: TrainConfig,
    update: int,
    episodes_seen: int,
    global_step: int,
    eval_metrics: Dict[str, float] | None = None,
    holdout_eval_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    metrics = eval_metrics or {}
    holdout_metrics = holdout_eval_metrics or {}
    return {
        "model_state_dict": net.state_dict(),
        "model_arch": "patch_fusion_gru_ppo_rnd_frontier_prior",
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
            "rnd_hidden_dim": cfg.rnd_hidden_dim,
            "rnd_output_dim": cfg.rnd_output_dim,
            "uses_prev_action": True,
            "uses_prev_reward": True,
            "algorithm": "ppo_rnd_frontier_prior",
        },
        "input_dim": input_dim,
        "num_actions": N_ACTIONS,
        "switches": asdict(switches),
        "env_config": asdict(env.config),
        "train_config": asdict(cfg),
        "update": update,
        "episode": update,
        "episodes_seen": episodes_seen,
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
        f.write("model_arch: patch_fusion_gru_ppo_rnd_frontier_prior\n")
        for key, value in asdict(cfg).items():
            f.write(f"{key}: {value}\n")
        f.write(f"input_dim: {input_dim}\n")
        f.write(f"num_actions: {N_ACTIONS}\n")
        f.write(f"actions: forward={ACTION_FORWARD}, left={ACTION_LEFT}, right={ACTION_RIGHT}\n")
        f.write("\n[env_config]\n")
        for key, value in asdict(env.config).items():
            f.write(f"{key}: {value}\n")
        f.write("\n[switches]\n")
        for key, value in asdict(switches).items():
            f.write(f"{key}: {value}\n")


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "gui.current_environment.sensory_grid_env",
        "model_family": "patch_fusion_gru_ppo_rnd_frontier_prior",
        "default_switches": asdict(build_default_switches()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recurrent PPO/RND with frontier prior")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--save_dir", type=str, default="runs/recurrent_ppo_rnd_frontier_prior")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--holdout_eval_episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = TrainConfig(
        updates=args.updates,
        save_dir=args.save_dir,
        device=args.device,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        eval_episodes=args.eval_episodes,
        holdout_eval_episodes=args.holdout_eval_episodes,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)

    env, switches = build_training_env(cfg)
    input_dim = env.observation_dim(switches)
    device = choose_device(cfg.device)

    net = build_network(env, switches, cfg).to(device)
    setattr(net, "_train_cfg", cfg)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    intrinsic_rms = RunningMeanStd()

    envs: List[SensoryGridEnv] = []
    current_obs: List[Dict[str, Any]] = []
    current_states: List[Dict[str, np.ndarray]] = []
    diag_trackers: List[Dict[str, object]] = []
    for env_idx in range(cfg.n_envs):
        rollout_env, rollout_switches = build_training_env(cfg)
        assert asdict(rollout_switches) == asdict(switches)
        obs, reset_info = rollout_env.reset(seed=cfg.seed + env_idx)
        envs.append(rollout_env)
        current_obs.append(obs)
        current_states.append(obs_to_state(obs, rollout_env.config, switches))
        diag_trackers.append(init_episode_diagnostics(float(reset_info.get("coverage", 0.0))))

    hidden = net.initial_hidden(cfg.n_envs, device)
    prev_actions = np.zeros((cfg.n_envs, N_ACTIONS), dtype=np.float32)
    prev_rewards = np.zeros((cfg.n_envs,), dtype=np.float32)
    episode_starts = np.ones((cfg.n_envs,), dtype=np.float32)
    episode_returns = np.zeros((cfg.n_envs,), dtype=np.float32)

    eval_env, eval_switches = build_training_env(cfg)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_train_config(save_dir, cfg, env, switches, input_dim)

    csv_path = save_dir / "training_log.csv"
    eval_fieldnames = eval_metric_names("eval")
    holdout_eval_fieldnames = eval_metric_names("holdout_eval")
    train_diag_fieldnames = [f"train_{name}_mean" for name in episode_diagnostic_metric_names()]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "update",
            "episodes_seen",
            "global_step",
            "rollout_ext_reward_mean",
            "rollout_int_reward_mean",
            "train_reward_mean",
            "train_coverage_mean",
            "train_final_health_mean",
            "train_length_mean",
            *train_diag_fieldnames,
            "mean_total_loss",
            "mean_policy_loss",
            "mean_ext_value_loss",
            "mean_int_value_loss",
            "mean_rnd_loss",
            "mean_entropy",
            "mean_turn_balance_loss",
            "mean_turn_floor_loss",
            *eval_fieldnames,
            *holdout_eval_fieldnames,
            "eval_soft_score",
        ])

    best_selection = (-float("inf"),) * 6
    best_eval_reward = -float("inf")
    best_eval_coverage = -float("inf")
    best_eval_survival = -float("inf")
    best_eval_soft = -float("inf")
    episodes_seen = 0
    global_step = 0
    reset_seed_cursor = cfg.seed + cfg.n_envs + 10_000

    for update in range(1, cfg.updates + 1):
        rollout_batch, current_obs, current_states, prev_actions, prev_rewards, hidden, episode_starts, diag_trackers, episode_returns, reset_seed_cursor, episodes_seen, global_step = collect_rollout(
            envs,
            switches,
            net,
            device,
            cfg,
            current_obs,
            current_states,
            prev_actions,
            prev_rewards,
            hidden,
            episode_starts,
            diag_trackers,
            episode_returns,
            intrinsic_rms,
            reset_seed_cursor,
            episodes_seen,
            global_step,
        )
        train_batch = rollout_batch_to_training_format(rollout_batch, cfg)
        mean_total_loss, mean_policy_loss, mean_ext_value_loss, mean_int_value_loss, mean_rnd_loss, mean_entropy, mean_turn_balance_loss, mean_turn_floor_loss = ppo_update(
            net,
            optimizer,
            train_batch,
            device,
            cfg,
        )

        do_eval = update == 1 or update % cfg.eval_every == 0 or update == cfg.updates
        if do_eval:
            eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                net,
                device,
                cfg,
                episodes=cfg.eval_episodes,
                seed_start=cfg.eval_seed_start,
                metric_prefix="eval",
            )
            holdout_eval_metrics = evaluate_policy(
                eval_env,
                eval_switches,
                net,
                device,
                cfg,
                episodes=cfg.holdout_eval_episodes,
                seed_start=cfg.holdout_eval_seed_start,
                metric_prefix="holdout_eval",
            )
        else:
            eval_metrics = empty_eval_metrics("eval")
            holdout_eval_metrics = empty_eval_metrics("holdout_eval")

        eval_score = composite_eval_score(eval_metrics, cfg) if do_eval else float("nan")
        train_stats = rollout_batch["completed_episode_stats"]
        print(
            f"update={update:04d} episodes={episodes_seen:05d} step={global_step:07d} "
            f"rollout_ext={rollout_batch['rollout_ext_reward_mean']:+.3f} "
            f"rollout_int={rollout_batch['rollout_int_reward_mean']:+.3f} "
            f"train_cov={train_stats['train_coverage_mean']:.3f} "
            f"train_new={train_stats['train_new_tile_rate_mean']:.3f} "
            f"train_revisit={train_stats['train_revisit_rate_mean']:.3f} "
            f"loss={mean_total_loss:.4f} pg={mean_policy_loss:.4f} "
            f"vext={mean_ext_value_loss:.4f} vint={mean_int_value_loss:.4f} rnd={mean_rnd_loss:.4f} ent={mean_entropy:.4f} "
            f"turnbal={mean_turn_balance_loss:.4f} turnfloor={mean_turn_floor_loss:.4f} "
            f"eval_cov={eval_metrics['eval_coverage_mean']:.3f} "
            f"holdout_cov={holdout_eval_metrics['holdout_eval_coverage_mean']:.3f} "
            f"soft={eval_score:.3f}",
            flush=True,
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                update,
                episodes_seen,
                global_step,
                rollout_batch["rollout_ext_reward_mean"],
                rollout_batch["rollout_int_reward_mean"],
                train_stats["train_reward_mean"],
                train_stats["train_coverage_mean"],
                train_stats["train_final_health_mean"],
                train_stats["train_length_mean"],
                *[train_stats[name] for name in train_diag_fieldnames],
                mean_total_loss,
                mean_policy_loss,
                mean_ext_value_loss,
                mean_int_value_loss,
                mean_rnd_loss,
                mean_entropy,
                mean_turn_balance_loss,
                mean_turn_floor_loss,
                *[eval_metrics[name] for name in eval_fieldnames],
                *[holdout_eval_metrics[name] for name in holdout_eval_fieldnames],
                eval_score,
            ])

        if do_eval and cfg.save_eval_checkpoints:
            payload = checkpoint_payload(
                net,
                input_dim,
                switches,
                env,
                cfg,
                update,
                episodes_seen,
                global_step,
                eval_metrics,
                holdout_eval_metrics,
            )
            torch.save(payload, save_dir / f"ckpt_upd{update:04d}.pt")

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
        checkpoint_payload(net, input_dim, switches, env, cfg, cfg.updates, episodes_seen, global_step, {}, {}),
        save_dir / "final_model.pt",
    )
    print(f"Training finished. Logs and checkpoints saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()

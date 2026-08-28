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

TRAINER_GUI_INTERFACE_VERSION = "recurrent-patch-fusion-dqn-reference"
TRAINER_DISPLAY_NAME = "Recurrent patch-fusion DQN reference"


@dataclass
class TrainConfig:
    episodes: int = 1200
    seed: int = 7
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_gradient_norm: float = 10.0
    save_dir: str = "runs/15x15/recurrent_patch_fusion_dqn_reference"
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

    eval_every: int = 20
    eval_episodes: int = 10
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

    # v5 change 1: best_model uses soft score only, no hard survival gate.
    best_model_survival_weight: float = 0.30
    best_model_reward_weight: float = 0.04
    best_model_health_weight: float = 0.02

    # v5 change 2: slight survival pull, but still weaker than v4.1.
    training_survival_bonus: float = 0.60


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

        for _ in range(batch_size):
            ep = self._sample_episode()
            start = random.randrange(max(1, len(ep)))
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


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / max(1, cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def current_aux_weight(episode: int, cfg: TrainConfig) -> float:
    if episode >= cfg.aux_weight_decay_episodes:
        return cfg.aux_health_delta_loss_weight_end
    frac = episode / max(1, cfg.aux_weight_decay_episodes)
    return cfg.aux_health_delta_loss_weight_start + frac * (
        cfg.aux_health_delta_loss_weight_end - cfg.aux_health_delta_loss_weight_start
    )


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

    q_seq, _, health_delta_seq = online_net.forward_sequence(states_main, prev_actions_main, prev_rewards_main, h_online)
    chosen_q = q_seq.gather(-1, actions_main.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_online_q, _, _ = online_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_online)
        next_actions = torch.argmax(next_online_q, dim=-1, keepdim=True)
        next_target_q, _, _ = target_net.forward_sequence(next_states_main, next_prev_actions_main, next_prev_rewards_main, h_target)
        next_q = next_target_q.gather(-1, next_actions).squeeze(-1)
        targets = rewards_main + cfg.gamma * (1.0 - dones_main) * next_q

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
) -> Dict[str, float]:
    rewards: List[float] = []
    coverages: List[float] = []
    final_healths: List[float] = []
    lengths: List[float] = []
    survived_flags: List[float] = []
    death_flags: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        state = obs_to_state(obs, env.config, switches)
        done = False
        ep_reward = 0.0
        hidden = None
        prev_action = np.zeros((1, 1, N_ACTIONS), dtype=np.float32)
        prev_reward = np.zeros((1, 1, 1), dtype=np.float32)
        last_info = None

        while not done:
            state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
            prev_action_t = torch.from_numpy(prev_action).to(device)
            prev_reward_t = torch.from_numpy(prev_reward).to(device)
            q, hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
            action = int(torch.argmax(q[:, -1], dim=-1).item())
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
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

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_coverage_mean": float(np.mean(coverages)),
        "eval_final_health_mean": float(np.mean(final_healths)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_survival_rate": float(np.mean(survived_flags)),
        "eval_death_rate": float(np.mean(death_flags)),
    }


def composite_eval_score(eval_metrics: Dict[str, float], cfg: TrainConfig) -> float:
    reward_term = max(float(eval_metrics.get("eval_reward_mean", 0.0)), 0.0)
    return (
        1.00 * float(eval_metrics.get("eval_coverage_mean", 0.0))
        + cfg.best_model_survival_weight * float(eval_metrics.get("eval_survival_rate", 0.0))
        + cfg.best_model_reward_weight * reward_term
        + cfg.best_model_health_weight * max(float(eval_metrics.get("eval_final_health_mean", 0.0)), 0.0)
    )


def selection_key(eval_metrics: Dict[str, float], cfg: TrainConfig) -> Tuple[float, float, float, float, float]:
    return (
        composite_eval_score(eval_metrics, cfg),
        float(eval_metrics.get("eval_coverage_mean", float("-inf"))),
        float(eval_metrics.get("eval_survival_rate", float("-inf"))),
        float(eval_metrics.get("eval_reward_mean", float("-inf"))),
        float(eval_metrics.get("eval_final_health_mean", float("-inf"))),
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
) -> Dict[str, object]:
    metrics = eval_metrics or {}
    return {
        "model_state_dict": online_net.state_dict(),
        "model_arch": "patch_fusion_gru_dueling_double_dqn_reference",
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
        "selection_key": list(selection_key(metrics, cfg)) if metrics else [],
        "composite_eval_score": composite_eval_score(metrics, cfg) if metrics else None,
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
    }


def write_train_config(save_dir: Path, cfg: TrainConfig, env: SensoryGridEnv, switches: ObservationSwitches, input_dim: int) -> None:
    with open(save_dir / "train_config.txt", "w", encoding="utf-8") as f:
        f.write("model_arch: patch_fusion_gru_dueling_double_dqn_reference\n")
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
    state: Dict[str, np.ndarray],
    prev_action: np.ndarray,
    prev_reward: float,
    hidden: torch.Tensor | None,
    device: torch.device,
    epsilon: float,
) -> Tuple[int, torch.Tensor]:
    state_t = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device) for k, v in state.items()}
    prev_action_t = torch.from_numpy(prev_action).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[prev_reward]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, new_hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        if random.random() < epsilon:
            action = random.randrange(N_ACTIONS)
        else:
            action = int(torch.argmax(q[:, -1], dim=-1).item())
    return action, new_hidden.detach()


def episode_priority(last_info: Dict[str, object], ep_reward: float) -> float:
    coverage = float(last_info.get("coverage", 0.0))
    survived = 1.0 if (last_info.get("truncated") and not last_info.get("terminated")) else 0.0
    final_health = max(float(last_info.get("health", 0.0)), 0.0)
    reward_pos = max(float(ep_reward), 0.0)
    return 1.0 + 1.8 * coverage + 1.5 * survived + 0.08 * final_health + 0.02 * reward_pos


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "gui.current_environment.sensory_grid_env",
        "model_family": "patch_fusion_gru_dueling_double_dqn_aux",
        "default_switches": asdict(build_default_switches()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recurrent patch-fusion Double DQN reference trainer")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--save_dir", type=str, default="runs/15x15/recurrent_patch_fusion_dqn_reference")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = TrainConfig(episodes=args.episodes, save_dir=args.save_dir, device=args.device)
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
            "mean_loss",
            "mean_q_loss",
            "mean_aux_loss",
            "aux_weight",
            "eval_reward_mean",
            "eval_coverage_mean",
            "eval_final_health_mean",
            "eval_length_mean",
            "eval_survival_rate",
            "eval_death_rate",
            "eval_soft_score",
        ])

    best_selection = (-float("inf"),) * 5
    best_eval_reward = -float("inf")
    best_eval_coverage = -float("inf")
    best_eval_survival = -float("inf")
    best_eval_soft = -float("inf")
    global_step = 0

    for episode in range(1, cfg.episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + episode)
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

        while not done:
            epsilon = linear_epsilon(global_step, cfg)
            action, hidden = choose_action(online_net, state, prev_action, prev_reward, hidden, device, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action, switches)
            next_state = obs_to_state(next_obs, env.config, switches)
            done = terminated or truncated

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
            })

            state = next_state
            prev_action = onehot_action(action)
            prev_reward = float(reward)
            ep_reward += reward
            last_info = info
            global_step += 1

        assert last_info is not None
        replay.add_episode(episode_transitions, priority=episode_priority(last_info, ep_reward))

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
            eval_metrics = evaluate_policy(eval_env, eval_switches, online_net, device, episodes=cfg.eval_episodes)
        else:
            eval_metrics = {
                "eval_reward_mean": float("nan"),
                "eval_coverage_mean": float("nan"),
                "eval_final_health_mean": float("nan"),
                "eval_length_mean": float("nan"),
                "eval_survival_rate": float("nan"),
                "eval_death_rate": float("nan"),
            }

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_q_loss = float(np.mean(q_losses)) if q_losses else 0.0
        mean_aux_loss = float(np.mean(aux_losses)) if aux_losses else 0.0
        epsilon = linear_epsilon(global_step, cfg)
        eval_score = composite_eval_score(eval_metrics, cfg) if do_eval else float("nan")
        print(
            f"episode={episode:04d} step={global_step:06d} eps={epsilon:.3f} "
            f"reward={ep_reward:+.3f} coverage={last_info['coverage']:.3f} "
            f"health={last_info['health']} len={last_info['steps']} "
            f"loss={mean_loss:.4f} q={mean_q_loss:.4f} aux={mean_aux_loss:.4f} aw={aux_weight:.3f} "
            f"eval_surv={eval_metrics['eval_survival_rate']:.3f} "
            f"eval_cov={eval_metrics['eval_coverage_mean']:.3f} "
            f"eval_reward={eval_metrics['eval_reward_mean']:.3f} soft={eval_score:.3f}",
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
                mean_loss,
                mean_q_loss,
                mean_aux_loss,
                aux_weight,
                eval_metrics["eval_reward_mean"],
                eval_metrics["eval_coverage_mean"],
                eval_metrics["eval_final_health_mean"],
                eval_metrics["eval_length_mean"],
                eval_metrics["eval_survival_rate"],
                eval_metrics["eval_death_rate"],
                eval_score,
            ])

        if do_eval and cfg.save_eval_checkpoints:
            payload = checkpoint_payload(online_net, input_dim, switches, env, cfg, episode, global_step, eval_metrics)
            torch.save(payload, save_dir / f"ckpt_ep{episode:04d}.pt")

            current_key = selection_key(eval_metrics, cfg)
            if current_key > best_selection:
                best_selection = current_key
                torch.save(payload, save_dir / "best_model.pt")

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
        checkpoint_payload(online_net, input_dim, switches, env, cfg, cfg.episodes, global_step, {}),
        save_dir / "final_model.pt",
    )
    print(f"Training finished. Logs and checkpoints saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()

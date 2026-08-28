from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

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
    OBJ_EMPTY,
    OBJ_FIRE,
    OBJ_GLASS,
    OBJ_ICE,
    ObservationSwitches,
    SensoryGridEnv,
)
from models.shared.recurrent_dqn_interface import (
    ModelConfig,
    PatchEncoder,
    choose_device,
    onehot_action,
    reset_runtime_context as base_reset_runtime_context,
)


TRAINER_GUI_INTERFACE_VERSION = "goal-conditioned-memory-dqn"
TRAINER_DISPLAY_NAME = "Goal-conditioned memory-map Double DQN"
MODEL_ARCH = "memory_goal_gru_dueling_double_dqn"
MEMORY_CHANNELS = 7
SCALAR_STATE_DIM = 10
THERMAL_OBJECT_IDS = {OBJ_FIRE, OBJ_ICE}
HAZARD_OBJECT_IDS = {OBJ_FIRE, OBJ_ICE, OBJ_GLASS}


@dataclass
class TrainConfig:
    episodes: int = 1200
    seed: int = 7
    gamma: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_gradient_norm: float = 10.0
    save_dir: str = "runs/goal_conditioned_memory_dqn"
    device: str = "auto"

    replay_capacity_episodes: int = 2000
    replay_priority_alpha: float = 0.65
    replay_priority_epsilon: float = 1e-3
    batch_size: int = 32
    burn_in: int = 12
    unroll_len: int = 32
    train_after_episodes: int = 20
    train_updates_per_episode: int = 10
    n_step_return: int = 3

    target_soft_tau: float = 0.02
    target_hard_sync_every_episodes: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 45_000
    teacher_probability: float = 0.90
    imitation_weight_start: float = 0.25
    imitation_weight_end: float = 0.05
    imitation_anneal_steps: int = 90_000

    eval_every: int = 25
    eval_episodes: int = 20
    holdout_eval_episodes: int = 20
    eval_seed_start: int = 1000
    holdout_eval_seed_start: int = 2000
    last_checkpoint_every_episodes: int = 5

    conv_channels_1: int = 16
    conv_channels_2: int = 32
    vision_embed_dim: int = 96
    scalar_patch_embed_dim: int = 48
    scalar_state_embed_dim: int = 48
    memory_map_embed_dim: int = 96
    obs_embed_dim: int = 256
    gru_hidden_dim: int = 256
    head_hidden_dim: int = 128

    # The game reward is deliberately not used for Q-learning. These are a
    # separate learning objective and leave all environment dynamics unchanged.
    seen_phase_threshold: float = 0.80
    early_seen_cell_reward: float = 0.035
    early_visit_reward: float = 0.45
    late_visit_reward: float = 1.00
    step_cost: float = 0.002
    # Consecutive revisits cost initial * growth^(streak - 1). The streak
    # resets only when the agent physically enters a new cell.
    revisit_penalty_initial: float = 0.015
    revisit_penalty_growth: float = 1.65
    revisit_penalty_cap: float = 0.50

    thermal_exclusion_radius: int = 1
    goal_commitment_steps: int = 12
    early_goal_center_weight: float = 1.20
    early_goal_information_weight: float = 1.00
    late_goal_component_weight: float = 1.00
    late_goal_route_unvisited_weight: float = 1.25

    # Kept equal to v9 so v10 changes the learner, not game settings.
    training_survival_bonus: float = 0.60


class MemoryGoalDuelingQNetwork(nn.Module):
    """GRU Double-DQN with current sensory patches and an agent-owned map."""

    def __init__(
        self,
        patch_size: int,
        map_size: int,
        num_actions: int,
        cfg: ModelConfig,
        memory_map_embed_dim: int,
        scalar_state_dim: int = SCALAR_STATE_DIM,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.vision_encoder = PatchEncoder(6, patch_size, cfg.conv_channels_1, cfg.conv_channels_2, cfg.vision_embed_dim)
        self.temperature_encoder = PatchEncoder(
            1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim
        )
        self.smell_encoder = PatchEncoder(
            1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim
        )
        self.visited_encoder = PatchEncoder(
            1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim
        )
        self.hazard_encoder = PatchEncoder(
            1, patch_size, max(8, cfg.conv_channels_1 // 2), max(16, cfg.conv_channels_2 // 2), cfg.scalar_patch_embed_dim
        )
        self.memory_encoder = PatchEncoder(
            MEMORY_CHANNELS, map_size, cfg.conv_channels_1, cfg.conv_channels_2, memory_map_embed_dim
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_state_dim, cfg.scalar_state_embed_dim), nn.ReLU())
        fusion_dim = (
            cfg.vision_embed_dim
            + 4 * cfg.scalar_patch_embed_dim
            + memory_map_embed_dim
            + cfg.scalar_state_embed_dim
        )
        self.obs_fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.obs_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.obs_embed_dim, cfg.obs_embed_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=cfg.obs_embed_dim + self.num_actions + 1,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim), nn.ReLU(), nn.Linear(cfg.head_hidden_dim, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.head_hidden_dim), nn.ReLU(), nn.Linear(cfg.head_hidden_dim, self.num_actions)
        )

    def encode_obs_sequence(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = state["scalars"].shape[:2]

        def flatten_seq(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch_size * seq_len, *value.shape[2:])

        parts = [
            self.vision_encoder(flatten_seq(state["vision"])),
            self.temperature_encoder(flatten_seq(state["temperature_patch"])),
            self.smell_encoder(flatten_seq(state["smell_patch"])),
            self.visited_encoder(flatten_seq(state["visited_patch"])),
            self.hazard_encoder(flatten_seq(state["hazard_patch"])),
            self.memory_encoder(flatten_seq(state["memory_map"])),
            self.scalar_encoder(flatten_seq(state["scalars"])),
        ]
        fused = self.obs_fusion(torch.cat(parts, dim=1))
        return fused.view(batch_size, seq_len, -1)

    def forward_sequence(
        self,
        state: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        obs_embed = self.encode_obs_sequence(state)
        out, hidden_out = self.gru(torch.cat([obs_embed, prev_action, prev_reward], dim=-1), hidden)
        value = self.value_head(out)
        advantage = self.adv_head(out)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q, hidden_out, None


class AgentMemory:
    """Map constructed only from ego observations, observed orientation and actions."""

    def __init__(self, grid_size: int, patch_size: int, cfg: TrainConfig):
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.cfg = cfg
        self.observed = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.visited = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.safe = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.exclusion = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.blocked = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.goal: Tuple[int, int] | None = None
        self.goal_steps_remaining = 0
        self.position = (grid_size // 2, grid_size // 2)
        self.direction = 0
        self.initialized = False

    @staticmethod
    def _ego_to_map(position: Tuple[int, int], direction: int, ego_dr: int, ego_dc: int) -> Tuple[int, int]:
        row, col = position
        if direction == 0:
            return row + ego_dr, col + ego_dc
        if direction == 1:
            return row + ego_dc, col - ego_dr
        if direction == 2:
            return row - ego_dr, col - ego_dc
        return row - ego_dc, col + ego_dr

    @staticmethod
    def _direction_from_obs(obs: Dict[str, Any]) -> int:
        direction = np.asarray(obs.get("direction_onehot", np.zeros(4)), dtype=np.float32)
        return int(np.argmax(direction)) if direction.size == 4 else int(obs.get("direction", 0)) % 4

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def reset(self, obs: Dict[str, Any]) -> int:
        self.observed.fill(0.0)
        self.visited.fill(0.0)
        self.safe.fill(0.0)
        self.exclusion.fill(0.0)
        self.blocked.fill(0.0)
        self.goal = None
        self.goal_steps_remaining = 0
        self.position = (self.grid_size // 2, self.grid_size // 2)
        self.direction = self._direction_from_obs(obs)
        self.initialized = True
        return self.observe(obs)

    def _mark_thermal_exclusion(self, row: int, col: int) -> None:
        radius = max(0, int(self.cfg.thermal_exclusion_radius))
        r0 = max(0, row - radius)
        r1 = min(self.grid_size, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(self.grid_size, col + radius + 1)
        self.exclusion[r0:r1, c0:c1] = 1.0

    def observe(self, obs: Dict[str, Any]) -> int:
        if not self.initialized:
            return self.reset(obs)
        self.direction = self._direction_from_obs(obs)
        vision = np.asarray(obs.get("vision"), dtype=np.int64)
        half = vision.shape[0] // 2
        new_seen = 0
        for patch_row in range(vision.shape[0]):
            for patch_col in range(vision.shape[1]):
                map_pos = self._ego_to_map(
                    self.position,
                    self.direction,
                    patch_row - half,
                    patch_col - half,
                )
                if not self._in_bounds(map_pos):
                    continue
                row, col = map_pos
                if self.observed[row, col] < 0.5:
                    new_seen += 1
                self.observed[row, col] = 1.0
                object_id = int(vision[patch_row, patch_col])
                if object_id == OBJ_EMPTY:
                    self.safe[row, col] = 1.0
                elif object_id in HAZARD_OBJECT_IDS:
                    self.safe[row, col] = 0.0
                if object_id in THERMAL_OBJECT_IDS:
                    self._mark_thermal_exclusion(row, col)
                elif object_id == OBJ_GLASS:
                    self.exclusion[row, col] = 1.0
        self.visited[self.position] = 1.0
        return int(new_seen)

    def update_after_action(self, action: int, next_obs: Dict[str, Any]) -> int:
        if not self.initialized:
            return self.reset(next_obs)
        previous_direction = self.direction
        if int(action) == ACTION_FORWARD:
            direction_vectors = ((-1, 0), (0, 1), (1, 0), (0, -1))
            dr, dc = direction_vectors[previous_direction]
            ahead = (self.position[0] + dr, self.position[1] + dc)
            did_move = int(next_obs.get("consecutive_no_move_steps", 0)) == 0
            if did_move and self._in_bounds(ahead):
                self.position = ahead
            elif self._in_bounds(ahead):
                self.blocked[ahead] = 1.0
        self.direction = self._direction_from_obs(next_obs)
        return self.observe(next_obs)

    @property
    def seen_fraction(self) -> float:
        return float(np.mean(self.observed > 0.5))

    def in_late_phase(self) -> bool:
        return self.seen_fraction >= float(self.cfg.seen_phase_threshold)

    def _route(self, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        if not self._in_bounds(target):
            return []
        traversable = (self.blocked < 0.5) & (self.exclusion < 0.5)
        # A newly observed fire/ice mask can include the current cell. The
        # agent may leave it, but will not route through it again.
        traversable[self.position] = True
        if not traversable[target]:
            return []
        queue: Deque[Tuple[int, int]] = deque([self.position])
        parents: Dict[Tuple[int, int], Tuple[int, int] | None] = {self.position: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (current[0] + dr, current[1] + dc)
                if self._in_bounds(nxt) and traversable[nxt] and nxt not in parents:
                    parents[nxt] = current
                    queue.append(nxt)
        if target not in parents:
            return []
        route = [target]
        while route[-1] != self.position:
            parent = parents[route[-1]]
            assert parent is not None
            route.append(parent)
        route.reverse()
        return route

    def _frontier_information(self, target: Tuple[int, int]) -> int:
        half = self.patch_size // 2
        row0 = max(0, target[0] - half)
        row1 = min(self.grid_size, target[0] + half + 1)
        col0 = max(0, target[1] - half)
        col1 = min(self.grid_size, target[1] + half + 1)
        return int(np.count_nonzero(self.observed[row0:row1, col0:col1] < 0.5))

    def _safe_components(self) -> Iterable[List[Tuple[int, int]]]:
        available = (self.safe > 0.5) & (self.visited < 0.5) & (self.exclusion < 0.5) & (self.blocked < 0.5)
        explored = np.zeros_like(available, dtype=bool)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if not available[row, col] or explored[row, col]:
                    continue
                component: List[Tuple[int, int]] = []
                queue: Deque[Tuple[int, int]] = deque([(row, col)])
                explored[row, col] = True
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nxt = (current[0] + dr, current[1] + dc)
                        if self._in_bounds(nxt) and available[nxt] and not explored[nxt]:
                            explored[nxt] = True
                            queue.append(nxt)
                yield component

    def _select_goal(self) -> Tuple[int, int] | None:
        # Continue a still-valid goal long enough to avoid turn-by-turn dithering.
        if self.goal is not None and self.goal_steps_remaining > 0:
            route = self._route(self.goal)
            if len(route) >= 2:
                self.goal_steps_remaining -= 1
                return self.goal

        self.goal = None
        self.goal_steps_remaining = 0
        best_target: Tuple[int, int] | None = None
        best_score = -float("inf")
        if not self.in_late_phase():
            # Restrict the revelation phase to the central 13x13 region.
            for row in range(1, self.grid_size - 1):
                for col in range(1, self.grid_size - 1):
                    target = (row, col)
                    route = self._route(target)
                    if len(route) < 2:
                        continue
                    information = self._frontier_information(target)
                    if information <= 0:
                        continue
                    route_cost = max(1, len(route) - 1)
                    center = 1.0 if 1 <= row < self.grid_size - 1 and 1 <= col < self.grid_size - 1 else 0.0
                    score = (
                        float(self.cfg.early_goal_information_weight) * information
                        + float(self.cfg.early_goal_center_weight) * center
                    ) / route_cost
                    if score > best_score:
                        best_target, best_score = target, score
        else:
            for component in self._safe_components():
                component_target = None
                component_route: List[Tuple[int, int]] = []
                route_unvisited = -1
                for candidate in component:
                    route = self._route(candidate)
                    if len(route) < 2:
                        continue
                    route_count = int(sum(self.visited[row, col] < 0.5 for row, col in route[1:]))
                    if route_count > route_unvisited or (route_count == route_unvisited and len(route) > len(component_route)):
                        component_target = candidate
                        component_route = route
                        route_unvisited = route_count
                if component_target is None:
                    continue
                score = (
                    float(self.cfg.late_goal_component_weight) * len(component)
                    + float(self.cfg.late_goal_route_unvisited_weight) * route_unvisited
                ) / max(1, len(component_route) - 1)
                if score > best_score:
                    best_target, best_score = component_target, score
        if best_target is not None:
            self.goal = best_target
            self.goal_steps_remaining = max(0, int(self.cfg.goal_commitment_steps) - 1)
        return best_target

    def teacher_action(self) -> int:
        target = self._select_goal()
        if target is None:
            return ACTION_FORWARD
        route = self._route(target)
        if len(route) < 2:
            return ACTION_FORWARD
        nxt = route[1]
        dr = nxt[0] - self.position[0]
        dc = nxt[1] - self.position[1]
        desired_direction = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}.get((dr, dc), self.direction)
        turn = (desired_direction - self.direction) % 4
        if turn == 0:
            return ACTION_FORWARD
        if turn == 1:
            return ACTION_RIGHT
        return ACTION_LEFT

    def map_tensor(self) -> np.ndarray:
        goal_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if self.goal is not None and self._in_bounds(self.goal):
            goal_map[self.goal] = 1.0
        agent_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        agent_map[self.position] = 1.0
        return np.stack(
            [self.observed, self.visited, self.safe, self.exclusion, self.blocked, goal_map, agent_map], axis=0
        ).astype(np.float32)

    def goal_features(self) -> np.ndarray:
        if self.goal is None:
            delta_row, delta_col = 0.0, 0.0
        else:
            delta_row = float(self.goal[0] - self.position[0]) / max(1.0, self.grid_size - 1)
            delta_col = float(self.goal[1] - self.position[1]) / max(1.0, self.grid_size - 1)
        return np.array([delta_row, delta_col, float(self.in_late_phase()), self.seen_fraction], dtype=np.float32)


def model_config(cfg: TrainConfig) -> ModelConfig:
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
    return SensoryGridEnv(env_cfg), build_default_switches()


def state_from_observation(obs: Dict[str, Any], env_cfg: EnvConfig, switches: ObservationSwitches, memory: AgentMemory) -> Dict[str, np.ndarray]:
    patch_size = int(env_cfg.patch_size)
    vision_ids = np.asarray(obs["vision"], dtype=np.int64)
    vision = np.eye(6, dtype=np.float32)[vision_ids].transpose(2, 0, 1)
    temperature = np.asarray(obs["temperature_patch_c"], dtype=np.float32)
    temperature = np.clip((temperature - env_cfg.ambient_temperature_c) / max(1.0, env_cfg.fire_temp_delta_amp), -1.0, 1.0)
    smell = np.clip(np.asarray(obs["smell_patch"], dtype=np.float32), 0.0, 1.0)
    visited_patch = np.asarray(obs["visited_patch"], dtype=np.float32)
    hazard_patch = np.asarray(obs["hazard_patch"], dtype=np.float32)
    direction = np.asarray(obs["direction_onehot"], dtype=np.float32)
    # Health is deliberately removed; energy remains only as an observed state.
    scalars = np.concatenate(
        [np.array([0.0, float(obs["energy_norm"])], dtype=np.float32), direction, memory.goal_features()], axis=0
    )
    assert scalars.shape == (SCALAR_STATE_DIM,)
    return {
        "vision": vision.astype(np.float32),
        "temperature_patch": temperature.reshape(1, patch_size, patch_size).astype(np.float32),
        "smell_patch": smell.reshape(1, patch_size, patch_size).astype(np.float32),
        "visited_patch": visited_patch.reshape(1, patch_size, patch_size).astype(np.float32),
        "hazard_patch": hazard_patch.reshape(1, patch_size, patch_size).astype(np.float32),
        "memory_map": memory.map_tensor(),
        "scalars": scalars.astype(np.float32),
    }


def copy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: value.copy() for name, value in state.items()}


class PrioritizedSequenceReplay:
    """Episode storage with priorities refreshed from sequence TD errors."""

    def __init__(self, capacity_episodes: int, alpha: float, epsilon: float):
        self.episodes: Deque[List[Dict[str, object]]] = deque(maxlen=int(capacity_episodes))
        self.priorities: Deque[float] = deque(maxlen=int(capacity_episodes))
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, transitions: List[Dict[str, object]]) -> None:
        if transitions:
            initial = max(self.priorities, default=1.0)
            self.episodes.append(transitions)
            self.priorities.append(float(initial))

    def sample(self, batch_size: int, total_len: int) -> Dict[str, object]:
        if not self.episodes:
            raise RuntimeError("Replay is empty")
        priorities = np.asarray(self.priorities, dtype=np.float64)
        weights = np.power(np.maximum(priorities, self.epsilon), self.alpha)
        if not np.isfinite(weights).all() or weights.sum() <= 0.0:
            weights = np.ones_like(weights)

        state_batch: List[List[Dict[str, np.ndarray]]] = []
        next_state_batch: List[List[Dict[str, np.ndarray]]] = []
        prev_action_batch, prev_reward_batch = [], []
        action_batch, reward_batch, done_batch = [], [], []
        next_prev_action_batch, next_prev_reward_batch = [], []
        valid_batch, teacher_batch, sampled_indices = [], [], []
        for episode_index in random.choices(range(len(self.episodes)), weights=weights.tolist(), k=int(batch_size)):
            episode = self.episodes[episode_index]
            max_start = max(0, len(episode) - 1)
            start = random.randint(0, max_start)
            fallback = copy_state(episode[-1]["next_state"])
            states, next_states = [], []
            prev_actions = np.zeros((total_len, N_ACTIONS), dtype=np.float32)
            prev_rewards = np.zeros((total_len, 1), dtype=np.float32)
            actions = np.zeros((total_len,), dtype=np.int64)
            rewards = np.zeros((total_len,), dtype=np.float32)
            dones = np.ones((total_len,), dtype=np.float32)
            next_prev_actions = np.zeros((total_len, N_ACTIONS), dtype=np.float32)
            next_prev_rewards = np.zeros((total_len, 1), dtype=np.float32)
            valid = np.zeros((total_len,), dtype=np.float32)
            teacher_actions = np.zeros((total_len,), dtype=np.int64)
            for offset in range(total_len):
                item_index = start + offset
                if item_index >= len(episode):
                    states.append(copy_state(fallback))
                    next_states.append(copy_state(fallback))
                    continue
                item = episode[item_index]
                states.append(copy_state(item["state"]))
                next_states.append(copy_state(item["next_state"]))
                prev_actions[offset] = np.asarray(item["prev_action"], dtype=np.float32)
                prev_rewards[offset, 0] = float(item["prev_reward"])
                actions[offset] = int(item["action"])
                rewards[offset] = float(item["reward"])
                dones[offset] = float(item["done"])
                next_prev_actions[offset] = np.asarray(item["next_prev_action"], dtype=np.float32)
                next_prev_rewards[offset, 0] = float(item["next_prev_reward"])
                valid[offset] = 1.0
                teacher_actions[offset] = int(item["teacher_action"])
            state_batch.append(states)
            next_state_batch.append(next_states)
            prev_action_batch.append(prev_actions)
            prev_reward_batch.append(prev_rewards)
            action_batch.append(actions)
            reward_batch.append(rewards)
            done_batch.append(dones)
            next_prev_action_batch.append(next_prev_actions)
            next_prev_reward_batch.append(next_prev_rewards)
            valid_batch.append(valid)
            teacher_batch.append(teacher_actions)
            sampled_indices.append(episode_index)

        def collate(sequences: List[List[Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
            return {
                name: np.stack([[item[name] for item in sequence] for sequence in sequences]).astype(np.float32)
                for name in sequences[0][0]
            }

        return {
            "states": collate(state_batch),
            "next_states": collate(next_state_batch),
            "prev_actions": np.stack(prev_action_batch).astype(np.float32),
            "prev_rewards": np.stack(prev_reward_batch).astype(np.float32),
            "actions": np.stack(action_batch).astype(np.int64),
            "rewards": np.stack(reward_batch).astype(np.float32),
            "dones": np.stack(done_batch).astype(np.float32),
            "next_prev_actions": np.stack(next_prev_action_batch).astype(np.float32),
            "next_prev_rewards": np.stack(next_prev_reward_batch).astype(np.float32),
            "valid": np.stack(valid_batch).astype(np.float32),
            "teacher_actions": np.stack(teacher_batch).astype(np.int64),
            "episode_indices": np.asarray(sampled_indices, dtype=np.int64),
        }

    def update_priorities(self, episode_indices: np.ndarray, sequence_errors: np.ndarray) -> None:
        updates: Dict[int, float] = {}
        for index, error in zip(episode_indices.tolist(), sequence_errors.tolist()):
            value = max(float(error) + self.epsilon, self.epsilon)
            updates[int(index)] = max(updates.get(int(index), 0.0), value)
        for index, value in updates.items():
            if 0 <= index < len(self.priorities):
                self.priorities[index] = value


def linear_epsilon(step: int, cfg: TrainConfig) -> float:
    fraction = min(1.0, float(step) / max(1, int(cfg.epsilon_decay_steps)))
    return float(cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start))


def imitation_weight(step: int, cfg: TrainConfig) -> float:
    fraction = min(1.0, float(step) / max(1, int(cfg.imitation_anneal_steps)))
    return float(cfg.imitation_weight_start + fraction * (cfg.imitation_weight_end - cfg.imitation_weight_start))


def n_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    valid: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    n_step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    steps = max(1, int(n_step))
    returns = torch.zeros_like(rewards)
    active = valid > 0.5
    ended = torch.zeros_like(active)
    for offset in range(steps):
        if offset == 0:
            reward_slice, valid_slice, done_slice = rewards, valid, dones
        else:
            remaining = rewards.shape[1] - offset
            reward_slice = torch.zeros_like(rewards)
            valid_slice = torch.zeros_like(valid)
            done_slice = torch.ones_like(dones)
            if remaining > 0:
                reward_slice[:, :remaining] = rewards[:, offset:]
                valid_slice[:, :remaining] = valid[:, offset:]
                done_slice[:, :remaining] = dones[:, offset:]
        usable = active & (valid_slice > 0.5) & ~ended
        returns = returns + (float(gamma) ** offset) * reward_slice * usable.to(rewards.dtype)
        ended = ended | ((done_slice > 0.5) & usable)
        active = usable
    bootstrap = active & ~ended
    target_available = (valid > 0.5) & (ended | bootstrap)
    targets = returns + (float(gamma) ** steps) * bootstrap.to(rewards.dtype) * next_q
    return targets, target_available


def train_step(
    online_net: MemoryGoalDuelingQNetwork,
    target_net: MemoryGoalDuelingQNetwork,
    optimizer: optim.Optimizer,
    replay: PrioritizedSequenceReplay,
    device: torch.device,
    cfg: TrainConfig,
    current_step: int,
) -> Tuple[float, float, float]:
    total_len = int(cfg.burn_in) + int(cfg.unroll_len)
    batch = replay.sample(cfg.batch_size, total_len)
    states = {name: torch.from_numpy(value).to(device) for name, value in batch["states"].items()}
    next_states = {name: torch.from_numpy(value).to(device) for name, value in batch["next_states"].items()}
    prev_actions = torch.from_numpy(batch["prev_actions"]).to(device)
    prev_rewards = torch.from_numpy(batch["prev_rewards"]).to(device)
    actions = torch.from_numpy(batch["actions"]).to(device)
    rewards = torch.from_numpy(batch["rewards"]).to(device)
    dones = torch.from_numpy(batch["dones"]).to(device)
    next_prev_actions = torch.from_numpy(batch["next_prev_actions"]).to(device)
    next_prev_rewards = torch.from_numpy(batch["next_prev_rewards"]).to(device)
    valid = torch.from_numpy(batch["valid"]).to(device)
    teacher_actions = torch.from_numpy(batch["teacher_actions"]).to(device)

    if cfg.burn_in > 0:
        with torch.no_grad():
            _, online_hidden, _ = online_net.forward_sequence(
                {name: value[:, :cfg.burn_in] for name, value in states.items()},
                prev_actions[:, :cfg.burn_in],
                prev_rewards[:, :cfg.burn_in],
            )
            _, target_hidden, _ = target_net.forward_sequence(
                {name: value[:, :cfg.burn_in] for name, value in states.items()},
                prev_actions[:, :cfg.burn_in],
                prev_rewards[:, :cfg.burn_in],
            )
        online_hidden, target_hidden = online_hidden.detach(), target_hidden.detach()
    else:
        online_hidden = target_hidden = None

    main = slice(cfg.burn_in, None)
    main_states = {name: value[:, main] for name, value in states.items()}
    main_next_states = {name: value[:, main] for name, value in next_states.items()}
    q_values, _, _ = online_net.forward_sequence(main_states, prev_actions[:, main], prev_rewards[:, main], online_hidden)
    selected_q = q_values.gather(-1, actions[:, main].unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        online_next, _, _ = online_net.forward_sequence(
            main_next_states, next_prev_actions[:, main], next_prev_rewards[:, main], online_hidden
        )
        target_next, _, _ = target_net.forward_sequence(
            main_next_states, next_prev_actions[:, main], next_prev_rewards[:, main], target_hidden
        )
        next_action = torch.argmax(online_next, dim=-1, keepdim=True)
        next_q = target_next.gather(-1, next_action).squeeze(-1)
        targets, target_available = n_step_targets(
            rewards[:, main], dones[:, main], valid[:, main], next_q, cfg.gamma, cfg.n_step_return
        )

    mask = (valid[:, main] > 0.5) & target_available
    if not bool(mask.any().item()):
        return 0.0, 0.0, 0.0
    td_error = selected_q - targets
    q_loss = nn.functional.smooth_l1_loss(selected_q[mask], targets[mask])
    teacher_mask = valid[:, main] > 0.5
    teacher_loss = nn.functional.cross_entropy(q_values[teacher_mask], teacher_actions[:, main][teacher_mask])
    total_loss = q_loss + imitation_weight(current_step, cfg) * teacher_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite v10 loss")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_gradient_norm)
    optimizer.step()
    per_sequence = (
        torch.sum(torch.abs(td_error.detach()) * mask.to(td_error.dtype), dim=1)
        / torch.clamp(mask.sum(dim=1), min=1)
    ).cpu().numpy()
    replay.update_priorities(np.asarray(batch["episode_indices"]), per_sequence)
    return float(total_loss.item()), float(q_loss.item()), float(teacher_loss.item())


def choose_action(
    net: MemoryGoalDuelingQNetwork,
    state: Dict[str, np.ndarray],
    prev_action: np.ndarray,
    prev_reward: float,
    hidden: torch.Tensor | None,
    teacher_action: int,
    epsilon: float,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[int, torch.Tensor]:
    state_t = {name: torch.from_numpy(value).unsqueeze(0).unsqueeze(0).to(device) for name, value in state.items()}
    prev_action_t = torch.from_numpy(prev_action).view(1, 1, -1).to(device)
    prev_reward_t = torch.tensor([[[prev_reward]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, next_hidden, _ = net.forward_sequence(state_t, prev_action_t, prev_reward_t, hidden)
        greedy_action = int(torch.argmax(q[:, -1], dim=-1).item())
    if random.random() < epsilon:
        action = int(teacher_action) if random.random() < cfg.teacher_probability else random.randrange(N_ACTIONS)
    else:
        action = greedy_action
    return action, next_hidden.detach()


def revisit_penalty(streak: int, cfg: TrainConfig) -> float:
    """Return the capped exponential penalty for a consecutive revisit streak."""
    if streak <= 0:
        return 0.0
    penalty = float(cfg.revisit_penalty_initial) * (float(cfg.revisit_penalty_growth) ** (int(streak) - 1))
    return float(min(penalty, float(cfg.revisit_penalty_cap)))


def learning_reward(
    late_phase: bool,
    new_seen: int,
    new_visit: bool,
    current_revisit_penalty: float,
    cfg: TrainConfig,
) -> float:
    if not late_phase:
        return float(
            cfg.early_seen_cell_reward * new_seen
            + cfg.early_visit_reward * float(new_visit)
            - cfg.step_cost
            - current_revisit_penalty
        )
    return float(cfg.late_visit_reward * float(new_visit) - cfg.step_cost - current_revisit_penalty)


def run_episode(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: MemoryGoalDuelingQNetwork,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
    training: bool,
    global_step: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, float], int]:
    obs, reset_info = env.reset(seed=seed)
    memory = AgentMemory(env.config.grid_size, env.config.patch_size, cfg)
    memory.reset(obs)
    teacher_action = memory.teacher_action()
    state = state_from_observation(obs, env.config, switches, memory)
    hidden = None
    previous_action = np.zeros((N_ACTIONS,), dtype=np.float32)
    previous_reward = 0.0
    done = False
    transitions: List[Dict[str, object]] = []
    objective_reward = 0.0
    raw_reward_total = 0.0
    new_tiles = 0
    revisit_streak = 0
    max_revisit_streak = 0
    revisit_penalty_total = 0.0
    while not done:
        late_phase_before = memory.in_late_phase()
        if training:
            action, hidden = choose_action(
                net, state, previous_action, previous_reward, hidden, teacher_action,
                linear_epsilon(global_step, cfg), cfg, device,
            )
        else:
            state_t = {name: torch.from_numpy(value).unsqueeze(0).unsqueeze(0).to(device) for name, value in state.items()}
            previous_action_t = torch.from_numpy(previous_action).view(1, 1, -1).to(device)
            previous_reward_t = torch.tensor([[[previous_reward]]], dtype=torch.float32, device=device)
            with torch.no_grad():
                q, hidden, _ = net.forward_sequence(state_t, previous_action_t, previous_reward_t, hidden)
                action = int(torch.argmax(q[:, -1], dim=-1).item())
            hidden = hidden.detach()
        next_obs, raw_reward, terminated, truncated, info = env.step(action, switches)
        new_seen = memory.update_after_action(action, next_obs)
        new_visit = float(info.get("reward_terms", {}).get("explore_reward", 0.0)) > 0.0
        wall_hit = float(info.get("reward_terms", {}).get("wall_penalty", 0.0)) < 0.0
        current_revisit_penalty = 0.0
        if action == ACTION_FORWARD and not wall_hit:
            if new_visit:
                revisit_streak = 0
            else:
                revisit_streak += 1
                current_revisit_penalty = revisit_penalty(revisit_streak, cfg)
        max_revisit_streak = max(max_revisit_streak, revisit_streak)
        revisit_penalty_total += current_revisit_penalty
        reward = learning_reward(late_phase_before, new_seen, new_visit, current_revisit_penalty, cfg)
        next_teacher_action = memory.teacher_action()
        next_state = state_from_observation(next_obs, env.config, switches, memory)
        done = bool(terminated or truncated)
        if training:
            transitions.append({
                "state": copy_state(state),
                "prev_action": previous_action.copy(),
                "prev_reward": float(previous_reward),
                "action": int(action),
                "reward": float(reward),
                "next_state": copy_state(next_state),
                "done": float(done),
                "next_prev_action": onehot_action(action),
                "next_prev_reward": float(reward),
                "teacher_action": int(teacher_action),
            })
        objective_reward += reward
        raw_reward_total += float(raw_reward)
        new_tiles += int(new_visit)
        state = next_state
        previous_action = onehot_action(action)
        previous_reward = float(reward)
        obs = next_obs
        teacher_action = next_teacher_action
        global_step += 1
    metrics = {
        "coverage": float(info.get("coverage", reset_info.get("coverage", 0.0))),
        "seen": memory.seen_fraction,
        "phase_reached": float(memory.in_late_phase()),
        "length": float(info.get("steps", 0)),
        "new_tile_rate": float(new_tiles / max(1, int(info.get("steps", 0)))),
        "max_revisit_streak": float(max_revisit_streak),
        "revisit_penalty_total": float(revisit_penalty_total),
        "objective_reward": float(objective_reward),
        "raw_reward": float(raw_reward_total),
    }
    return transitions, metrics, global_step


@torch.no_grad()
def evaluate_policy(
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    net: MemoryGoalDuelingQNetwork,
    cfg: TrainConfig,
    device: torch.device,
    episodes: int,
    seed_start: int,
    prefix: str,
) -> Dict[str, float]:
    values: Dict[str, List[float]] = {
        "coverage": [],
        "seen": [],
        "phase_reached": [],
        "length": [],
        "new_tile_rate": [],
        "max_revisit_streak": [],
        "revisit_penalty_total": [],
        "objective_reward": [],
        "raw_reward": [],
    }
    net.eval()
    for offset in range(int(episodes)):
        _, episode_metrics, _ = run_episode(env, switches, net, cfg, device, seed_start + offset, training=False)
        for name in values:
            values[name].append(float(episode_metrics[name]))
    return {f"{prefix}_{name}_mean": float(np.mean(items)) for name, items in values.items()}


def selection_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    # The target is lexicographic: reach 80% sensing first, then visit more cells.
    seen = float(metrics.get("eval_seen_mean", -float("inf")))
    coverage = float(metrics.get("eval_coverage_mean", -float("inf")))
    phase_reached = seen >= 0.80
    return (
        float(phase_reached),
        coverage if phase_reached else seen,
        coverage,
        float(metrics.get("eval_new_tile_rate_mean", -float("inf"))),
    )


def checkpoint_payload(
    net: MemoryGoalDuelingQNetwork,
    target_net: MemoryGoalDuelingQNetwork | None,
    optimizer: optim.Optimizer | None,
    cfg: TrainConfig,
    env: SensoryGridEnv,
    switches: ObservationSwitches,
    episode: int,
    global_step: int,
    eval_metrics: Dict[str, float] | None = None,
    holdout_metrics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model_state_dict": net.state_dict(),
        "model_arch": MODEL_ARCH,
        "model_kwargs": {
            "patch_size": env.config.patch_size,
            "map_size": env.config.grid_size,
            "num_actions": N_ACTIONS,
            "scalar_state_dim": SCALAR_STATE_DIM,
            "memory_map_embed_dim": cfg.memory_map_embed_dim,
            "conv_channels_1": cfg.conv_channels_1,
            "conv_channels_2": cfg.conv_channels_2,
            "vision_embed_dim": cfg.vision_embed_dim,
            "scalar_patch_embed_dim": cfg.scalar_patch_embed_dim,
            "scalar_state_embed_dim": cfg.scalar_state_embed_dim,
            "obs_embed_dim": cfg.obs_embed_dim,
            "gru_hidden_dim": cfg.gru_hidden_dim,
            "head_hidden_dim": cfg.head_hidden_dim,
            "uses_health_input": False,
            "uses_agent_memory_map": True,
            "uses_goal_features": True,
        },
        "num_actions": N_ACTIONS,
        "switches": asdict(switches),
        "env_config": asdict(env.config),
        "train_config": asdict(cfg),
        "episode": int(episode),
        "global_step": int(global_step),
        "eval_metrics": eval_metrics or {},
        "holdout_eval_metrics": holdout_metrics or {},
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
    }
    if target_net is not None:
        payload["target_model_state_dict"] = target_net.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def build_network(env: SensoryGridEnv, cfg: TrainConfig) -> MemoryGoalDuelingQNetwork:
    return MemoryGoalDuelingQNetwork(
        patch_size=env.config.patch_size,
        map_size=env.config.grid_size,
        num_actions=N_ACTIONS,
        cfg=model_config(cfg),
        memory_map_embed_dim=cfg.memory_map_embed_dim,
    )


def write_config(save_dir: Path, cfg: TrainConfig, env: SensoryGridEnv) -> None:
    with open(save_dir / "train_config.txt", "w", encoding="utf-8") as handle:
        handle.write(f"model_arch: {MODEL_ARCH}\n")
        for name, value in asdict(cfg).items():
            handle.write(f"{name}: {value}\n")
        handle.write("\n[env_config]\n")
        for name, value in asdict(env.config).items():
            handle.write(f"{name}: {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal-conditioned memory-map Double DQN trainer")
    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument("--save_dir", type=str, default=TrainConfig.save_dir)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--eval_episodes", type=int, default=TrainConfig.eval_episodes)
    parser.add_argument("--holdout_eval_episodes", type=int, default=TrainConfig.holdout_eval_episodes)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    cfg = TrainConfig(
        episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        eval_episodes=args.eval_episodes,
        holdout_eval_episodes=args.holdout_eval_episodes,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    device = choose_device(cfg.device)
    env, switches = build_training_env(cfg)
    eval_env, eval_switches = build_training_env(cfg)
    online_net = build_network(env, cfg).to(device)
    target_net = build_network(env, cfg).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(online_net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    replay = PrioritizedSequenceReplay(cfg.replay_capacity_episodes, cfg.replay_priority_alpha, cfg.replay_priority_epsilon)
    start_episode, global_step = 1, 0
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=device)
        if payload.get("model_arch") != MODEL_ARCH:
            raise ValueError("Only goal-conditioned memory DQN checkpoints can be resumed.")
        online_net.load_state_dict(payload["model_state_dict"])
        target_net.load_state_dict(payload.get("target_model_state_dict", payload["model_state_dict"]))
        if isinstance(payload.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_episode = int(payload.get("episode", 0)) + 1
        global_step = int(payload.get("global_step", 0))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_config(save_dir, cfg, env)
    log_path = save_dir / "training_log.csv"
    fields = [
        "episode", "global_step", "epsilon", "objective_reward", "raw_game_reward", "coverage", "seen",
        "phase_reached", "length", "new_tile_rate", "max_revisit_streak", "revisit_penalty_total",
        "mean_loss", "mean_q_loss", "mean_imitation_loss",
        "eval_coverage_mean", "eval_seen_mean", "eval_phase_reached_mean", "eval_new_tile_rate_mean",
        "eval_max_revisit_streak_mean", "eval_revisit_penalty_total_mean",
        "holdout_coverage_mean", "holdout_seen_mean", "holdout_phase_reached_mean", "holdout_new_tile_rate_mean",
        "holdout_max_revisit_streak_mean", "holdout_revisit_penalty_total_mean",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_key = (-float("inf"),) * 4
    for episode in range(start_episode, cfg.episodes + 1):
        online_net.train()
        transitions, metrics, global_step = run_episode(
            env, switches, online_net, cfg, device, cfg.seed + episode, training=True, global_step=global_step
        )
        replay.add_episode(transitions)
        losses, q_losses, imitation_losses = [], [], []
        if len(replay) >= cfg.train_after_episodes:
            for _ in range(cfg.train_updates_per_episode):
                loss, q_loss, imitation_loss = train_step(
                    online_net, target_net, optimizer, replay, device, cfg, global_step
                )
                losses.append(loss)
                q_losses.append(q_loss)
                imitation_losses.append(imitation_loss)
        with torch.no_grad():
            for target_parameter, online_parameter in zip(target_net.parameters(), online_net.parameters()):
                target_parameter.data.mul_(1.0 - cfg.target_soft_tau).add_(online_parameter.data, alpha=cfg.target_soft_tau)
        if episode % cfg.target_hard_sync_every_episodes == 0:
            target_net.load_state_dict(online_net.state_dict())

        do_eval = episode == 1 or episode % cfg.eval_every == 0 or episode == cfg.episodes
        if do_eval:
            eval_metrics = evaluate_policy(eval_env, eval_switches, online_net, cfg, device, cfg.eval_episodes, cfg.eval_seed_start, "eval")
            holdout_metrics = evaluate_policy(
                eval_env, eval_switches, online_net, cfg, device, cfg.holdout_eval_episodes, cfg.holdout_eval_seed_start, "holdout"
            )
        else:
            eval_metrics = {
                f"eval_{name}_mean": float("nan")
                for name in ("coverage", "seen", "phase_reached", "new_tile_rate", "max_revisit_streak", "revisit_penalty_total")
            }
            holdout_metrics = {
                f"holdout_{name}_mean": float("nan")
                for name in ("coverage", "seen", "phase_reached", "new_tile_rate", "max_revisit_streak", "revisit_penalty_total")
            }
        row = {
            "episode": episode,
            "global_step": global_step,
            "epsilon": linear_epsilon(global_step, cfg),
            "objective_reward": metrics["objective_reward"],
            "raw_game_reward": metrics["raw_reward"],
            "coverage": metrics["coverage"],
            "seen": metrics["seen"],
            "phase_reached": metrics["phase_reached"],
            "length": metrics["length"],
            "new_tile_rate": metrics["new_tile_rate"],
            "max_revisit_streak": metrics["max_revisit_streak"],
            "revisit_penalty_total": metrics["revisit_penalty_total"],
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "mean_q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "mean_imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
            **{name: eval_metrics[name] for name in eval_metrics if name in fields},
            **{name: holdout_metrics[name] for name in holdout_metrics if name in fields},
        }
        print(
            f"episode={episode:04d} step={global_step:06d} eps={row['epsilon']:.3f} "
            f"objective={row['objective_reward']:+.3f} coverage={row['coverage']:.3f} seen={row['seen']:.3f} "
            f"loss={row['mean_loss']:.4f} eval_cov={eval_metrics['eval_coverage_mean']:.3f} "
            f"eval_seen={eval_metrics['eval_seen_mean']:.3f} holdout_cov={holdout_metrics['holdout_coverage_mean']:.3f}",
            flush=True,
        )
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)

        if episode % cfg.last_checkpoint_every_episodes == 0 or episode == cfg.episodes:
            torch.save(
                checkpoint_payload(online_net, target_net, optimizer, cfg, env, switches, episode, global_step, eval_metrics, holdout_metrics),
                save_dir / "last_train_state.pt",
            )
        if do_eval:
            payload = checkpoint_payload(online_net, None, None, cfg, env, switches, episode, global_step, eval_metrics, holdout_metrics)
            torch.save(payload, save_dir / f"ckpt_ep{episode:04d}.pt")
            if selection_key(eval_metrics) > best_key:
                best_key = selection_key(eval_metrics)
                torch.save(payload, save_dir / "best_exploration_model.pt")
                torch.save(payload, save_dir / "best_model.pt")
    torch.save(
        checkpoint_payload(online_net, None, None, cfg, env, switches, cfg.episodes, global_step),
        save_dir / "final_model.pt",
    )
    print(f"Training finished. Logs and checkpoints saved to: {save_dir.resolve()}")


def build_model_from_checkpoint(payload: Dict[str, Any], device: str = "cpu") -> MemoryGoalDuelingQNetwork:
    if payload.get("model_arch") != MODEL_ARCH:
        raise ValueError(f"Not a goal-conditioned memory DQN checkpoint: {payload.get('model_arch')}")
    kwargs = payload.get("model_kwargs", {})
    cfg = ModelConfig(
        conv_channels_1=int(kwargs["conv_channels_1"]),
        conv_channels_2=int(kwargs["conv_channels_2"]),
        vision_embed_dim=int(kwargs["vision_embed_dim"]),
        scalar_patch_embed_dim=int(kwargs["scalar_patch_embed_dim"]),
        scalar_state_embed_dim=int(kwargs["scalar_state_embed_dim"]),
        obs_embed_dim=int(kwargs["obs_embed_dim"]),
        gru_hidden_dim=int(kwargs["gru_hidden_dim"]),
        head_hidden_dim=int(kwargs["head_hidden_dim"]),
    )
    net = MemoryGoalDuelingQNetwork(
        patch_size=int(kwargs["patch_size"]), map_size=int(kwargs["map_size"]), num_actions=int(kwargs["num_actions"]),
        cfg=cfg, memory_map_embed_dim=int(kwargs["memory_map_embed_dim"]), scalar_state_dim=int(kwargs["scalar_state_dim"]),
    )
    net.load_state_dict(payload["model_state_dict"], strict=True)
    net.to(choose_device(device)).eval()
    return net


def reset_runtime_context(context: Dict[str, Any] | None = None, device: str | None = None) -> Dict[str, Any]:
    return base_reset_runtime_context(context, device)


def predict_action_for_gui(
    net: nn.Module,
    obs: Dict[str, Any],
    env_cfg: EnvConfig,
    switches: ObservationSwitches,
    runtime_context: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    ctx = dict(runtime_context or reset_runtime_context())
    cfg = TrainConfig()
    memory = ctx.get("memory")
    if not isinstance(memory, AgentMemory):
        memory = AgentMemory(env_cfg.grid_size, env_cfg.patch_size, cfg)
        memory.reset(obs)
    else:
        pending_action = ctx.get("pending_action")
        if pending_action is not None:
            memory.update_after_action(int(pending_action), obs)
            ctx["pending_action"] = None
    memory.teacher_action()
    state = state_from_observation(obs, env_cfg, switches, memory)
    device = choose_device(str(ctx.get("device", "cpu")))
    state_t = {name: torch.from_numpy(value).unsqueeze(0).unsqueeze(0).to(device) for name, value in state.items()}
    previous_action = np.asarray(ctx.get("prev_action", np.zeros(N_ACTIONS)), dtype=np.float32)
    previous_reward = float(ctx.get("prev_reward", 0.0))
    previous_action_t = torch.from_numpy(previous_action).view(1, 1, -1).to(device)
    previous_reward_t = torch.tensor([[[previous_reward]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        q, hidden, _ = net.forward_sequence(state_t, previous_action_t, previous_reward_t, ctx.get("hidden"))
        action = int(torch.argmax(q[:, -1], dim=-1).item())
    ctx["memory"] = memory
    ctx["hidden"] = hidden.detach()
    ctx["device"] = str(device)
    return action, ctx


def update_runtime_context_after_env_step(
    runtime_context: Dict[str, Any] | None,
    action: int,
    reward: float,
    done: bool = False,
) -> Dict[str, Any]:
    if done:
        return reset_runtime_context(runtime_context)
    ctx = dict(runtime_context or reset_runtime_context())
    ctx["prev_action"] = onehot_action(action)
    ctx["prev_reward"] = float(reward)
    ctx["pending_action"] = int(action)
    return ctx


def get_gui_interface_spec() -> Dict[str, Any]:
    return {
        "trainer_name": TRAINER_DISPLAY_NAME,
        "trainer_gui_interface_version": TRAINER_GUI_INTERFACE_VERSION,
        "checkpoint_load_order": ["trainer_module", "checkpoint"],
        "env_module": "sensory_grid_env_v5",
        "model_family": MODEL_ARCH,
        "default_switches": asdict(build_default_switches()),
    }


if __name__ == "__main__":
    main()

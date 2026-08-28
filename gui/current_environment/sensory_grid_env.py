
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import numpy as np

OBJ_EMPTY = 0
OBJ_FIRE = 1
OBJ_MEAT = 2
OBJ_FLOWER = 3
OBJ_GLASS = 4
OBJ_ICE = 5

OBJ_LABELS = {
    OBJ_EMPTY: "Empty",
    OBJ_FIRE: "Fire",
    OBJ_MEAT: "Meat",
    OBJ_FLOWER: "Flower",
    OBJ_GLASS: "Glass",
    OBJ_ICE: "Ice",
}

OBJ_SYMBOLS = {
    OBJ_EMPTY: "",
    OBJ_FIRE: "🔥",
    OBJ_MEAT: "🍖",
    OBJ_FLOWER: "🌸",
    OBJ_GLASS: "🔷",
    OBJ_ICE: "🧊",
}

ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
N_ACTIONS = 3

ACTION_LABELS = {
    ACTION_FORWARD: "forward",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

DIR_VECTORS = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1),   # left
}

DIR_SYMBOLS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}


@dataclass
class EnvConfig:
    grid_size: int = 15
    patch_size: int = 5
    init_health: int = 10
    max_health: int = 10
    init_energy: float = 10.0
    max_energy: float = 10.0
    max_steps: int = 250

    n_fire: int = 2
    n_ice: int = 1
    n_meat: int = 3
    n_flower: int = 2
    n_glass: int = 2

    step_penalty: float = -0.01
    explore_reward: float = 0.08
    wall_penalty: float = -0.05
    damage_reward_scale: float = 0.15
    heal_reward_scale: float = 0.08
    flower_penalty: float = -0.03
    death_penalty: float = 1.0
    survival_bonus: float = 0.0

    glass_damage: int = 2
    meat_heal: int = 2
    fire_damage: int = 3
    ice_damage: int = 3

    ambient_temperature_c: float = 22.0
    comfort_low_c: float = 18.0
    comfort_high_c: float = 24.0
    fire_temp_delta_amp: float = 13.0
    ice_temp_delta_amp: float = 13.0
    temp_sigma: float = 2.1

    meat_smell_amp: float = 0.75
    flower_smell_amp: float = 1.00
    smell_sigma_meat: float = 2.2
    smell_sigma_flower: float = 2.8

    time_energy_cost: float = 0.20
    forward_energy_cost: float = 0.60
    turn_energy_cost: float = 0.10
    thermal_extra_energy_max: float = 0.50
    discomfort_temp_scale_c: float = 10.0

    energy_reward_scale: float = 0.025
    discomfort_reward_scale: float = 0.04
    no_move_penalty: float = 0.01
    turn_streak_penalty: float = 0.005


@dataclass
class ObservationSwitches:
    include_vision: bool = True
    include_temperature: bool = False
    include_smell: bool = False
    include_temperature_patch: bool = True
    include_smell_patch: bool = True
    include_visited_memory: bool = True
    include_hazard_memory: bool = True


class SensoryGridEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

        g = config.grid_size
        self.grid = np.zeros((g, g), dtype=np.int32)
        self.temperature_field_c = np.full((g, g), config.ambient_temperature_c, dtype=np.float32)
        self.smell_field = np.zeros((g, g), dtype=np.float32)
        self.visited_map = np.zeros((g, g), dtype=np.float32)
        self.hazard_memory = np.zeros((g, g), dtype=np.float32)

        self.agent_pos = (0, 0)
        self.direction = 0
        self.health = config.init_health
        self.energy = config.init_energy
        self.steps = 0
        self.last_reward = 0.0
        self.last_event = "Episode reset."
        self.last_discomfort = 0.0
        self.last_thermal_extra = 0.0
        self.last_time_base_cost = config.time_energy_cost
        self.last_forward_extra_cost = 0.0
        self.last_turn_extra_cost = 0.0
        self.last_energy_feedback_penalty = 0.0
        self.last_discomfort_penalty = 0.0
        self.last_reward_terms: Dict[str, float] = {}
        self.consecutive_no_move_steps = 0
        self.consecutive_turn_steps = 0
        self.terminated = False
        self.truncated = False
        self.object_counts = {
            OBJ_FIRE: 0,
            OBJ_ICE: 0,
            OBJ_MEAT: 0,
            OBJ_FLOWER: 0,
            OBJ_GLASS: 0,
        }
        self.seed_value = None

    def seed(self, seed: int | None = None) -> None:
        self.seed_value = seed
        self.rng.seed(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> Tuple[dict, dict]:
        if seed is not None:
            self.seed(seed)

        c = self.config
        c.max_health = c.init_health
        c.max_energy = c.init_energy

        g = c.grid_size
        self.grid = np.zeros((g, g), dtype=np.int32)
        self.temperature_field_c = np.full((g, g), c.ambient_temperature_c, dtype=np.float32)
        self.smell_field = np.zeros((g, g), dtype=np.float32)
        self.visited_map = np.zeros((g, g), dtype=np.float32)
        self.hazard_memory = np.zeros((g, g), dtype=np.float32)

        self.health = c.init_health
        self.energy = c.max_energy
        self.steps = 0
        self.last_reward = 0.0
        self.last_event = "Episode reset."
        self.last_discomfort = 0.0
        self.last_thermal_extra = 0.0
        self.last_time_base_cost = c.time_energy_cost
        self.last_forward_extra_cost = 0.0
        self.last_turn_extra_cost = 0.0
        self.last_energy_feedback_penalty = 0.0
        self.last_discomfort_penalty = 0.0
        self.last_reward_terms = {}
        self.consecutive_no_move_steps = 0
        self.consecutive_turn_steps = 0
        self.terminated = False
        self.truncated = False

        centre = g // 2
        self.agent_pos = (centre, centre)
        self.direction = self.rng.choice([0, 1, 2, 3])

        forbidden = {self.agent_pos}
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = centre + dr
                cc = centre + dc
                if 0 <= rr < g and 0 <= cc < g:
                    forbidden.add((rr, cc))

        placements = [
            (OBJ_FIRE, c.n_fire),
            (OBJ_ICE, c.n_ice),
            (OBJ_MEAT, c.n_meat),
            (OBJ_FLOWER, c.n_flower),
            (OBJ_GLASS, c.n_glass),
        ]
        for obj_type, count in placements:
            for _ in range(count):
                pos = self._sample_empty_position(forbidden)
                self.grid[pos] = obj_type
                forbidden.add(pos)

        self._recount_objects()
        self._rebuild_scalar_fields()
        self.visited_map[self.agent_pos] = 1.0

        obs = self.get_observation(ObservationSwitches())
        info = self._build_info(health_delta=0, contacted=OBJ_EMPTY)
        return obs, info

    def step(self, action: int, switches: ObservationSwitches | None = None) -> Tuple[dict, float, bool, bool, dict]:
        if switches is None:
            switches = ObservationSwitches()

        if self.terminated or self.truncated:
            obs = self.get_observation(switches)
            return obs, 0.0, self.terminated, self.truncated, self._build_info(0, OBJ_EMPTY)

        c = self.config
        reward_terms = {
            "step_penalty": float(c.step_penalty),
            "explore_reward": 0.0,
            "wall_penalty": 0.0,
            "contact_damage_penalty": 0.0,
            "heal_reward": 0.0,
            "flower_penalty": 0.0,
            "energy_feedback_penalty": 0.0,
            "discomfort_penalty": 0.0,
            "fatigue_penalty": 0.0,
            "death_penalty": 0.0,
            "survival_bonus": 0.0,
            "no_move_streak_penalty": 0.0,
            "turn_streak_penalty": 0.0,
        }
        reward = reward_terms["step_penalty"]
        health_before = self.health
        contacted = OBJ_EMPTY
        event_parts: List[str] = []

        attempted_forward = action == ACTION_FORWARD
        turn_cost = 0.0
        did_move = False
        did_turn = False

        if action == ACTION_LEFT:
            self.direction = (self.direction - 1) % 4
            turn_cost = c.turn_energy_cost
            did_turn = True
            event_parts.append("Turned left")
        elif action == ACTION_RIGHT:
            self.direction = (self.direction + 1) % 4
            turn_cost = c.turn_energy_cost
            did_turn = True
            event_parts.append("Turned right")
        elif action == ACTION_FORWARD:
            next_r, next_c = self._front_position()
            if not self._in_bounds(next_r, next_c):
                reward += c.wall_penalty
                reward_terms["wall_penalty"] += c.wall_penalty
                event_parts.append("Bumped into wall")
            else:
                did_move = True
                self.agent_pos = (next_r, next_c)
                first_visit = self.visited_map[self.agent_pos] < 0.5
                self.visited_map[self.agent_pos] = 1.0
                if first_visit:
                    reward += c.explore_reward
                    reward_terms["explore_reward"] += c.explore_reward
                    event_parts.append("Entered new tile")

                contacted = int(self.grid[self.agent_pos])
                if contacted == OBJ_FIRE:
                    self.health -= c.fire_damage
                    penalty = c.fire_damage * c.damage_reward_scale
                    reward -= penalty
                    reward_terms["contact_damage_penalty"] -= penalty
                    self.hazard_memory[self.agent_pos] = 1.0
                    event_parts.append(f"Stepped on fire and took {c.fire_damage} damage")
                elif contacted == OBJ_ICE:
                    self.health -= c.ice_damage
                    penalty = c.ice_damage * c.damage_reward_scale
                    reward -= penalty
                    reward_terms["contact_damage_penalty"] -= penalty
                    self.hazard_memory[self.agent_pos] = 1.0
                    event_parts.append(f"Stepped on ice and took {c.ice_damage} damage")
                elif contacted == OBJ_MEAT:
                    heal = min(c.meat_heal, c.max_health - self.health)
                    self.health += heal
                    bonus = heal * c.heal_reward_scale
                    reward += bonus
                    reward_terms["heal_reward"] += bonus
                    self.grid[self.agent_pos] = OBJ_EMPTY
                    self._recount_objects()
                    self._rebuild_scalar_fields()
                    event_parts.append(f"Ate meat and recovered {heal} health")
                elif contacted == OBJ_FLOWER:
                    reward += c.flower_penalty
                    reward_terms["flower_penalty"] += c.flower_penalty
                    self.grid[self.agent_pos] = OBJ_EMPTY
                    self._recount_objects()
                    self._rebuild_scalar_fields()
                    event_parts.append("Crushed a flower")
                elif contacted == OBJ_GLASS:
                    self.health -= c.glass_damage
                    penalty = c.glass_damage * c.damage_reward_scale
                    reward -= penalty
                    reward_terms["contact_damage_penalty"] -= penalty
                    self.hazard_memory[self.agent_pos] = 1.0
                    event_parts.append(f"Stepped on glass and took {c.glass_damage} damage")
                else:
                    event_parts.append("Moved onto empty ground")
        else:
            raise ValueError(f"Unknown action {action}")

        if did_move:
            self.consecutive_no_move_steps = 0
            self.consecutive_turn_steps = 0
        else:
            self.consecutive_no_move_steps += 1
            if did_turn:
                self.consecutive_turn_steps += 1
            else:
                self.consecutive_turn_steps = 0

        no_move_streak_penalty = max(0, self.consecutive_no_move_steps - 1) * c.no_move_penalty
        if no_move_streak_penalty > 0:
            reward -= no_move_streak_penalty
            reward_terms["no_move_streak_penalty"] -= no_move_streak_penalty
            event_parts.append(f"No-move streak penalty {no_move_streak_penalty:.3f}")

        turn_streak_penalty = max(0, self.consecutive_turn_steps - 1) * c.turn_streak_penalty
        if turn_streak_penalty > 0:
            reward -= turn_streak_penalty
            reward_terms["turn_streak_penalty"] -= turn_streak_penalty
            event_parts.append(f"Turn streak penalty {turn_streak_penalty:.3f}")

        discomfort, thermal_extra = self._temperature_cost_terms(self.agent_pos)
        time_cost = c.time_energy_cost
        forward_cost = c.forward_energy_cost if attempted_forward else 0.0
        total_energy_cost = time_cost + forward_cost + turn_cost + thermal_extra

        self.energy -= total_energy_cost

        energy_feedback_penalty = total_energy_cost * c.energy_reward_scale
        if energy_feedback_penalty > 0:
            reward -= energy_feedback_penalty
            reward_terms["energy_feedback_penalty"] -= energy_feedback_penalty
            event_parts.append(f"Metabolic cost {energy_feedback_penalty:.3f}")

        discomfort_penalty = discomfort * c.discomfort_reward_scale
        if discomfort_penalty > 0:
            reward -= discomfort_penalty
            reward_terms["discomfort_penalty"] -= discomfort_penalty
            event_parts.append(f"Thermal discomfort penalty {discomfort_penalty:.3f}")

        fatigue_losses = 0
        while self.energy <= 0.0 and self.health > 0:
            self.health -= 1
            self.energy += c.max_energy
            fatigue_losses += 1
        if fatigue_losses > 0:
            fatigue_penalty = fatigue_losses * c.damage_reward_scale
            reward -= fatigue_penalty
            reward_terms["fatigue_penalty"] -= fatigue_penalty
            event_parts.append(f"Fatigue consumed {fatigue_losses} health")

        self.steps += 1
        if self.health <= 0:
            self.terminated = True
            reward -= c.death_penalty
            reward_terms["death_penalty"] -= c.death_penalty
            event_parts.append("Health reached zero")
        if self.steps >= c.max_steps:
            self.truncated = True
            event_parts.append("Reached max steps")
            if not self.terminated and c.survival_bonus > 0:
                reward += c.survival_bonus
                reward_terms["survival_bonus"] += c.survival_bonus
                event_parts.append(f"Survival bonus {c.survival_bonus:.3f}")

        self.hazard_memory *= 0.985
        self.last_reward = float(reward)
        self.last_event = "; ".join(event_parts) if event_parts else "No event"
        self.last_discomfort = float(discomfort)
        self.last_thermal_extra = float(thermal_extra)
        self.last_time_base_cost = float(time_cost)
        self.last_forward_extra_cost = float(forward_cost)
        self.last_turn_extra_cost = float(turn_cost)
        self.last_energy_feedback_penalty = float(energy_feedback_penalty)
        self.last_discomfort_penalty = float(discomfort_penalty)
        self.last_reward_terms = reward_terms.copy()

        obs = self.get_observation(switches)
        info = self._build_info(health_delta=self.health - health_before, contacted=contacted)
        return obs, float(reward), self.terminated, self.truncated, info

    def get_observation(self, switches: ObservationSwitches) -> dict:
        patch_size = self.config.patch_size
        vision_patch = self._egocentric_patch(self.grid, self.agent_pos, self.direction, patch_size, fill_value=OBJ_EMPTY)
        temperature_patch_c = self._egocentric_patch(
            self.temperature_field_c, self.agent_pos, self.direction, patch_size, fill_value=self.config.ambient_temperature_c
        )
        smell_patch = self._egocentric_patch(self.smell_field, self.agent_pos, self.direction, patch_size, fill_value=0.0)
        visited_patch = self._egocentric_patch(self.visited_map, self.agent_pos, self.direction, patch_size, fill_value=0.0)
        hazard_patch = self._egocentric_patch(self.hazard_memory, self.agent_pos, self.direction, patch_size, fill_value=0.0)

        temperature_c = float(self.temperature_field_c[self.agent_pos])
        smell_scalar = float(self.smell_field[self.agent_pos])
        discomfort, thermal_extra = self._temperature_cost_terms(self.agent_pos)

        return {
            "vision": vision_patch.copy(),
            "vision_active": switches.include_vision,
            "temperature": temperature_c if switches.include_temperature else 0.0,
            "temperature_active": switches.include_temperature,
            "smell": smell_scalar if switches.include_smell else 0.0,
            "smell_active": switches.include_smell,
            "temperature_patch_c": temperature_patch_c.copy(),
            "temperature_patch_active": switches.include_temperature_patch,
            "smell_patch": smell_patch.copy(),
            "smell_patch_active": switches.include_smell_patch,
            "health": float(self.health),
            "health_norm": float(self.health / max(1, self.config.max_health)),
            "energy": float(self.energy),
            "energy_norm": float(self.energy / max(1e-6, self.config.max_energy)),
            "direction": int(self.direction),
            "direction_onehot": self._direction_onehot(),
            "visited_patch": visited_patch.copy(),
            "visited_active": switches.include_visited_memory,
            "hazard_patch": hazard_patch.copy(),
            "hazard_active": switches.include_hazard_memory,
            "discomfort": float(discomfort),
            "thermal_extra_this_tick": float(thermal_extra),
            "time_base_cost": float(self.config.time_energy_cost),
            "forward_extra_cost": float(self.config.forward_energy_cost),
            "turn_extra_cost": float(self.config.turn_energy_cost),
            "consecutive_no_move_steps": int(self.consecutive_no_move_steps),
            "consecutive_turn_steps": int(self.consecutive_turn_steps),
            "reward_terms": self.last_reward_terms.copy(),
            "agent_input": self.flatten_observation(switches),
        }

    def flatten_observation(self, switches: ObservationSwitches) -> np.ndarray:
        parts: List[np.ndarray] = []
        if switches.include_vision:
            patch_ids = self._egocentric_patch(self.grid, self.agent_pos, self.direction, self.config.patch_size, fill_value=OBJ_EMPTY)
            one_hot = np.eye(6, dtype=np.float32)[patch_ids.astype(np.int32)]
            parts.append(one_hot.reshape(-1))
        if switches.include_temperature:
            parts.append(np.array([self.temperature_field_c[self.agent_pos]], dtype=np.float32))
        if switches.include_smell:
            parts.append(np.array([self.smell_field[self.agent_pos]], dtype=np.float32))
        if switches.include_temperature_patch:
            parts.append(self.temperature_patch_normalized().reshape(-1))
        if switches.include_smell_patch:
            parts.append(self.smell_patch_clipped().reshape(-1))
        parts.append(np.array([self.health / max(1, self.config.max_health)], dtype=np.float32))
        parts.append(np.array([self.energy / max(1e-6, self.config.max_energy)], dtype=np.float32))
        parts.append(self._direction_onehot())
        if switches.include_visited_memory:
            visited_patch = self._egocentric_patch(self.visited_map, self.agent_pos, self.direction, self.config.patch_size, fill_value=0.0)
            parts.append(visited_patch.astype(np.float32).reshape(-1))
        if switches.include_hazard_memory:
            hazard_patch = self._egocentric_patch(self.hazard_memory, self.agent_pos, self.direction, self.config.patch_size, fill_value=0.0)
            parts.append(hazard_patch.astype(np.float32).reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def observation_dim(self, switches: ObservationSwitches) -> int:
        return int(self.flatten_observation(switches).shape[0])

    def vision_patch(self) -> np.ndarray:
        return self._egocentric_patch(self.grid, self.agent_pos, self.direction, self.config.patch_size, fill_value=OBJ_EMPTY)

    def temperature_patch_c(self) -> np.ndarray:
        return self._egocentric_patch(
            self.temperature_field_c, self.agent_pos, self.direction, self.config.patch_size, fill_value=self.config.ambient_temperature_c
        )

    def smell_patch_values(self) -> np.ndarray:
        return self._egocentric_patch(self.smell_field, self.agent_pos, self.direction, self.config.patch_size, fill_value=0.0)

    def visited_patch(self) -> np.ndarray:
        return self._egocentric_patch(self.visited_map, self.agent_pos, self.direction, self.config.patch_size, fill_value=0.0)

    def hazard_patch(self) -> np.ndarray:
        return self._egocentric_patch(self.hazard_memory, self.agent_pos, self.direction, self.config.patch_size, fill_value=0.0)

    def temperature_patch_normalized(self) -> np.ndarray:
        patch = self.temperature_patch_c().astype(np.float32)
        scale = max(1.0, self.config.fire_temp_delta_amp, self.config.ice_temp_delta_amp)
        return np.clip((patch - self.config.ambient_temperature_c) / scale, -1.0, 1.0)

    def smell_patch_clipped(self) -> np.ndarray:
        return np.clip(self.smell_patch_values().astype(np.float32), 0.0, 1.0)

    def reveal_world_ids(self) -> np.ndarray:
        return self.grid.copy()

    def reveal_temperature_field_c(self) -> np.ndarray:
        return self.temperature_field_c.copy()

    def reveal_smell_field(self) -> np.ndarray:
        return self.smell_field.copy()

    def current_scalars(self) -> Dict[str, float]:
        discomfort, thermal_extra = self._temperature_cost_terms(self.agent_pos)
        return {
            "temperature_c": float(self.temperature_field_c[self.agent_pos]),
            "temperature_min_c": float(np.min(self.temperature_field_c)),
            "temperature_max_c": float(np.max(self.temperature_field_c)),
            "smell": float(self.smell_field[self.agent_pos]),
            "smell_max": float(max(1.0, np.max(self.smell_field))),
            "health": float(self.health),
            "energy": float(self.energy),
            "coverage": float(np.mean(self.visited_map > 0.5)),
            "discomfort": float(discomfort),
            "thermal_extra_this_tick": float(thermal_extra),
            "time_base_cost": float(self.config.time_energy_cost),
            "forward_extra_cost": float(self.config.forward_energy_cost),
            "turn_extra_cost": float(self.config.turn_energy_cost),
            "consecutive_no_move_steps": float(self.consecutive_no_move_steps),
            "consecutive_turn_steps": float(self.consecutive_turn_steps),
            "episode_done": float(self.terminated or self.truncated),
        }

    def _build_info(self, health_delta: int, contacted: int) -> dict:
        return {
            "health": self.health,
            "energy": self.energy,
            "health_delta": health_delta,
            "coverage": float(np.mean(self.visited_map > 0.5)),
            "last_event": self.last_event,
            "last_reward": self.last_reward,
            "contacted": contacted,
            "contacted_label": OBJ_LABELS[contacted],
            "steps": self.steps,
            "object_counts": self.object_counts.copy(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "discomfort": self.last_discomfort,
            "thermal_extra_this_tick": self.last_thermal_extra,
            "time_base_cost": self.last_time_base_cost,
            "forward_extra_cost": self.last_forward_extra_cost,
            "turn_extra_cost": self.last_turn_extra_cost,
            "consecutive_no_move_steps": self.consecutive_no_move_steps,
            "consecutive_turn_steps": self.consecutive_turn_steps,
            "energy_feedback_penalty": self.last_energy_feedback_penalty,
            "discomfort_penalty": self.last_discomfort_penalty,
            "reward_terms": self.last_reward_terms.copy(),
        }

    def _recount_objects(self) -> None:
        self.object_counts = {
            OBJ_FIRE: int(np.sum(self.grid == OBJ_FIRE)),
            OBJ_ICE: int(np.sum(self.grid == OBJ_ICE)),
            OBJ_MEAT: int(np.sum(self.grid == OBJ_MEAT)),
            OBJ_FLOWER: int(np.sum(self.grid == OBJ_FLOWER)),
            OBJ_GLASS: int(np.sum(self.grid == OBJ_GLASS)),
        }

    def _sample_empty_position(self, forbidden: set[Tuple[int, int]]) -> Tuple[int, int]:
        candidates = [
            (r, c)
            for r in range(self.config.grid_size)
            for c in range(self.config.grid_size)
            if (r, c) not in forbidden and self.grid[r, c] == OBJ_EMPTY
        ]
        if not candidates:
            raise RuntimeError("No empty position available for object placement.")
        return self.rng.choice(candidates)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.config.grid_size and 0 <= c < self.config.grid_size

    def _front_position(self) -> Tuple[int, int]:
        dr, dc = DIR_VECTORS[self.direction]
        return self.agent_pos[0] + dr, self.agent_pos[1] + dc

    def _direction_onehot(self) -> np.ndarray:
        out = np.zeros(4, dtype=np.float32)
        out[self.direction] = 1.0
        return out

    def _temperature_cost_terms(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        temp = float(self.temperature_field_c[pos])
        if self.config.comfort_low_c <= temp <= self.config.comfort_high_c:
            discomfort = 0.0
        else:
            nearest = self.config.comfort_low_c if temp < self.config.comfort_low_c else self.config.comfort_high_c
            deviation = abs(temp - nearest)
            discomfort = min(deviation / max(1e-6, self.config.discomfort_temp_scale_c), 1.0)
        thermal_extra = discomfort * self.config.thermal_extra_energy_max
        return float(discomfort), float(thermal_extra)

    def _rebuild_scalar_fields(self) -> None:
        c = self.config
        g = c.grid_size
        rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")

        temp_field = np.full((g, g), c.ambient_temperature_c, dtype=np.float32)
        smell_field = np.zeros((g, g), dtype=np.float32)

        for r, col in zip(*np.where(self.grid == OBJ_FIRE)):
            d2 = (rr - r) ** 2 + (cc - col) ** 2
            temp_field += c.fire_temp_delta_amp * np.exp(-d2 / (2.0 * c.temp_sigma ** 2))

        for r, col in zip(*np.where(self.grid == OBJ_ICE)):
            d2 = (rr - r) ** 2 + (cc - col) ** 2
            temp_field -= c.ice_temp_delta_amp * np.exp(-d2 / (2.0 * c.temp_sigma ** 2))

        for r, col in zip(*np.where(self.grid == OBJ_MEAT)):
            d2 = (rr - r) ** 2 + (cc - col) ** 2
            smell_field += c.meat_smell_amp * np.exp(-d2 / (2.0 * c.smell_sigma_meat ** 2))

        for r, col in zip(*np.where(self.grid == OBJ_FLOWER)):
            d2 = (rr - r) ** 2 + (cc - col) ** 2
            smell_field += c.flower_smell_amp * np.exp(-d2 / (2.0 * c.smell_sigma_flower ** 2))

        self.temperature_field_c = temp_field.astype(np.float32)
        self.smell_field = smell_field.astype(np.float32)

    @staticmethod
    def _ego_to_world(agent_pos: Tuple[int, int], direction: int, ego_dr: int, ego_dc: int) -> Tuple[int, int]:
        r0, c0 = agent_pos
        if direction == 0:       # facing up
            return r0 + ego_dr, c0 + ego_dc
        if direction == 1:       # facing right
            return r0 + ego_dc, c0 - ego_dr
        if direction == 2:       # facing down
            return r0 - ego_dr, c0 - ego_dc
        if direction == 3:       # facing left
            return r0 - ego_dc, c0 + ego_dr
        raise ValueError(f"Invalid direction {direction}")

    def _egocentric_patch(
        self,
        field: np.ndarray,
        agent_pos: Tuple[int, int],
        direction: int,
        patch_size: int,
        fill_value: float | int,
    ) -> np.ndarray:
        half = patch_size // 2
        patch = np.full((patch_size, patch_size), fill_value, dtype=field.dtype)
        for pr in range(patch_size):
            for pc in range(patch_size):
                ego_dr = pr - half
                ego_dc = pc - half
                wr, wc = self._ego_to_world(agent_pos, direction, ego_dr, ego_dc)
                if 0 <= wr < field.shape[0] and 0 <= wc < field.shape[1]:
                    patch[pr, pc] = field[wr, wc]
        return patch

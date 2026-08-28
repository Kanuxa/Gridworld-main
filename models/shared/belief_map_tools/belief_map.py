"""Agent-owned map assembled from egocentric Gridworld observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, DIR_VECTORS, OBJ_FIRE, OBJ_GLASS, OBJ_ICE, OBJ_MEAT, OBJ_FLOWER


UNKNOWN_OBJECT = -1
PHASE1_LOW_TEMP = 13.0
PHASE1_HIGH_TEMP = 29.0
COMFORT_LOW_TEMP = 18.0
COMFORT_HIGH_TEMP = 24.0


@dataclass(frozen=True)
class BeliefDelta:
    """Observable belief change produced by one sensory update.

    This is intentionally derived only from the prior agent-side belief and
    the new observation.  A controller can use it to invalidate a stale route
    without receiving any hidden environment state.
    """

    newly_seen: int = 0
    object_changes: int = 0
    temperature_changes: int = 0
    smell_changes: int = 0
    updated_cells: int = 0

    @property
    def surprise(self) -> bool:
        return self.object_changes > 0 or self.temperature_changes > 0 or self.smell_changes > 0


@dataclass
class BeliefMap:
    """World-coordinate state available to the planner, with no hidden-map access."""

    grid_size: int
    patch_size: int
    max_health: float = 10.0
    max_energy: float = 10.0
    time_energy_cost: float = 0.20
    forward_energy_cost: float = 0.60
    turn_energy_cost: float = 0.10
    thermal_extra_energy_max: float = 0.50
    max_steps: int = 250
    position: Tuple[int, int] = field(init=False)
    heading: int = field(default=0, init=False)
    seen: np.ndarray = field(init=False)
    visited: np.ndarray = field(init=False)
    object_id: np.ndarray = field(init=False)
    temperature: np.ndarray = field(init=False)
    temperature_known: np.ndarray = field(init=False)
    smell: np.ndarray = field(init=False)
    smell_known: np.ndarray = field(init=False)
    last_seen_step: np.ndarray = field(init=False)
    semantic_change_count: np.ndarray = field(init=False)
    recently_changed: np.ndarray = field(init=False)
    health: float = field(default=0.0, init=False)
    energy: float = field(default=0.0, init=False)
    steps: int = field(default=0, init=False)
    temperature_change_threshold_c: float = 0.75
    smell_change_threshold: float = 0.10
    last_delta: BeliefDelta = field(default_factory=BeliefDelta, init=False)

    def __post_init__(self) -> None:
        if self.grid_size < 1 or self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("grid_size and patch_size must be positive; patch_size must be odd.")
        centre = self.grid_size // 2
        self.position = (centre, centre)
        self.seen = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.object_id = np.full((self.grid_size, self.grid_size), UNKNOWN_OBJECT, dtype=np.int16)
        self.temperature = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        self.temperature_known = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.smell = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        self.smell_known = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.last_seen_step = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        self.semantic_change_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.recently_changed = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    def reset(self, observation: Dict[str, Any]) -> int:
        centre = self.grid_size // 2
        self.position = (centre, centre)
        self.heading = int(observation["direction"]) % 4
        self.seen.fill(False)
        self.visited.fill(False)
        self.object_id.fill(UNKNOWN_OBJECT)
        self.temperature.fill(np.nan)
        self.temperature_known.fill(False)
        self.smell.fill(np.nan)
        self.smell_known.fill(False)
        self.last_seen_step.fill(-1)
        self.semantic_change_count.fill(0)
        self.recently_changed.fill(False)
        self.visited[self.position] = True
        self.health = float(observation.get("health", 0.0))
        self.energy = float(observation.get("energy", 0.0))
        self.steps = 0
        self.last_delta = BeliefDelta()
        return self.observe(observation)

    def in_bounds(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    @staticmethod
    def ego_to_world(position: Tuple[int, int], heading: int, ego_row: int, ego_col: int) -> Tuple[int, int]:
        row, col = position
        if heading == 0:
            return row + ego_row, col + ego_col
        if heading == 1:
            return row + ego_col, col - ego_row
        if heading == 2:
            return row - ego_row, col - ego_col
        return row - ego_col, col + ego_row

    def observe(self, observation: Dict[str, Any]) -> int:
        """Project the current sensory patch into the agent-owned belief map."""
        vision = np.asarray(observation["vision"], dtype=np.int16)
        temperatures = np.asarray(observation["temperature_patch_c"], dtype=np.float32)
        smells = np.asarray(observation["smell_patch"], dtype=np.float32)
        if vision.shape != temperatures.shape or vision.shape != smells.shape:
            raise ValueError("Vision, temperature, and smell patches must have matching shapes.")
        if vision.shape[0] != self.patch_size or vision.shape[1] != self.patch_size:
            raise ValueError("Observation patch shape does not match BeliefMap patch_size.")
        self.heading = int(observation["direction"]) % 4
        self.health = float(observation.get("health", self.health))
        self.energy = float(observation.get("energy", self.energy))
        half = self.patch_size // 2
        newly_seen = 0
        object_changes = 0
        temperature_changes = 0
        smell_changes = 0
        updated_cells = 0
        self.recently_changed.fill(False)
        for patch_row in range(self.patch_size):
            for patch_col in range(self.patch_size):
                world = self.ego_to_world(self.position, self.heading, patch_row - half, patch_col - half)
                if not self.in_bounds(world):
                    continue
                row, col = world
                was_seen = bool(self.seen[row, col])
                if not was_seen:
                    newly_seen += 1
                observed_object = int(vision[patch_row, patch_col])
                observed_temperature = float(temperatures[patch_row, patch_col])
                observed_smell = float(smells[patch_row, patch_col])
                changed = False
                if was_seen and int(self.object_id[row, col]) != observed_object:
                    object_changes += 1
                    changed = True
                if self.temperature_known[row, col] and abs(float(self.temperature[row, col]) - observed_temperature) >= self.temperature_change_threshold_c:
                    temperature_changes += 1
                    changed = True
                if self.smell_known[row, col] and abs(float(self.smell[row, col]) - observed_smell) >= self.smell_change_threshold:
                    smell_changes += 1
                    changed = True
                if changed:
                    self.recently_changed[row, col] = True
                    self.semantic_change_count[row, col] = min(np.iinfo(np.int16).max, self.semantic_change_count[row, col] + 1)
                self.seen[row, col] = True
                self.object_id[row, col] = observed_object
                self.temperature[row, col] = observed_temperature
                self.temperature_known[row, col] = True
                self.smell[row, col] = observed_smell
                self.smell_known[row, col] = True
                self.last_seen_step[row, col] = self.steps
                updated_cells += 1
        self.visited[self.position] = True
        self.last_delta = BeliefDelta(
            newly_seen=newly_seen,
            object_changes=object_changes,
            temperature_changes=temperature_changes,
            smell_changes=smell_changes,
            updated_cells=updated_cells,
        )
        return newly_seen

    def update_after_action(self, action: int, observation: Dict[str, Any], transition_info: Dict[str, Any] | None = None) -> int:
        """Advance internal pose from the executed action, then incorporate observation."""
        previous_heading = self.heading
        self.steps += 1
        if action == ACTION_FORWARD:
            did_move = True if transition_info is None else bool(transition_info.get("did_move", True))
            if did_move:
                dr, dc = DIR_VECTORS[previous_heading]
                next_position = (self.position[0] + dr, self.position[1] + dc)
                if not self.in_bounds(next_position):
                    raise RuntimeError("Planner attempted to move outside the known grid.")
                self.position = next_position
        elif action not in (ACTION_LEFT, ACTION_RIGHT):
            raise ValueError(f"Unknown action: {action}")
        return self.observe(observation)

    @property
    def seen_fraction(self) -> float:
        return float(np.mean(self.seen))

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.visited))

    @property
    def direct_hazard(self) -> np.ndarray:
        return np.isin(self.object_id, [OBJ_FIRE, OBJ_ICE, OBJ_GLASS]) & self.seen

    @property
    def phase1_forbidden(self) -> np.ndarray:
        temperature_forbidden = self.temperature_known & (
            (self.temperature < PHASE1_LOW_TEMP) | (self.temperature > PHASE1_HIGH_TEMP)
        )
        return temperature_forbidden | self.direct_hazard

    @property
    def phase1_safe(self) -> np.ndarray:
        return self.seen & self.temperature_known & ~self.phase1_forbidden

    @property
    def phase2_discomfort_cost(self) -> np.ndarray:
        cost = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        known_temp = np.where(self.temperature_known, self.temperature, np.nan)
        deviation = np.maximum(np.maximum(COMFORT_LOW_TEMP - known_temp, known_temp - COMFORT_HIGH_TEMP), 0.0)
        cost[self.temperature_known] = np.minimum(deviation[self.temperature_known] / 10.0, 1.0)
        return cost

    @property
    def meat(self) -> np.ndarray:
        return (self.object_id == OBJ_MEAT) & self.seen

    @property
    def flower(self) -> np.ndarray:
        return (self.object_id == OBJ_FLOWER) & self.seen

    @property
    def observation_age(self) -> np.ndarray:
        """Normalised time since each cell was last observed; unseen is one."""
        age = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        observed = self.last_seen_step >= 0
        if np.any(observed):
            age[observed] = np.clip(
                (self.steps - self.last_seen_step[observed]) / max(1, self.max_steps),
                0.0,
                1.0,
            )
        return age

    def visibility_gain(self, target: Tuple[int, int]) -> int:
        """Number of currently unseen in-bounds cells in a target's vision square."""
        half = self.patch_size // 2
        row0 = max(0, target[0] - half)
        row1 = min(self.grid_size, target[0] + half + 1)
        col0 = max(0, target[1] - half)
        col1 = min(self.grid_size, target[1] + half + 1)
        return int(np.sum(~self.seen[row0:row1, col0:col1]))

    def export_channels(self, phase: int) -> np.ndarray:
        """Return the legacy 11-channel spatial input used by V11 models."""
        agent = np.zeros_like(self.seen, dtype=np.float32)
        agent[self.position] = 1.0
        phase1_forbidden = self.phase1_forbidden.astype(np.float32)
        channels = [
            self.seen.astype(np.float32),
            self.visited.astype(np.float32),
            self.temperature_known.astype(np.float32),
            np.nan_to_num(self.temperature / 35.0, nan=0.0).astype(np.float32),
            self.direct_hazard.astype(np.float32),
            phase1_forbidden,
            self.phase2_discomfort_cost.astype(np.float32),
            self.meat.astype(np.float32),
            self.flower.astype(np.float32),
            agent,
            np.full_like(agent, float(phase), dtype=np.float32),
        ]
        return np.stack(channels, axis=0)

    def export_adaptive_channels(self, phase: int) -> np.ndarray:
        """Return V11 channels plus freshness/change/smell signals for V12.

        This separate method preserves the 11-channel checkpoint contract of
        existing V11 models while giving new cross-environment models extra
        sensory context for target selection.
        """
        adaptive_channels = [
            self.observation_age.astype(np.float32),
            self.recently_changed.astype(np.float32),
            np.clip(self.semantic_change_count.astype(np.float32) / 3.0, 0.0, 1.0),
            self.smell_known.astype(np.float32),
            np.nan_to_num(self.smell, nan=0.0).astype(np.float32),
        ]
        return np.concatenate((self.export_channels(phase), np.stack(adaptive_channels, axis=0)), axis=0)

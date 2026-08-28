"""Standalone agent map and deterministic thermal frontier planner.

This module intentionally has no dependency on ``coverage_v7`` or any neural
model.  It knows only the environment's public observation and action API.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    DIR_VECTORS,
    OBJ_FIRE,
    OBJ_GLASS,
    OBJ_ICE,
)


KNOWN, VISITED, VISIT_DENSITY, HAZARD = range(4)
FIRE, ICE, GLASS, MEAT, FLOWER = range(4, 9)
TEMPERATURE, SMELL, AGENT, BOUNDARY = range(9, 13)
MAP_CHANNELS = 13
HEADING_NAMES = ("up", "right", "down", "left")


def ego_to_world(pos: Tuple[int, int], direction: int, ego_dr: int, ego_dc: int) -> Tuple[int, int]:
    """Convert an egocentric 5x5-patch offset to the local world map."""
    row, col = pos
    if direction == 0:
        return row + ego_dr, col + ego_dc
    if direction == 1:
        return row + ego_dc, col - ego_dr
    if direction == 2:
        return row - ego_dr, col - ego_dc
    if direction == 3:
        return row - ego_dc, col + ego_dr
    raise ValueError(f"Invalid direction: {direction}")


def user_coordinate(row: int, col: int, grid_size: int) -> Dict[str, int | str]:
    """Convert top-left zero-based indices to the requested `(x;y)` system."""
    x, y = int(col) + 1, int(grid_size) - int(row)
    return {"x": x, "y": y, "text": f"({x};{y})"}


class CoverageMemory:
    """A map reconstructed solely from actions and the current 5x5 observation."""

    def __init__(self, grid_size: int, patch_size: int, ambient_temperature_c: float):
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.ambient_temperature_c = float(ambient_temperature_c)
        self.reset(0)

    def reset(self, direction: int) -> None:
        g = self.grid_size
        self.map = np.zeros((MAP_CHANNELS, g, g), dtype=np.float32)
        self.visit_count = np.zeros((g, g), dtype=np.int32)
        self.pos = (g // 2, g // 2)
        self.direction = int(direction)
        self.map[BOUNDARY, 0, :] = 1.0
        self.map[BOUNDARY, -1, :] = 1.0
        self.map[BOUNDARY, :, 0] = 1.0
        self.map[BOUNDARY, :, -1] = 1.0
        self._mark_position()

    def _mark_position(self) -> None:
        self.map[AGENT].fill(0.0)
        row, col = self.pos
        self.map[AGENT, row, col] = 1.0
        self.visit_count[row, col] += 1
        self.map[VISITED, row, col] = 1.0
        self.map[VISIT_DENSITY] = np.minimum(self.visit_count / 4.0, 1.0)

    def advance(self, action: int) -> None:
        if action == ACTION_LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == ACTION_RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == ACTION_FORWARD:
            dr, dc = DIR_VECTORS[self.direction]
            row, col = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                self.pos = (row, col)
        else:
            raise ValueError(f"Unknown action: {action}")
        self._mark_position()

    def forward_target(self) -> Tuple[int, int]:
        dr, dc = DIR_VECTORS[self.direction]
        return self.pos[0] + dr, self.pos[1] + dc

    def forward_is_new(self) -> bool:
        row, col = self.forward_target()
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size and self.visit_count[row, col] == 0

    def update(self, obs: Dict[str, object]) -> int:
        """Fuse only the public egocentric vision, temperature, and smell patches."""
        known_before = int(self.map[KNOWN].sum())
        self.direction = int(obs["direction"])
        vision = np.asarray(obs["vision"], dtype=np.int64)
        temperature = np.asarray(obs["temperature_patch_c"], dtype=np.float32)
        smell = np.asarray(obs["smell_patch"], dtype=np.float32)
        half = self.patch_size // 2
        for patch_row in range(self.patch_size):
            for patch_col in range(self.patch_size):
                row, col = ego_to_world(self.pos, self.direction, patch_row - half, patch_col - half)
                if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                    continue
                obj = int(vision[patch_row, patch_col])
                self.map[KNOWN, row, col] = 1.0
                self.map[FIRE:FLOWER + 1, row, col] = 0.0
                object_channel = {1: FIRE, 2: MEAT, 3: FLOWER, 4: GLASS, 5: ICE}.get(obj)
                if object_channel is not None:
                    self.map[object_channel, row, col] = 1.0
                self.map[HAZARD, row, col] = float(obj in (OBJ_FIRE, OBJ_ICE, OBJ_GLASS))
                self.map[TEMPERATURE, row, col] = np.clip(
                    (temperature[patch_row, patch_col] - self.ambient_temperature_c) / 13.0, -1.0, 1.0
                )
                self.map[SMELL, row, col] = np.clip(smell[patch_row, patch_col], 0.0, 1.0)
        return int(self.map[KNOWN].sum()) - known_before


class StaticFrontierPlanner:
    """Dijkstra frontier planner using only :class:`CoverageMemory` data."""

    def __init__(
        self,
        revisit_cost: float = 1.8,
        hazard_cost: float = 10.0,
        reserve_health_norm: float = 0.50,
        resource_energy_margin: float = 4.0,
        frontier_info_weight: float = 0.14,
        frontier_cost_weight: float = 0.035,
        frontier_hazard_weight: float = 0.65,
        repeat_visit_cost: float = 0.35,
        comfort_low_c: float = 18.0,
        comfort_high_c: float = 24.0,
        thermal_extra_energy_max: float = 0.50,
        discomfort_temp_scale_c: float = 10.0,
    ):
        self.revisit_cost = float(revisit_cost)
        self.hazard_cost = float(hazard_cost)
        self.reserve_health_norm = float(reserve_health_norm)
        self.resource_energy_margin = float(resource_energy_margin)
        self.frontier_info_weight = float(frontier_info_weight)
        self.frontier_cost_weight = float(frontier_cost_weight)
        self.frontier_hazard_weight = float(frontier_hazard_weight)
        self.repeat_visit_cost = float(repeat_visit_cost)
        self.comfort_low_c = float(comfort_low_c)
        self.comfort_high_c = float(comfort_high_c)
        self.thermal_extra_energy_max = float(thermal_extra_energy_max)
        self.discomfort_temp_scale_c = float(discomfort_temp_scale_c)

    @staticmethod
    def successors(state: Tuple[int, int, int]) -> Iterable[Tuple[int, Tuple[int, int, int], float]]:
        row, col, direction = state
        yield ACTION_LEFT, (row, col, (direction - 1) % 4), 0.3
        yield ACTION_RIGHT, (row, col, (direction + 1) % 4), 0.3
        dr, dc = DIR_VECTORS[direction]
        yield ACTION_FORWARD, (row + dr, col + dc, direction), 0.8

    def temperature_c_at(self, memory: CoverageMemory, row: int, col: int) -> float:
        if not (0 <= row < memory.grid_size and 0 <= col < memory.grid_size):
            return memory.ambient_temperature_c
        if memory.map[KNOWN, row, col] < 0.5:
            return memory.ambient_temperature_c
        return memory.ambient_temperature_c + 13.0 * float(memory.map[TEMPERATURE, row, col])

    def thermal_extra_at(self, memory: CoverageMemory, row: int, col: int) -> float:
        temperature_c = self.temperature_c_at(memory, row, col)
        if self.comfort_low_c <= temperature_c <= self.comfort_high_c:
            return 0.0
        nearest = self.comfort_low_c if temperature_c < self.comfort_low_c else self.comfort_high_c
        return float(min(abs(temperature_c - nearest) / self.discomfort_temp_scale_c, 1.0) * self.thermal_extra_energy_max)

    def action_energy_estimate(self, memory: CoverageMemory, action: int) -> Tuple[float, float, float]:
        if action == ACTION_FORWARD:
            row, col = memory.forward_target()
            base_energy = 0.8
        elif action in (ACTION_LEFT, ACTION_RIGHT):
            row, col = memory.pos
            base_energy = 0.3
        else:
            raise ValueError(f"Unknown action: {action}")
        thermal = self.thermal_extra_at(memory, row, col)
        return base_energy + thermal, thermal, self.temperature_c_at(memory, row, col)

    def action_toward(self, memory: CoverageMemory, target_row: int, target_col: int) -> Tuple[int | None, float, float]:
        """Return the first thermal/revisit-aware action of a safe target route."""
        g = memory.grid_size
        if not (0 <= target_row < g and 0 <= target_col < g):
            return None, float("inf"), float("inf")
        start = (memory.pos[0], memory.pos[1], memory.direction)
        if (target_row, target_col) == memory.pos:
            return ACTION_FORWARD, 0.0, 0.0
        dist: Dict[Tuple[int, int, int], float] = {start: 0.0}
        route_energy: Dict[Tuple[int, int, int], float] = {start: 0.0}
        first_action: Dict[Tuple[int, int, int], int] = {start: ACTION_FORWARD}
        queue: List[Tuple[float, Tuple[int, int, int]]] = [(0.0, start)]
        while queue:
            cost, state = heapq.heappop(queue)
            if cost != dist.get(state):
                continue
            row, col, _ = state
            if (row, col) == (target_row, target_col):
                return first_action[state], cost, route_energy[state]
            for action, next_state, base_energy in self.successors(state):
                next_row, next_col, _ = next_state
                if action == ACTION_FORWARD:
                    if not (0 <= next_row < g and 0 <= next_col < g) or memory.map[HAZARD, next_row, next_col] > 0.5:
                        continue
                    thermal = self.thermal_extra_at(memory, next_row, next_col)
                else:
                    thermal = self.thermal_extra_at(memory, row, col)
                transition = base_energy + thermal
                if action == ACTION_FORWARD:
                    transition += self.revisit_cost * float(memory.map[VISITED, next_row, next_col] > 0.5)
                    transition += self.repeat_visit_cost * float(memory.visit_count[next_row, next_col] > 1)
                next_cost = cost + transition
                if next_cost < dist.get(next_state, float("inf")):
                    dist[next_state] = next_cost
                    first_action[next_state] = action if state == start else first_action[state]
                    route_energy[next_state] = route_energy[state] + base_energy + thermal
                    heapq.heappush(queue, (next_cost, next_state))
        return None, float("inf"), float("inf")

    def action_prior(self, memory: CoverageMemory, health_norm: float = 1.0, energy_norm: float = 1.0) -> Tuple[np.ndarray, int, bool]:
        """Select a known meat target when needed, otherwise the best frontier."""
        g = memory.grid_size
        start = (memory.pos[0], memory.pos[1], memory.direction)
        dist: Dict[Tuple[int, int, int], float] = {start: 0.0}
        first_action: Dict[Tuple[int, int, int], int] = {start: ACTION_FORWARD}
        path_energy: Dict[Tuple[int, int, int], float] = {start: 0.0}
        best_cost = np.full((g, g), np.inf, dtype=np.float32)
        best_energy = np.full((g, g), np.inf, dtype=np.float32)
        best_state: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        queue: List[Tuple[float, Tuple[int, int, int]]] = [(0.0, start)]
        while queue:
            cost, state = heapq.heappop(queue)
            if cost != dist.get(state):
                continue
            row, col, _ = state
            if cost < best_cost[row, col]:
                best_cost[row, col] = cost
                best_energy[row, col] = path_energy[state]
                best_state[(row, col)] = state
            for action, next_state, base_energy in self.successors(state):
                nr, nc, _ = next_state
                if action == ACTION_FORWARD:
                    if not (0 <= nr < g and 0 <= nc < g):
                        continue
                    thermal = self.thermal_extra_at(memory, nr, nc)
                else:
                    thermal = self.thermal_extra_at(memory, row, col)
                transition = base_energy + thermal
                if action == ACTION_FORWARD:
                    transition += self.revisit_cost * float(memory.map[VISITED, nr, nc] > 0.5)
                    transition += self.repeat_visit_cost * float(memory.visit_count[nr, nc] > 1)
                    transition += self.hazard_cost * float(memory.map[HAZARD, nr, nc] > 0.5)
                next_cost = cost + transition
                if next_cost < dist.get(next_state, float("inf")):
                    dist[next_state] = next_cost
                    first_action[next_state] = action if state == start else first_action[state]
                    path_energy[next_state] = path_energy[state] + base_energy + thermal
                    heapq.heappush(queue, (next_cost, next_state))

        meat_targets = [
            (row, col) for row in range(g) for col in range(g)
            if memory.map[MEAT, row, col] > 0.5 and np.isfinite(best_cost[row, col])
        ]
        if meat_targets:
            row, col = min(meat_targets, key=lambda cell: best_cost[cell])
            current_energy = 10.0 * np.clip(float(energy_norm), 0.0, 1.0)
            current_health = 10.0 * np.clip(float(health_norm), 0.0, 1.0)
            energy_before_reserve = current_energy + max(0.0, current_health - 10.0 * self.reserve_health_norm) * 10.0
            if health_norm <= self.reserve_health_norm or float(best_energy[row, col]) + self.resource_energy_margin >= energy_before_reserve:
                prior = np.zeros(3, dtype=np.float32)
                prior[first_action[best_state[(row, col)]]] = 1.0
                return prior, row * g + col, True

        best: Tuple[float, int, int, int] | None = None
        for row in range(g):
            for col in range(g):
                if memory.map[VISITED, row, col] > 0.5 or not np.isfinite(best_cost[row, col]):
                    continue
                row0, row1 = max(0, row - 2), min(g, row + 3)
                col0, col1 = max(0, col - 2), min(g, col + 3)
                info_gain = float((memory.map[KNOWN, row0:row1, col0:col1] < 0.5).sum())
                score = self.frontier_info_weight * info_gain - self.frontier_cost_weight * float(best_cost[row, col]) - self.frontier_hazard_weight * float(memory.map[HAZARD, row, col] > 0.5)
                state = best_state[(row, col)]
                item = (score, -int(best_cost[row, col] * 1000), -row * g - col, first_action[state])
                if best is None or item > best:
                    best = item
        if best is None:
            return np.zeros(3, dtype=np.float32), memory.pos[0] * g + memory.pos[1], False
        prior = np.zeros(3, dtype=np.float32)
        prior[best[3]] = 1.0
        return prior, int(-best[2]), False

    def frontier_scores(
        self,
        memory: CoverageMemory,
        candidates: Iterable[Tuple[int, int]],
        meat_mode: bool = False,
    ) -> Dict[Tuple[int, int], float]:
        """Score legal target cells with the same route model as ``action_prior``.

        Scores are exposed for a neural residual policy.  They are computed
        from the agent-owned map only; an unreachable candidate receives no
        score and is left masked by the caller.  In meat-recovery mode, lower
        route cost is better, exactly matching the planner's nearest-meat
        preference.
        """
        cells = list(candidates)
        if not cells:
            return {}
        g = memory.grid_size
        start = (memory.pos[0], memory.pos[1], memory.direction)
        dist: Dict[Tuple[int, int, int], float] = {start: 0.0}
        best_cost = np.full((g, g), np.inf, dtype=np.float32)
        queue: List[Tuple[float, Tuple[int, int, int]]] = [(0.0, start)]
        while queue:
            cost, state = heapq.heappop(queue)
            if cost != dist.get(state):
                continue
            row, col, _ = state
            if cost < best_cost[row, col]:
                best_cost[row, col] = cost
            for action, next_state, base_energy in self.successors(state):
                nr, nc, _ = next_state
                if action == ACTION_FORWARD:
                    if not (0 <= nr < g and 0 <= nc < g):
                        continue
                    thermal = self.thermal_extra_at(memory, nr, nc)
                else:
                    thermal = self.thermal_extra_at(memory, row, col)
                transition = base_energy + thermal
                if action == ACTION_FORWARD:
                    transition += self.revisit_cost * float(memory.map[VISITED, nr, nc] > 0.5)
                    transition += self.repeat_visit_cost * float(memory.visit_count[nr, nc] > 1)
                    transition += self.hazard_cost * float(memory.map[HAZARD, nr, nc] > 0.5)
                next_cost = cost + transition
                if next_cost < dist.get(next_state, float("inf")):
                    dist[next_state] = next_cost
                    heapq.heappush(queue, (next_cost, next_state))

        scores: Dict[Tuple[int, int], float] = {}
        for row, col in cells:
            route_cost = float(best_cost[row, col])
            if not np.isfinite(route_cost):
                continue
            if meat_mode:
                scores[(row, col)] = -route_cost
                continue
            row0, row1 = max(0, row - 2), min(g, row + 3)
            col0, col1 = max(0, col - 2), min(g, col + 3)
            info_gain = float((memory.map[KNOWN, row0:row1, col0:col1] < 0.5).sum())
            hazard = float(memory.map[HAZARD, row, col] > 0.5)
            scores[(row, col)] = (
                self.frontier_info_weight * info_gain
                - self.frontier_cost_weight * route_cost
                - self.frontier_hazard_weight * hazard
            )
        return scores

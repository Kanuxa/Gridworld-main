"""Agent-owned allocentric memory used by v7 neural models.

The deterministic planner itself is maintained separately in ``planner_lab``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, DIR_VECTORS, OBJ_FIRE, OBJ_GLASS, OBJ_ICE


KNOWN, VISITED, VISIT_DENSITY, HAZARD = range(4)
FIRE, ICE, GLASS, MEAT, FLOWER = range(4, 9)
TEMPERATURE, SMELL, AGENT, BOUNDARY = range(9, 13)
MAP_CHANNELS = 13
HEADING_NAMES = ("up", "right", "down", "left")


def ego_to_world(pos: Tuple[int, int], direction: int, ego_dr: int, ego_dc: int) -> Tuple[int, int]:
    """Convert an environment egocentric-patch offset into world row/column."""
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
    """Return the requested bottom-left-origin, one-indexed coordinate."""
    x, y = int(col) + 1, int(grid_size) - int(row)
    return {"x": x, "y": y, "text": f"({x};{y})"}


def local_sensor_tensor(obs: Dict[str, object], ambient_temperature_c: float) -> np.ndarray:
    """Build the plan's 8x5x5 local visual/thermal/smell tensor."""
    vision = np.asarray(obs["vision"], dtype=np.int64)
    one_hot = np.eye(6, dtype=np.float32)[vision].transpose(2, 0, 1)
    temperature = np.asarray(obs["temperature_patch_c"], dtype=np.float32)
    temperature = np.clip((temperature - ambient_temperature_c) / 13.0, -1.0, 1.0)[None]
    smell = np.clip(np.asarray(obs["smell_patch"], dtype=np.float32), 0.0, 1.0)[None]
    return np.concatenate([one_hot, temperature, smell], axis=0).astype(np.float32)


class CoverageMemory:
    """15x15 map reconstructed from the agent's own observation/action history."""

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
        """Apply known deterministic action mechanics to the internal pose."""
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
        """Fuse the 5x5 observation and return the number of newly known cells."""
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
                self.map[TEMPERATURE, row, col] = np.clip((temperature[patch_row, patch_col] - self.ambient_temperature_c) / 13.0, -1.0, 1.0)
                self.map[SMELL, row, col] = np.clip(smell[patch_row, patch_col], 0.0, 1.0)
        return int(self.map[KNOWN].sum()) - known_before

    def state(self, obs: Dict[str, object], remaining_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scalars = np.array([
            float(obs["health_norm"]),
            np.clip(float(obs["energy_norm"]), -2.0, 1.0),
            np.clip(remaining_steps / 250.0, 0.0, 1.0),
            np.sin(self.direction * np.pi / 2.0),
            np.cos(self.direction * np.pi / 2.0),
            float(self.map[VISITED].mean()),
        ], dtype=np.float32)
        return self.map.copy(), local_sensor_tensor(obs, self.ambient_temperature_c), scalars

    def world_to_agent_frame(self, row: int, col: int) -> Tuple[int, int]:
        """Map a world cell into an agent-centred, heading-aligned frame."""
        relative_row, relative_col = int(row) - self.pos[0], int(col) - self.pos[1]
        if self.direction == 0:
            ego_row, ego_col = relative_row, relative_col
        elif self.direction == 1:
            ego_row, ego_col = -relative_col, relative_row
        elif self.direction == 2:
            ego_row, ego_col = -relative_row, -relative_col
        else:
            ego_row, ego_col = relative_col, -relative_row
        centre = self.grid_size - 1
        return centre + ego_row, centre + ego_col

    def agent_centred_map(self) -> np.ndarray:
        """Return the full known map centred on the agent with coordinate channels."""
        size = 2 * self.grid_size - 1
        centre = self.grid_size - 1
        out = np.zeros((MAP_CHANNELS + 2, size, size), dtype=np.float32)
        out[BOUNDARY].fill(1.0)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                ego_row, ego_col = self.world_to_agent_frame(row, col)
                out[:MAP_CHANNELS, ego_row, ego_col] = self.map[:, row, col]
        coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        out[MAP_CHANNELS] = np.broadcast_to(coordinates[None, :], (size, size))
        out[MAP_CHANNELS + 1] = np.broadcast_to(coordinates[:, None], (size, size))
        out[AGENT, centre, centre] = 1.0
        return out


# The active, map-only deterministic planner is deliberately outside v7.
from models.non_model_based.partial_observation_frontier_planner.core import StaticFrontierPlanner

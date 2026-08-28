"""Full-information coverage controller with exact environment energy costs.

God mode is an oracle benchmark, not a valid partially observed agent. It
reads the complete object and temperature grids at reset and runs receding,
orientation-aware global routing. It mirrors the environment's energy,
health, meat, and hazard mechanics while planning.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    DIR_VECTORS,
    OBJ_FIRE,
    OBJ_GLASS,
    OBJ_ICE,
    OBJ_MEAT,
)


HEADING_NAMES = ("up", "right", "down", "left")


def user_coordinate(row: int, col: int, grid_size: int) -> Dict[str, int | str]:
    """Return a one-indexed coordinate with bottom-left origin."""
    x, y = int(col) + 1, int(grid_size) - int(row)
    return {"x": x, "y": y, "text": f"({x};{y})"}


@dataclass(frozen=True)
class OracleConfig:
    search_beam_width: int = 4096


@dataclass(frozen=True, slots=True)
class _State:
    row: int
    col: int
    heading: int
    visited: int
    meat_remaining: int
    health: int
    energy: float
    unique_cells: int
    repeat_forwards: int
    turn_actions: int
    turn_streak: int
    terminated: bool
    parent: "_State | None" = None
    action: int | None = None


class GodModeController:
    """Privileged global-route controller using the entire reset world state."""

    def __init__(self, env, cfg: OracleConfig = OracleConfig()):
        self.env = env
        self.cfg = cfg
        self.grid = env.reveal_world_ids()
        self.temperature_c = env.reveal_temperature_field_c()
        self.grid_size = int(env.config.grid_size)
        self._initial_grid = self.grid.copy()
        self._thermal = self._thermal_extra(self.temperature_c)
        self.turn_energy = env.config.time_energy_cost + env.config.turn_energy_cost + self._thermal
        self.forward_enter_energy = env.config.time_energy_cost + env.config.forward_energy_cost + self._thermal
        self._damage = np.zeros_like(self.grid, dtype=np.int16)
        self._damage[self.grid == OBJ_FIRE] = int(env.config.fire_damage)
        self._damage[self.grid == OBJ_ICE] = int(env.config.ice_damage)
        self._damage[self.grid == OBJ_GLASS] = int(env.config.glass_damage)

    def _thermal_extra(self, temperature_c: np.ndarray) -> np.ndarray:
        c = self.env.config
        nearest = np.where(temperature_c < c.comfort_low_c, c.comfort_low_c, c.comfort_high_c)
        deviation = np.where(
            (temperature_c >= c.comfort_low_c) & (temperature_c <= c.comfort_high_c),
            0.0,
            np.abs(temperature_c - nearest),
        )
        discomfort = np.minimum(deviation / max(1e-6, c.discomfort_temp_scale_c), 1.0)
        return (discomfort * c.thermal_extra_energy_max).astype(np.float32)

    def world_knowledge(self) -> Dict[str, object]:
        """Serializable privileged data, including exact per-cell action energy."""
        return {
            "access": "god_mode: complete map and temperature field revealed at reset",
            "coordinate_system": {"origin": "bottom-left", "bottom_left": "(1;1)", "top_right": f"({self.grid_size};{self.grid_size})"},
            "world_ids_top_left_origin": self._initial_grid.astype(int).tolist(),
            "temperature_c_top_left_origin": self.temperature_c.astype(float).tolist(),
            "turn_energy_by_cell": self.turn_energy.astype(float).tolist(),
            "forward_enter_energy_by_cell": self.forward_enter_energy.astype(float).tolist(),
            "energy_rule": {
                "turn": "time_energy_cost + turn_energy_cost + thermal_extra(current_cell)",
                "forward": "time_energy_cost + forward_energy_cost + thermal_extra(destination_cell)",
            },
        }

    def _flat(self, row: int, col: int) -> int:
        return row * self.grid_size + col

    def _bit(self, row: int, col: int) -> int:
        return 1 << self._flat(row, col)

    def _state_from_environment(self) -> _State:
        """Reconstruct the exact planning state from the true environment."""
        visited = 0
        for row, col in np.argwhere(self.env.visited_map > 0.5):
            visited |= self._bit(int(row), int(col))
        meat_remaining = 0
        self.grid = self.env.reveal_world_ids()
        for row, col in np.argwhere(self.grid == OBJ_MEAT):
            meat_remaining |= self._bit(int(row), int(col))
        row, col = self.env.agent_pos
        return _State(
            row=int(row),
            col=int(col),
            heading=int(self.env.direction),
            visited=visited,
            meat_remaining=meat_remaining,
            health=int(self.env.health),
            energy=float(self.env.energy),
            unique_cells=int(visited.bit_count()),
            repeat_forwards=0,
            turn_actions=0,
            turn_streak=0,
            terminated=False,
        )

    def _advance(self, state: _State, action: int) -> _State | None:
        """Apply one real environment transition to a search state exactly."""
        c = self.env.config
        row, col, heading = state.row, state.col, state.heading
        visited, meat_remaining = state.visited, state.meat_remaining
        health, energy = state.health, state.energy
        unique_cells, repeat_forwards = state.unique_cells, state.repeat_forwards
        turn_actions, turn_streak = state.turn_actions, state.turn_streak

        if action == ACTION_FORWARD:
            dr, dc = DIR_VECTORS[heading]
            row, col = row + dr, col + dc
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                return None
            bit = self._bit(row, col)
            if visited & bit:
                repeat_forwards += 1
            else:
                visited |= bit
                unique_cells += 1
            health -= int(self._damage[row, col])
            if meat_remaining & bit:
                health += min(int(c.meat_heal), int(c.max_health) - health)
                meat_remaining &= ~bit
            energy -= float(self.forward_enter_energy[row, col])
            turn_streak = 0
        elif action in (ACTION_LEFT, ACTION_RIGHT):
            # An immediate counter-turn, or a third consecutive turn, is
            # strictly dominated by a shorter route to the same orientation.
            if (state.action == ACTION_LEFT and action == ACTION_RIGHT) or (state.action == ACTION_RIGHT and action == ACTION_LEFT):
                return None
            if state.turn_streak >= 2:
                return None
            heading = (heading + (-1 if action == ACTION_LEFT else 1)) % 4
            energy -= float(self.turn_energy[row, col])
            turn_actions += 1
            turn_streak += 1
        else:
            raise ValueError(f"Unknown action: {action}")

        while energy <= 0.0 and health > 0:
            health -= 1
            energy += float(c.max_energy)
        return _State(
            row=row,
            col=col,
            heading=heading,
            visited=visited,
            meat_remaining=meat_remaining,
            health=health,
            energy=energy,
            unique_cells=unique_cells,
            repeat_forwards=repeat_forwards,
            turn_actions=turn_actions,
            turn_streak=turn_streak,
            terminated=health <= 0,
            parent=state,
            action=action,
        )

    @staticmethod
    def _search_rank(state: _State) -> Tuple[int, int, float, int, int]:
        """Coverage is primary; health and efficiency preserve future reach."""
        return (
            state.unique_cells,
            state.health,
            state.energy,
            -state.repeat_forwards,
            -state.turn_actions,
        )

    @staticmethod
    def _actions_to(state: _State) -> Tuple[int, ...]:
        actions: List[int] = []
        while state.parent is not None:
            assert state.action is not None
            actions.append(state.action)
            state = state.parent
        return tuple(reversed(actions))

    def plan_episode(self) -> Tuple[Tuple[int, ...], Dict[str, object]]:
        """Search complete trajectories from reset through the episode horizon.

        Every candidate is simulated with the actual energy, fatigue, hazard,
        meat, and coverage mechanics. The search never replans from local
        observations: it chooses a complete initial-map trajectory, then the
        runner executes that trajectory unchanged.
        """
        root = self._state_from_environment()
        beam = [root]
        terminal_states: List[_State] = []
        expanded_transitions = 0
        retained_states = 1
        horizon = int(self.env.config.max_steps)

        for _ in range(horizon):
            children: List[_State] = []
            for state in beam:
                if state.terminated:
                    continue
                for action in (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT):
                    child = self._advance(state, action)
                    if child is None:
                        continue
                    expanded_transitions += 1
                    if child.terminated:
                        terminal_states.append(child)
                    else:
                        children.append(child)
            if not children:
                beam = []
                break
            beam = heapq.nlargest(self.cfg.search_beam_width, children, key=self._search_rank)
            retained_states += len(beam)

        # A complete route must finish only at an environment terminal state
        # or at the 250-step horizon.  Intermediate states cannot be returned
        # merely because they temporarily retain more health or energy.
        completed_states = terminal_states + beam
        best = max(completed_states, key=self._search_rank)
        route = self._actions_to(best)
        return route, {
            "algorithm": "full_episode_global_beam_search",
            "search_horizon": horizon,
            "search_beam_width": self.cfg.search_beam_width,
            "expanded_transitions": expanded_transitions,
            "retained_states": retained_states,
            "planned_steps": len(route),
            "planned_unique_cells": best.unique_cells,
            "planned_coverage": best.unique_cells / float(self.grid_size * self.grid_size),
            "planned_forward_actions": len(route) - best.turn_actions,
            "planned_turn_actions": best.turn_actions,
            "planned_repeat_forwards": best.repeat_forwards,
            "planned_final_health": best.health,
            "planned_final_energy": best.energy,
            "planned_terminated": best.terminated,
        }

    def action_costs_at_agent(self) -> Dict[str, float | None]:
        """Exact action energy; an out-of-bounds forward move is ``None``."""
        row, col = self.env.agent_pos
        dr, dc = DIR_VECTORS[self.env.direction]
        forward_row, forward_col = row + dr, col + dc
        forward: float | None = None
        if 0 <= forward_row < self.grid_size and 0 <= forward_col < self.grid_size:
            forward = float(self.forward_enter_energy[forward_row, forward_col])
        return {
            "turn_left": float(self.turn_energy[row, col]),
            "turn_right": float(self.turn_energy[row, col]),
            "forward": forward,
        }

    def current_cell_coordinate(self) -> Dict[str, int | str]:
        return user_coordinate(*self.env.agent_pos, self.grid_size)

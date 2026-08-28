"""Per-episode JSON tracing for route and energy debugging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, OBJ_EMPTY, OBJ_LABELS
from models.model_based.ppo.spatial_target_ppo_with_planner.memory import HEADING_NAMES, user_coordinate


ACTION_NAMES = {ACTION_FORWARD: "forward", ACTION_LEFT: "turn_left", ACTION_RIGHT: "turn_right"}


class EpisodeTrace:
    """Collects a complete, human-readable trace without exposing state to policy."""

    def __init__(self, output_dir: str | Path, episode: int, seed: int, env, obs: Dict[str, object]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode = int(episode)
        self.seed = int(seed)
        self.grid_size = int(env.config.grid_size)
        self.route = []
        self.special_cells = self._special_cells(env)
        start = user_coordinate(self.grid_size // 2, self.grid_size // 2, self.grid_size)
        self.initial_agent = {
            "coordinate": start,
            "heading": HEADING_NAMES[int(obs["direction"])],
            "health": float(obs["health"]),
            "energy": float(obs["energy"]),
        }

    def _special_cells(self, env) -> Dict[str, list]:
        grid = env.reveal_world_ids()  # Analysis-only metadata; never provided to the policy.
        out: Dict[str, list] = {}
        for obj_id, label in OBJ_LABELS.items():
            if obj_id == OBJ_EMPTY:
                continue
            cells = []
            for row, col in np.argwhere(grid == obj_id):
                cells.append(user_coordinate(int(row), int(col), self.grid_size))
            out[label.lower()] = cells
        return out

    def add(
        self,
        action: int,
        coord_before: Dict[str, int | str],
        coord_after: Dict[str, int | str],
        heading_before: int,
        heading_after: int,
        energy_before: float,
        obs_after: Dict[str, object],
        reward: float,
        info: Dict[str, object],
        was_new: bool,
        prior_visits: int,
        turn_streak_before: int,
        previous_action: int | None,
        decision_context: Dict[str, object],
        action_mask,
        planner_target: int,
    ) -> None:
        components = {
            "time": float(info.get("time_base_cost", 0.0)),
            "forward": float(info.get("forward_extra_cost", 0.0)),
            "turn": float(info.get("turn_extra_cost", 0.0)),
            "thermal": float(info.get("thermal_extra_this_tick", 0.0)),
        }
        spent = float(sum(components.values()))
        event = {
            "step": int(info["steps"]),
            "action": ACTION_NAMES[action],
            "position_before": coord_before,
            "position_after": coord_after,
            "heading_before": HEADING_NAMES[heading_before],
            "heading_after": HEADING_NAMES[heading_after],
            "turn_direction": "left" if action == ACTION_LEFT else ("right" if action == ACTION_RIGHT else None),
            "opposite_to_previous_turn": bool(
                previous_action in (ACTION_LEFT, ACTION_RIGHT)
                and action in (ACTION_LEFT, ACTION_RIGHT)
                and action != previous_action
            ),
            "moved_to_new_cell": bool(was_new),
            "target_visit_count_before": int(prior_visits),
            "consecutive_turns_before": int(turn_streak_before),
            "consecutive_turns_after": int(info.get("consecutive_turn_steps", 0)),
            "forward_status_before": str(decision_context["forward_status"]),
            "visible_safe_unvisited_directions": list(decision_context["safe_unvisited_directions"]),
            "planner_action": ACTION_NAMES[int(decision_context["planner_action"])],
            "planner_mode": str(decision_context.get("resource_mode", "coverage")),
            "planner_expected_action_energy": float(
                decision_context.get("planner_expected_action_energy", 0.0)
            ),
            "planner_expected_thermal_energy": float(
                decision_context.get("planner_expected_thermal_energy", 0.0)
            ),
            "planner_target_temperature_c": float(
                decision_context.get("planner_target_temperature_c", 0.0)
            ),
            "forward_expected_thermal_energy": float(
                decision_context.get("forward_expected_thermal_energy", 0.0)
            ),
            "forward_temperature_c": float(decision_context.get("forward_temperature_c", 0.0)),
            "planner_committed": bool(decision_context.get("planner_committed", False)),
            "planner_commitment_probability": float(
                decision_context.get("planner_commitment_probability", 0.0)
            ),
            "soft_planner_weight": float(decision_context.get("soft_planner_weight", 0.0)),
            "residual_logit_limit": float(decision_context.get("residual_logit_limit", 0.0)),
            "selected_target_flat_index": decision_context.get("selected_target_flat_index"),
            "teacher_target_flat_index": decision_context.get("teacher_target_flat_index"),
            "selected_target_coordinate": decision_context.get("selected_target_coordinate"),
            "teacher_target_coordinate": decision_context.get("teacher_target_coordinate"),
            "target_matched_teacher": decision_context.get("target_matched_teacher"),
            "route_cost": decision_context.get("route_cost"),
            "route_energy": decision_context.get("route_energy"),
            "forced_action": (
                ACTION_NAMES[int(decision_context["forced_action"])]
                if decision_context.get("forced_action") is not None else None
            ),
            "forced_action_reason": decision_context.get("forced_action_reason"),
            "directed_turn_action": (
                ACTION_NAMES[int(decision_context["directed_turn_action"])]
                if decision_context.get("directed_turn_action") is not None else None
            ),
            "forward_blocked_by_local_rule": bool(decision_context["forward_blocked_by_local_rule"]),
            "resource_bridge_override": bool(decision_context.get("resource_bridge_override", False)),
            "conserve_forward_meat": bool(decision_context.get("conserve_forward_meat", False)),
            "thermal_turn_allowed": bool(decision_context.get("thermal_turn_allowed", False)),
            "turn_escape": bool(decision_context.get("turn_escape", False)),
            "masked_actions": [
                ACTION_NAMES[index] for index, value in enumerate(action_mask) if float(value) < -1e8
            ],
            "energy": {
                "before": float(energy_before),
                "spent": spent,
                "after": float(obs_after["energy"]),
                "cost_components": components,
            },
            "health_after": float(obs_after["health"]),
            "health_delta": float(info.get("health_delta", 0.0)),
            "contacted": str(info.get("contacted_label", "Empty")),
            "environment_reward": float(reward),
            "environment_event": str(info.get("last_event", "")),
            "planner_target_flat_index": int(planner_target),
        }
        self.route.append(event)

    def write(self, summary: Dict[str, object]) -> Path:
        payload = {
            "episode": self.episode,
            "seed": self.seed,
            "coordinate_system": {
                "origin": "bottom-left",
                "bottom_left": "(1;1)",
                "top_right": f"({self.grid_size};{self.grid_size})",
                "format": {"x": "left to right", "y": "bottom to top"},
            },
            "special_cells": self.special_cells,
            "initial_agent": self.initial_agent,
            "route": self.route,
            "summary": summary,
        }
        path = self.output_dir / f"episode_{self.episode:05d}_seed_{self.seed}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

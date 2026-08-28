"""Run the privileged god-mode coverage oracle and write complete traces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, EnvConfig, OBJ_EMPTY, OBJ_LABELS, ObservationSwitches, SensoryGridEnv
from models.benchmarks.full_information_oracle.oracle import GodModeController, HEADING_NAMES, OracleConfig, user_coordinate


ACTION_NAMES = {ACTION_FORWARD: "forward", ACTION_LEFT: "turn_left", ACTION_RIGHT: "turn_right"}


@dataclass
class GodModeRunConfig:
    episodes: int = 50
    seed: int = 50_000
    save_dir: str = "runs/runs_15x15/runs_god_mode"
    trace_every: int = 10
    search_beam_width: int = 4096


def build_env() -> Tuple[SensoryGridEnv, ObservationSwitches]:
    return SensoryGridEnv(EnvConfig(survival_bonus=0.0)), ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )


def special_cells(env) -> Dict[str, list]:
    grid = env.reveal_world_ids()
    result: Dict[str, list] = {}
    for obj_id, label in OBJ_LABELS.items():
        if obj_id != OBJ_EMPTY:
            result[label.lower()] = [user_coordinate(int(row), int(col), env.config.grid_size) for row, col in np.argwhere(grid == obj_id)]
    return result


def write_trace(path: Path, episode: int, seed: int, env, controller: GodModeController, route: List[Dict[str, object]], summary: Dict[str, float], initial_special_cells: Dict[str, list], episode_plan: Dict[str, object], planned_actions: Tuple[int, ...]) -> None:
    payload = {
        "episode": int(episode),
        "seed": int(seed),
        "mode": "god_mode_oracle",
        "privileged_access": "The controller reads the complete object map and full temperature field at reset. This is an upper-bound benchmark, not a fair partial-observation policy.",
        "coordinate_system": {"origin": "bottom-left", "bottom_left": "(1;1)", "top_right": f"({env.config.grid_size};{env.config.grid_size})"},
        "special_cells": initial_special_cells,
        "world_knowledge": controller.world_knowledge(),
        "episode_plan": {**episode_plan, "actions": [ACTION_NAMES[action] for action in planned_actions]},
        "route": route,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def run_episode(cfg: GodModeRunConfig, episode: int, seed: int, trace_dir: Path | None = None) -> Dict[str, float]:
    env, switches = build_env()
    obs, _ = env.reset(seed=seed)
    controller = GodModeController(
        env,
        OracleConfig(
            search_beam_width=cfg.search_beam_width,
        ),
    )
    initial_special_cells = special_cells(env)
    route: List[Dict[str, object]] = []
    planned_tuple, plan_info = controller.plan_episode()
    all_planned_actions = planned_tuple
    planned: List[int] = list(planned_tuple)
    forwards = turns = repeats = 0
    health_loss = 0.0
    previous_action: int | None = None
    decisions = 1

    while not (env.terminated or env.truncated):
        if not planned:
            raise RuntimeError("The global plan ended before the environment episode ended.")
        action = int(planned.pop(0))
        before_row, before_col = env.agent_pos
        before_heading = int(env.direction)
        before_energy = float(env.energy)
        action_costs = controller.action_costs_at_agent()
        selected_cost = action_costs[ACTION_NAMES[action]]
        if selected_cost is None:
            raise RuntimeError("The full-episode planner selected an unavailable forward move.")
        was_new = False
        prior_visits = 0
        if action == ACTION_FORWARD:
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[before_heading]
            next_row, next_col = before_row + dr, before_col + dc
            if 0 <= next_row < env.config.grid_size and 0 <= next_col < env.config.grid_size:
                prior_visits = int(env.visited_map[next_row, next_col] > 0.5)
                was_new = prior_visits == 0
        next_obs, reward, terminated, truncated, info = env.step(action, switches)
        after_row, after_col = env.agent_pos
        energy_components = {
            "time": float(info.get("time_base_cost", 0.0)),
            "forward": float(info.get("forward_extra_cost", 0.0)),
            "turn": float(info.get("turn_extra_cost", 0.0)),
            "thermal": float(info.get("thermal_extra_this_tick", 0.0)),
        }
        route.append({
            "step": int(info["steps"]),
            "action": ACTION_NAMES[action],
            "position_before": user_coordinate(before_row, before_col, env.config.grid_size),
            "position_after": user_coordinate(after_row, after_col, env.config.grid_size),
            "heading_before": HEADING_NAMES[before_heading],
            "heading_after": HEADING_NAMES[int(env.direction)],
            "turn_direction": "left" if action == ACTION_LEFT else ("right" if action == ACTION_RIGHT else None),
            "opposite_to_previous_turn": bool(previous_action in (ACTION_LEFT, ACTION_RIGHT) and action in (ACTION_LEFT, ACTION_RIGHT) and action != previous_action),
            "moved_to_new_cell": bool(was_new),
            "target_visit_count_before": int(prior_visits),
            "god_mode_exact_energy_before_action": action_costs,
            "selected_action_exact_energy": float(selected_cost),
            "global_plan": {"remaining_episode_actions": len(planned), **plan_info},
            "energy": {"before": before_energy, "spent": float(sum(energy_components.values())), "after": float(next_obs["energy"]), "cost_components": energy_components},
            "health_after": float(next_obs["health"]),
            "health_delta": float(info.get("health_delta", 0.0)),
            "contacted": str(info.get("contacted_label", "Empty")),
            "environment_reward": float(reward),
            "environment_event": str(info.get("last_event", "")),
        })
        forwards += int(action == ACTION_FORWARD)
        turns += int(action in (ACTION_LEFT, ACTION_RIGHT))
        repeats += int(action == ACTION_FORWARD and not was_new)
        health_loss += max(0.0, -float(info.get("health_delta", 0.0)))
        previous_action = action
        obs = next_obs

    summary = {
        "coverage": float(info["coverage"]),
        "steps": float(info["steps"]),
        "forward_actions": float(forwards),
        "turn_actions": float(turns),
        "repeat_forwards": float(repeats),
        "unique_per_forward": float((forwards - repeats) / max(1, forwards)),
        "health_loss": float(health_loss),
        "survived": float(not info["terminated"]),
        "terminated": float(info["terminated"]),
        "global_plan_builds": float(decisions),
    }
    if trace_dir is not None:
        write_trace(trace_dir / f"episode_{episode:05d}_seed_{seed}.json", episode, seed, env, controller, route, summary, initial_special_cells, plan_info, all_planned_actions)
    return summary


def evaluate(cfg: GodModeRunConfig, seeds: Iterable[int], trace_dir: Path | None = None) -> Dict[str, float]:
    rows = [run_episode(cfg, index + 1, seed, trace_dir if trace_dir is not None and (index + 1) % cfg.trace_every == 0 else None) for index, seed in enumerate(seeds)]
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result["coverage_p10"] = float(np.quantile([row["coverage"] for row in rows], 0.10))
    return result


def main() -> None:
    defaults = GodModeRunConfig()
    parser = argparse.ArgumentParser(description="Full-map god-mode coverage oracle")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="First episode uses seed + 1")
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--trace-every", type=int, default=defaults.trace_every)
    parser.add_argument("--search-beam-width", type=int, default=defaults.search_beam_width)
    args = parser.parse_args()
    cfg = GodModeRunConfig(
        episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        trace_every=max(1, args.trace_every),
        search_beam_width=max(1, args.search_beam_width),
    )
    output = Path(cfg.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "god_mode_config.txt").write_text("\n".join(f"{key}: {value}" for key, value in asdict(cfg).items()) + "\n", encoding="utf-8")
    metrics = evaluate(cfg, [cfg.seed + index + 1 for index in range(cfg.episodes)], output / "traces")
    with (output / "god_mode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    (output / "summary.json").write_text(json.dumps({"config": asdict(cfg), "metrics": metrics}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

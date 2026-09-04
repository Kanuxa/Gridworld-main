from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.non_model_based.partial_observation_frontier_planner.core import CoverageMemory, HAZARD, HEADING_NAMES, MEAT, StaticFrontierPlanner, user_coordinate
from models.non_model_based.partial_observation_frontier_planner.trace import EpisodeTrace


BASELINE_PRESET = "15x15-baseline"
FOUR_X_AREA_SCALED_PRESET = "31x31-area-scaled"
AREA_SCALED_PRESET = "45x45-area-scaled"
ENVIRONMENT_PRESETS = (BASELINE_PRESET, FOUR_X_AREA_SCALED_PRESET, AREA_SCALED_PRESET)


def environment_config(preset: str) -> EnvConfig:
    """Return a named, reproducible environment benchmark configuration."""
    if preset == BASELINE_PRESET:
        return EnvConfig(survival_bonus=0.0)
    if preset == FOUR_X_AREA_SCALED_PRESET:
        return EnvConfig(
            grid_size=31,
            patch_size=5,
            init_health=40,
            max_health=40,
            init_energy=10.0,
            max_energy=10.0,
            max_steps=1000,
            n_fire=8,
            n_ice=4,
            n_meat=12,
            n_flower=8,
            n_glass=8,
            fire_damage=3,
            ice_damage=3,
            glass_damage=2,
            meat_heal=2,
            survival_bonus=0.0,
        )
    if preset == AREA_SCALED_PRESET:
        return EnvConfig(
            grid_size=45,
            patch_size=5,
            init_health=90,
            max_health=90,
            init_energy=10.0,
            max_energy=10.0,
            max_steps=2250,
            n_fire=18,
            n_ice=9,
            n_meat=27,
            n_flower=18,
            n_glass=18,
            fire_damage=3,
            ice_damage=3,
            glass_damage=2,
            meat_heal=2,
            survival_bonus=0.0,
        )
    raise ValueError(f"Unknown environment preset: {preset}. Choose one of: {', '.join(ENVIRONMENT_PRESETS)}")


@dataclass
class PlannerLabConfig:
    episodes: int = 50
    seed: int = 50000
    save_dir: str = "runs/runs_15x15/runs_planner/baseline"
    trace_every: int = 10
    max_consecutive_turns: int = 2
    meat_conserve_health_norm: float = 0.80
    revisit_cost: float = 1.80
    repeat_visit_cost: float = 0.35
    hazard_cost: float = 10.0
    reserve_health_norm: float = 0.50
    resource_energy_margin: float = 4.0
    frontier_info_weight: float = 0.14
    frontier_cost_weight: float = 0.035
    frontier_hazard_weight: float = 0.65
    comfort_low_c: float = 18.0
    comfort_high_c: float = 24.0
    thermal_extra_energy_max: float = 0.50
    discomfort_temp_scale_c: float = 10.0
    force_safe_frontier_forward: bool = True
    environment_preset: str = BASELINE_PRESET


def build_env(preset: str = BASELINE_PRESET) -> Tuple[SensoryGridEnv, ObservationSwitches]:
    env = SensoryGridEnv(environment_config(preset))
    switches = ObservationSwitches(
        include_vision=True,
        include_temperature=False,
        include_smell=False,
        include_temperature_patch=True,
        include_smell_patch=True,
        include_visited_memory=True,
        include_hazard_memory=True,
    )
    return env, switches


def visible_safe_unvisited_directions(memory: CoverageMemory) -> List[str]:
    result = []
    for direction, name in enumerate(HEADING_NAMES):
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
        row, col = memory.pos[0] + dr, memory.pos[1] + dc
        if 0 <= row < memory.grid_size and 0 <= col < memory.grid_size and memory.visit_count[row, col] == 0 and memory.map[HAZARD, row, col] < 0.5:
            result.append(name)
    return result


def directed_turn_action(memory: CoverageMemory, target: int, planner_action: int, mask: np.ndarray) -> int | None:
    target_row, target_col = divmod(int(target), memory.grid_size)
    current_distance = abs(target_row - memory.pos[0]) + abs(target_col - memory.pos[1])
    candidates = []
    for action, turn in ((ACTION_LEFT, -1), (ACTION_RIGHT, 1)):
        if mask[action] < -1e8:
            continue
        direction = (memory.direction + turn) % 4
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
        row, col = memory.pos[0] + dr, memory.pos[1] + dc
        safe_new = 0 <= row < memory.grid_size and 0 <= col < memory.grid_size and memory.visit_count[row, col] == 0 and memory.map[HAZARD, row, col] < 0.5
        score = 10.0 * float(safe_new) + float(current_distance - abs(target_row - row) - abs(target_col - col)) + 0.25 * float(action == planner_action)
        candidates.append((score, -action, action))
    return max(candidates)[2] if candidates else None


def choose_action(memory: CoverageMemory, health_norm: float, turn_streak: int, previous_action: int | None, target: int, planner_action: int, meat_mode: bool, cfg: PlannerLabConfig) -> Tuple[int, np.ndarray, Dict[str, object]]:
    """Apply the local 5x5 anti-loop rules to the global frontier suggestion."""
    mask = np.zeros(3, dtype=np.float32)
    row, col = memory.forward_target()
    if not (0 <= row < memory.grid_size and 0 <= col < memory.grid_size):
        forward_status = "wall"
    elif memory.map[HAZARD, row, col] > 0.5:
        forward_status = "hazard"
    elif memory.visit_count[row, col] > 0:
        forward_status = "visited"
    else:
        forward_status = "unvisited"
    safe_directions = visible_safe_unvisited_directions(memory)
    safe_nonforward = [direction for direction in safe_directions if direction != HEADING_NAMES[memory.direction]]
    forward_is_meat = 0 <= row < memory.grid_size and 0 <= col < memory.grid_size and memory.map[MEAT, row, col] > 0.5
    conserve_meat = not meat_mode and forward_is_meat and health_norm > cfg.meat_conserve_health_norm and bool(safe_nonforward)
    resource_bridge = meat_mode and planner_action == ACTION_FORWARD and forward_status in {"visited", "unvisited"}
    turn_escape = turn_streak >= cfg.max_consecutive_turns and forward_status in {"visited", "hazard"} and bool(safe_nonforward) and not resource_bridge

    if turn_streak >= cfg.max_consecutive_turns and not turn_escape:
        mask[ACTION_LEFT:] = -1e9
    elif previous_action == ACTION_LEFT and not turn_escape:
        mask[ACTION_RIGHT] = -1e9
    elif previous_action == ACTION_RIGHT and not turn_escape:
        mask[ACTION_LEFT] = -1e9
    if turn_streak == 1 and forward_status == "unvisited" and not (meat_mode and planner_action != ACTION_FORWARD):
        mask[ACTION_LEFT:] = -1e9

    forward_blocked = conserve_meat or forward_status == "wall" or (not resource_bridge and forward_status in {"hazard", "visited"} and bool(safe_nonforward))
    if forward_blocked:
        mask[ACTION_FORWARD] = -1e9
    directed_turn = directed_turn_action(memory, target, planner_action, mask) if forward_blocked or turn_escape else None
    if directed_turn is not None and (forward_blocked or turn_escape):
        mask[[index for index in range(3) if index != directed_turn]] = -1e9

    if meat_mode:
        action, reason = planner_action, "meat_recovery"
    elif turn_escape and directed_turn is not None:
        action, reason = directed_turn, "turn_escape"
    elif cfg.force_safe_frontier_forward and forward_status == "unvisited" and not conserve_meat:
        action, reason = ACTION_FORWARD, "safe_frontier_forward"
    else:
        action, reason = planner_action, "planner_controller"
    if mask[action] < -1e8:
        action = directed_turn_action(memory, target, planner_action, mask)
        reason = "safe_turn_fallback" if action is not None else "masked_planner_fallback"
    if action is None:
        allowed = np.flatnonzero(mask >= -1e8)
        if allowed.size:
            action = ACTION_FORWARD if mask[ACTION_FORWARD] >= -1e8 else int(allowed[0])
        else:
            # A conflicting turn cap and local forward block can eliminate
            # every action.  Preserve a deterministic, recoverable escape
            # instead of indexing an empty set of legal actions.
            mask[ACTION_FORWARD] = 0.0
            action, reason = ACTION_FORWARD, "emergency_forward"
    mask[[index for index in range(3) if index != action]] = -1e9
    context: Dict[str, object] = {
        "forward_status": forward_status,
        "safe_unvisited_directions": safe_directions,
        "planner_action": planner_action,
        "directed_turn_action": directed_turn,
        "forward_blocked_by_local_rule": forward_blocked,
        "resource_bridge_override": resource_bridge,
        "conserve_forward_meat": conserve_meat,
        "turn_escape": turn_escape,
        "forced_action": action,
        "forced_action_reason": reason,
    }
    return int(action), mask, context


def run_episode(
    cfg: PlannerLabConfig,
    episode: int,
    seed: int,
    trace_dir: Path | None = None,
    env: SensoryGridEnv | None = None,
) -> Dict[str, float]:
    """Run the planner once, optionally against an observation-compatible environment.

    ``env`` supports matched robustness benchmarks without changing the normal
    command-line planner behaviour.  It must expose the same public interface
    as :class:`SensoryGridEnv`; the controller still receives only its public
    observation dictionary.
    """
    if env is None:
        env, switches = build_env(cfg.environment_preset)
    else:
        _, switches = build_env(cfg.environment_preset)
    planner = StaticFrontierPlanner(
        revisit_cost=cfg.revisit_cost, repeat_visit_cost=cfg.repeat_visit_cost, hazard_cost=cfg.hazard_cost,
        reserve_health_norm=cfg.reserve_health_norm, resource_energy_margin=cfg.resource_energy_margin,
        frontier_info_weight=cfg.frontier_info_weight, frontier_cost_weight=cfg.frontier_cost_weight,
        frontier_hazard_weight=cfg.frontier_hazard_weight,
        comfort_low_c=cfg.comfort_low_c, comfort_high_c=cfg.comfort_high_c,
        thermal_extra_energy_max=cfg.thermal_extra_energy_max,
        discomfort_temp_scale_c=cfg.discomfort_temp_scale_c,
    )
    obs, _ = env.reset(seed=seed)
    memory = CoverageMemory(env.config.grid_size, env.config.patch_size, env.config.ambient_temperature_c)
    memory.reset(int(obs["direction"]))
    memory.update(obs)
    trace = EpisodeTrace(trace_dir, episode, seed, env, obs) if trace_dir else None
    forwards = repeats = 0
    health_loss = 0.0
    previous_action: int | None = None
    while True:
        prior, target, meat_mode = planner.action_prior(memory, float(obs["health_norm"]), float(obs["energy_norm"]))
        planner_action = int(np.argmax(prior))
        action, mask, context = choose_action(memory, float(obs["health_norm"]), int(obs["consecutive_turn_steps"]), previous_action, target, planner_action, meat_mode, cfg)
        expected_energy, expected_thermal, expected_temperature = planner.action_energy_estimate(memory, action)
        _, forward_thermal, forward_temperature = planner.action_energy_estimate(memory, ACTION_FORWARD)
        context.update({"resource_mode": "meat" if meat_mode else "coverage", "planner_expected_action_energy": expected_energy, "planner_expected_thermal_energy": expected_thermal, "planner_target_temperature_c": expected_temperature, "forward_expected_thermal_energy": forward_thermal, "forward_temperature_c": forward_temperature})
        was_new = memory.forward_is_new() if action == ACTION_FORWARD else False
        prior_visits = 0
        if action == ACTION_FORWARD:
            next_row, next_col = memory.forward_target()
            if 0 <= next_row < memory.grid_size and 0 <= next_col < memory.grid_size:
                prior_visits = int(memory.visit_count[next_row, next_col])
        before_position, before_heading = user_coordinate(*memory.pos, memory.grid_size), memory.direction
        energy_before, turn_streak = float(obs["energy"]), int(obs["consecutive_turn_steps"])
        memory.advance(action)
        next_obs, reward, terminated, truncated, info = env.step(action, switches)
        memory.update(next_obs)
        if trace:
            trace.add(action, before_position, user_coordinate(*memory.pos, memory.grid_size), before_heading, memory.direction, energy_before, next_obs, reward, info, was_new, prior_visits, turn_streak, previous_action, context, mask, target)
        forwards += int(action == ACTION_FORWARD)
        repeats += int(action == ACTION_FORWARD and not was_new)
        health_loss += max(0.0, -float(info["health_delta"]))
        previous_action, obs = action, next_obs
        if terminated or truncated:
            break
    summary = {"coverage": float(info["coverage"]), "steps": float(info["steps"]), "forward_actions": float(forwards), "repeat_forwards": float(repeats), "unique_per_forward": float((forwards - repeats) / max(1, forwards)), "health_loss": health_loss, "survived": float(not info["terminated"]), "terminated": float(info["terminated"])}
    if trace:
        trace.write(summary)
    return summary


def evaluate(cfg: PlannerLabConfig, seeds: Iterable[int], trace_dir: Path | None = None) -> Dict[str, float]:
    rows = [run_episode(cfg, index + 1, seed, trace_dir if trace_dir and (index + 1) % cfg.trace_every == 0 else None) for index, seed in enumerate(seeds)]
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result["coverage_p10"] = float(np.quantile([row["coverage"] for row in rows], 0.10))
    return result


def write_row(path: Path, cfg: PlannerLabConfig, metrics: Dict[str, float]) -> None:
    row = {**asdict(cfg), **metrics}
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def values(text: str) -> List[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    defaults = PlannerLabConfig()
    parser = argparse.ArgumentParser(description="Run standalone non-neural planner experiments")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="First run uses seed + 1")
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--trace-every", type=int, default=defaults.trace_every)
    parser.add_argument("--preset", choices=ENVIRONMENT_PRESETS, default=defaults.environment_preset)
    parser.add_argument("--resource-energy-margin", type=float, default=defaults.resource_energy_margin)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--revisit-costs", default="1.8")
    parser.add_argument("--frontier-cost-weights", default="0.035")
    parser.add_argument("--resource-energy-margins", default="4.0")
    parser.add_argument("--allow-planner-turns-on-safe-frontier", action="store_true")
    args = parser.parse_args()
    base = PlannerLabConfig(episodes=args.episodes, seed=args.seed, save_dir=args.save_dir, trace_every=max(1, args.trace_every), resource_energy_margin=args.resource_energy_margin, force_safe_frontier_forward=not args.allow_planner_turns_on_safe_frontier, environment_preset=args.preset)
    output = Path(base.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "planner_config.txt").write_text(
        "[planner]\n"
        + "\n".join(f"{key}: {value}" for key, value in asdict(base).items())
        + "\n\n[environment]\n"
        + "\n".join(f"{key}: {value}" for key, value in asdict(environment_config(base.environment_preset)).items())
        + "\n",
        encoding="utf-8",
    )
    seeds = [base.seed + index + 1 for index in range(base.episodes)]
    if not args.sweep:
        metrics = evaluate(base, seeds, output / "traces")
        write_row(output / "planner_metrics.csv", base, metrics)
        (output / "summary.json").write_text(json.dumps({"config": asdict(base), "environment_config": asdict(environment_config(base.environment_preset)), "metrics": metrics}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return
    for revisit, frontier, margin in product(values(args.revisit_costs), values(args.frontier_cost_weights), values(args.resource_energy_margins)):
        candidate = replace(base, revisit_cost=revisit, frontier_cost_weight=frontier, resource_energy_margin=margin)
        metrics = evaluate(candidate, seeds)
        write_row(output / "planner_sweep.csv", candidate, metrics)
        print(json.dumps({"revisit_cost": revisit, "frontier_cost_weight": frontier, "resource_energy_margin": margin, **metrics}), flush=True)


if __name__ == "__main__":
    main()

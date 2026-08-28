"""Evaluate a safe explorer across different stationary episode worlds.

V12 keeps planning and safety deterministic while allowing a neural frontier
ensemble to rank phase-1 targets.  It uses only the agent's sensory
observations and transition outcomes.  Each episode starts in a new seeded
world; optional profile families vary hazard/resource mixes and thermal context
with no exogenous environment changes during an episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LABELS, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.shared.belief_map_tools import BeliefMap, HeadingAwareAStar, PhasePlan, select_phase1_plan, select_phase2_plan
from models.shared.belief_map_tools.environment_profiles import (
    environment_config_for_seed,
    environment_profile_configs,
    environment_set_names,
)
from models.shared.belief_map_tools.trajectory_dataset import Phase1TrajectoryWriter
from models.shared.belief_map_tools.trace_logging import TRACE_SCHEMA_VERSION, EpisodeTraceWriter


@dataclass(frozen=True)
class AdaptiveExplorerConfig:
    seen_phase_threshold: float = 0.80


@dataclass
class AdaptiveEpisodeResult:
    seed: int
    environment_set: str
    environment_variant: str
    seen_fraction: float
    coverage: float
    phase: int
    reached_phase2: bool
    final_mode: str
    actions: int
    forward_actions: int
    turns: int
    replans: int
    information_replans: int
    coverage_replans: int
    observation_surprises: int
    object_changes: int
    temperature_changes: int
    smell_changes: int
    direct_hazard_entries: int
    meat_collected: int
    fatigue_health_losses: int
    health: float
    energy: float
    terminated: bool
    truncated: bool
    neural_phase1_decisions: int
    neural_uncertainty_fallbacks: int
    neural_feasibility_fallbacks: int
    end_reason: str
    phase2_entry_step: int
    phase2_entry_seen_fraction: float | None
    phase2_entry_coverage: float | None
    phase2_entry_health: float | None
    phase2_entry_energy: float | None
    phase1_forward_actions: int
    phase2_forward_actions: int
    phase1_turns: int
    phase2_turns: int
    wall_bumps: int
    time_energy_cost_total: float
    forward_energy_cost_total: float
    turn_energy_cost_total: float
    thermal_energy_cost_total: float
    meat_health_restored: float
    meat_health_wasted: float
    direct_hazard_health_losses: int
    trace_file: str


def mode_for_phase(phase: int) -> str:
    """Keep V11's two-phase schedule explicit in V12 result records."""
    return "information" if phase == 1 else "coverage"


def load_frontier_models(checkpoints: Sequence[Path], device_name: str):
    """Load optional committee weights once for a multi-episode evaluation."""
    if not checkpoints:
        return (), None
    import torch

    from models.shared.belief_map_tools.adaptive_selectors import load_frontier_model

    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    return tuple(load_frontier_model(path, device) for path in checkpoints), device


def make_ensemble_selector(checkpoints: Sequence[Path], device_name: str, loaded_models: Sequence[Any] | None = None, loaded_device: Any = None):
    """Construct episode-local selector statistics without reloading weights."""
    if loaded_models is None:
        loaded_models, loaded_device = load_frontier_models(checkpoints, device_name)
    if not loaded_models:
        return None
    from models.shared.belief_map_tools.adaptive_selectors import EnsemblePhase1Selector

    return EnsemblePhase1Selector(loaded_models, loaded_device)


def select_plan(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    phase: int,
    neural_selector: Any,
) -> PhasePlan | None:
    """Use the ensemble only for phase-1 target ranking.

    With no checkpoint this is deliberately the same target selection as V11.
    Phase 2 remains deterministic because the current learned phase-2 scorer
    is not competitive with the expert.
    """
    if phase == 2:
        return select_phase2_plan(belief, router)
    if neural_selector is not None:
        return neural_selector(belief, router)
    return select_phase1_plan(belief, router)


def _value_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def belief_snapshot(belief: BeliefMap) -> dict[str, Any]:
    """Compact planner state derived only from the agent-owned belief map."""
    phase1_candidates = 0
    for row, col in zip(*np.where(belief.phase1_safe)):
        phase1_candidates += int(belief.visibility_gain((int(row), int(col))) > 0)
    known_temperature = belief.temperature[belief.temperature_known]
    known_smell = belief.smell[belief.smell_known]
    return {
        "position": [int(belief.position[0]), int(belief.position[1])],
        "heading": int(belief.heading),
        "steps": int(belief.steps),
        "health": float(belief.health),
        "energy": float(belief.energy),
        "seen_fraction": float(belief.seen_fraction),
        "visited_fraction": float(belief.coverage_fraction),
        "seen_cells": int(np.sum(belief.seen)),
        "visited_cells": int(np.sum(belief.visited)),
        "known_hazard_cells": int(np.sum(belief.direct_hazard)),
        "phase1_candidate_count": int(phase1_candidates),
        "visible_meat_cells": int(np.sum(belief.meat)),
        "visible_flower_cells": int(np.sum(belief.flower)),
        "temperature": _value_summary(known_temperature),
        "smell": _value_summary(known_smell),
    }


def observation_snapshot(observation: dict[str, Any]) -> dict[str, Any]:
    """Raw 5x5 sensory values returned to the agent, never hidden world state."""
    keys = ("vision", "temperature_patch_c", "smell_patch", "direction", "health", "energy")
    return {key: observation[key] for key in keys if key in observation}


def plan_snapshot(plan: PhasePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "phase": int(plan.phase),
        "target": [int(plan.target[0]), int(plan.target[1])],
        "score": float(plan.score),
        "visibility_gain": int(plan.visibility_gain),
        "new_visited_along_route": int(plan.new_visited_along_route),
        "projected_energy_cost": float(plan.projected_energy_cost),
        "projected_health_after_route": float(plan.projected_health_after_route),
        "projected_health_at_horizon": float(plan.projected_health_at_horizon),
        "survival_feasible": bool(plan.survival_feasible),
        "route": {
            "cost": float(plan.route.cost),
            "length": int(len(plan.route.actions)),
            "forwards": int(sum(action == ACTION_FORWARD for action in plan.route.actions)),
            "turns": int(plan.route.turns),
            "used_fallback": bool(plan.route.used_fallback),
            "actions": [
                {"id": int(action), "label": ACTION_LABELS[int(action)]}
                for action in plan.route.actions
            ],
            "cells": [[int(row), int(col)] for row, col in plan.route.cells],
        },
    }


def plan_comparison(selected: PhasePlan | None, expert: PhasePlan | None) -> dict[str, Any] | None:
    """Compare targets/routes without treating neural and expert score scales as equal."""
    if selected is None or expert is None:
        return None
    return {
        "target_matches_expert": bool(selected.target == expert.target),
        "target_manhattan_from_expert": int(
            abs(selected.target[0] - expert.target[0]) + abs(selected.target[1] - expert.target[1])
        ),
        "visibility_gain_delta": int(selected.visibility_gain - expert.visibility_gain),
        "new_visited_delta": int(selected.new_visited_along_route - expert.new_visited_along_route),
        "route_cost_delta": float(selected.route.cost - expert.route.cost),
        "route_length_delta": int(len(selected.route.actions) - len(expert.route.actions)),
    }


def selection_snapshot(neural_selector: Any, phase: int) -> dict[str, Any]:
    if phase == 2:
        return {"source": "phase2_expert"}
    if neural_selector is None:
        return {"source": "deterministic_expert"}
    return asdict(neural_selector.last_decision)


def run_episode(
    seed: int,
    *,
    env_config: EnvConfig | None = None,
    controller_config: AdaptiveExplorerConfig | None = None,
    frontier_checkpoints: Sequence[Path] = (),
    device_name: str = "auto",
    frontier_models: Sequence[Any] | None = None,
    frontier_device: Any = None,
    phase1_trajectory: Phase1TrajectoryWriter | None = None,
    environment_set: str = "custom",
    environment_variant: str = "custom",
    trace_dir: Path | None = None,
    trace_observations: bool = False,
) -> AdaptiveEpisodeResult:
    """Run one V11-equivalent safe episode with an optional phase-1 ensemble."""
    env = SensoryGridEnv(env_config or EnvConfig())
    controller = controller_config or AdaptiveExplorerConfig()
    switches = ObservationSwitches()
    observation, _ = env.reset(seed=seed)
    belief = BeliefMap(
        env.config.grid_size,
        env.config.patch_size,
        max_health=env.config.max_health,
        max_energy=env.config.max_energy,
        time_energy_cost=env.config.time_energy_cost,
        forward_energy_cost=env.config.forward_energy_cost,
        turn_energy_cost=env.config.turn_energy_cost,
        thermal_extra_energy_max=env.config.thermal_extra_energy_max,
        max_steps=env.config.max_steps,
    )
    belief.reset(observation)
    router = HeadingAwareAStar()
    neural_selector = make_ensemble_selector(frontier_checkpoints, device_name, frontier_models, frontier_device)
    trace = EpisodeTraceWriter(trace_dir, seed) if trace_dir is not None else None
    trace_file = trace.path.name if trace is not None else ""
    if trace is not None:
        trace.record(
            "episode_start",
            seed=int(seed),
            environment_set=environment_set,
            environment_variant=environment_variant,
            environment_config=asdict(env.config),
            controller=asdict(controller),
            frontier_checkpoints=[str(path) for path in frontier_checkpoints],
            observation_switches=asdict(switches),
            initial_belief=belief_snapshot(belief),
            initial_observation=observation_snapshot(observation) if trace_observations else None,
        )

    phase = 1
    plan_index = 0
    active_plan_id: int | None = None

    def make_plan(replan_reason: str) -> PhasePlan | None:
        """Select and optionally trace a plan from the current agent belief."""
        nonlocal active_plan_id, plan_index
        selected_plan = select_plan(belief, router, phase, neural_selector)
        expert_plan: PhasePlan | None = None
        if phase == 1 and (phase1_trajectory is not None or trace is not None):
            # DAgger label/counterfactual: retain the V11 expert target from
            # the same learner belief, without reading the hidden world.
            expert_plan = select_phase1_plan(belief, router)
            if phase1_trajectory is not None and expert_plan is not None:
                phase1_trajectory.record(belief, expert_plan)
        if trace is not None:
            plan_index += 1
            active_plan_id = plan_index
            trace.record(
                "plan",
                seed=int(seed),
                plan_id=active_plan_id,
                episode_step=int(env.steps),
                phase=int(phase),
                mode=mode_for_phase(phase),
                replan_reason=replan_reason,
                belief=belief_snapshot(belief),
                selection=selection_snapshot(neural_selector, phase),
                selected_plan=plan_snapshot(selected_plan),
                expert_plan=plan_snapshot(expert_plan),
                comparison_to_expert=plan_comparison(selected_plan, expert_plan),
            )
        return selected_plan

    plan = make_plan("initial")
    route_actions = list(plan.route.actions) if plan is not None else []
    final_mode = "information"
    actions = forward_actions = turns = replans = 0
    information_replans = coverage_replans = 0
    observation_surprises = object_changes = temperature_changes = smell_changes = 0
    direct_hazard_entries = meat_collected = fatigue_health_losses = 0
    phase2_entry_step = -1
    phase2_entry_seen_fraction = phase2_entry_coverage = None
    phase2_entry_health = phase2_entry_energy = None
    phase1_forward_actions = phase2_forward_actions = phase1_turns = phase2_turns = 0
    wall_bumps = 0
    time_energy_cost_total = forward_energy_cost_total = turn_energy_cost_total = thermal_energy_cost_total = 0.0
    meat_health_restored = meat_health_wasted = 0.0
    direct_hazard_health_losses = 0
    stopped_without_plan = False

    while not env.terminated and not env.truncated:
        if not route_actions:
            plan = make_plan("route_exhausted")
            if plan is None or not plan.route.actions:
                stopped_without_plan = True
                break
            route_actions = list(plan.route.actions)
            replans += 1
            if phase == 1:
                information_replans += 1
            else:
                coverage_replans += 1
            final_mode = mode_for_phase(phase)

        phase_before_action = phase
        executed_plan_id = active_plan_id
        executed_target = None if plan is None else [int(plan.target[0]), int(plan.target[1])]
        route_actions_before = len(route_actions)
        pre_belief = belief_snapshot(belief) if trace is not None else None
        action = route_actions.pop(0)
        health_before = float(env.health)
        energy_before = float(env.energy)
        observation, reward, terminated, truncated, info = env.step(action, switches)
        belief.update_after_action(action, observation, info)
        delta = belief.last_delta
        actions += 1
        forward_actions += int(action == ACTION_FORWARD)
        turns += int(action != ACTION_FORWARD)
        if phase_before_action == 1:
            phase1_forward_actions += int(action == ACTION_FORWARD)
            phase1_turns += int(action != ACTION_FORWARD)
        else:
            phase2_forward_actions += int(action == ACTION_FORWARD)
            phase2_turns += int(action != ACTION_FORWARD)
        object_changes += delta.object_changes
        temperature_changes += delta.temperature_changes
        smell_changes += delta.smell_changes
        observation_surprises += int(delta.surprise)
        time_energy_cost_total += float(info.get("time_base_cost", 0.0))
        forward_energy_cost_total += float(info.get("forward_extra_cost", 0.0))
        turn_energy_cost_total += float(info.get("turn_extra_cost", 0.0))
        thermal_energy_cost_total += float(info.get("thermal_extra_this_tick", 0.0))
        wall_bumps += int(action == ACTION_FORWARD and "Bumped into wall" in str(info.get("last_event", "")))

        contacted = str(info.get("contacted_label", ""))
        meat_collected += int(contacted == "Meat")
        health_delta = int(info.get("health_delta", 0))
        if contacted == "Meat":
            restored = min(float(env.config.meat_heal), max(0.0, float(env.config.max_health) - health_before))
            meat_health_restored += restored
            meat_health_wasted += float(env.config.meat_heal) - restored
        if contacted in {"Fire", "Ice", "Glass"}:
            direct_hazard_health_losses += max(0, -health_delta)
        elif health_delta < 0:
            fatigue_health_losses += max(0, int(round(health_before - float(env.health))))
        if action == ACTION_FORWARD:
            row, col = belief.position
            direct_hazard_entries += int(belief.direct_hazard[row, col])

        if trace is not None:
            trace.record(
                "step",
                seed=int(seed),
                episode_step=int(actions),
                phase_before_action=int(phase_before_action),
                mode_before_action=mode_for_phase(phase_before_action),
                plan_id=executed_plan_id,
                active_target=executed_target,
                route_actions_before=int(route_actions_before),
                route_actions_after_action=int(len(route_actions)),
                action={"id": int(action), "label": ACTION_LABELS[int(action)]},
                reward=float(reward),
                health_before=health_before,
                energy_before=energy_before,
                health_after=float(env.health),
                energy_after=float(env.energy),
                transition_info=info,
                contacted_label=contacted,
                wall_bump=bool(action == ACTION_FORWARD and "Bumped into wall" in str(info.get("last_event", ""))),
                direct_hazard_at_position=bool(
                    action == ACTION_FORWARD and belief.direct_hazard[belief.position]
                ),
                belief_before=pre_belief,
                belief_after=belief_snapshot(belief),
                belief_delta=asdict(delta),
                observation=observation_snapshot(observation) if trace_observations else None,
            )

        # Match V11's stationary-world execution schedule: phase 1 replans
        # after every forward move; phase 2 replans only when forward movement
        # reveals information or normal resource consumption changes survival.
        if action == ACTION_FORWARD:
            if phase == 1 and belief.seen_fraction >= controller.seen_phase_threshold:
                phase = 2
                phase2_entry_step = int(actions)
                phase2_entry_seen_fraction = float(belief.seen_fraction)
                phase2_entry_coverage = float(env.current_scalars()["coverage"])
                phase2_entry_health = float(env.health)
                phase2_entry_energy = float(env.energy)
                plan = make_plan("phase_transition")
                route_actions = list(plan.route.actions) if plan is not None else []
                replans += 1
                coverage_replans += 1
                final_mode = mode_for_phase(phase)
            elif phase == 1:
                plan = make_plan("phase1_forward")
                route_actions = list(plan.route.actions) if plan is not None else []
                replans += 1
                information_replans += 1
                final_mode = mode_for_phase(phase)
            elif delta.newly_seen > 0:
                plan = make_plan("phase2_newly_seen")
                route_actions = list(plan.route.actions) if plan is not None else []
                replans += 1
                coverage_replans += 1
                final_mode = mode_for_phase(phase)

            if phase == 2 and contacted == "Meat":
                plan = make_plan("meat_contact")
                route_actions = list(plan.route.actions) if plan is not None else []
                replans += 1
                coverage_replans += 1
                final_mode = mode_for_phase(phase)
        if terminated or truncated:
            break

    if neural_selector is None:
        neural_decisions = uncertainty_fallbacks = feasibility_fallbacks = 0
    else:
        stats = neural_selector.stats
        neural_decisions = stats.model_decisions
        uncertainty_fallbacks = stats.uncertainty_fallbacks
        feasibility_fallbacks = stats.feasibility_fallbacks
    if env.terminated:
        end_reason = "health_zero_direct_hazard" if direct_hazard_health_losses > 0 else (
            "health_zero_fatigue" if fatigue_health_losses > 0 else "health_zero_other"
        )
    elif env.truncated:
        end_reason = "max_steps"
    elif stopped_without_plan:
        end_reason = "no_plan"
    else:
        end_reason = "stopped"
    result = AdaptiveEpisodeResult(
        seed=seed,
        environment_set=environment_set,
        environment_variant=environment_variant,
        seen_fraction=belief.seen_fraction,
        coverage=float(env.current_scalars()["coverage"]),
        phase=phase,
        reached_phase2=phase == 2,
        final_mode=final_mode,
        actions=actions,
        forward_actions=forward_actions,
        turns=turns,
        replans=replans,
        information_replans=information_replans,
        coverage_replans=coverage_replans,
        observation_surprises=observation_surprises,
        object_changes=object_changes,
        temperature_changes=temperature_changes,
        smell_changes=smell_changes,
        direct_hazard_entries=direct_hazard_entries,
        meat_collected=meat_collected,
        fatigue_health_losses=fatigue_health_losses,
        health=float(env.health),
        energy=float(env.energy),
        terminated=bool(env.terminated),
        truncated=bool(env.truncated),
        neural_phase1_decisions=neural_decisions,
        neural_uncertainty_fallbacks=uncertainty_fallbacks,
        neural_feasibility_fallbacks=feasibility_fallbacks,
        end_reason=end_reason,
        phase2_entry_step=phase2_entry_step,
        phase2_entry_seen_fraction=phase2_entry_seen_fraction,
        phase2_entry_coverage=phase2_entry_coverage,
        phase2_entry_health=phase2_entry_health,
        phase2_entry_energy=phase2_entry_energy,
        phase1_forward_actions=phase1_forward_actions,
        phase2_forward_actions=phase2_forward_actions,
        phase1_turns=phase1_turns,
        phase2_turns=phase2_turns,
        wall_bumps=wall_bumps,
        time_energy_cost_total=time_energy_cost_total,
        forward_energy_cost_total=forward_energy_cost_total,
        turn_energy_cost_total=turn_energy_cost_total,
        thermal_energy_cost_total=thermal_energy_cost_total,
        meat_health_restored=meat_health_restored,
        meat_health_wasted=meat_health_wasted,
        direct_hazard_health_losses=direct_hazard_health_losses,
        trace_file=trace_file,
    )
    if trace is not None:
        trace.record(
            "episode_end",
            seed=int(seed),
            end_reason=end_reason,
            final_belief=belief_snapshot(belief),
            result=asdict(result),
        )
        trace.close()
    return result


def mean(results: Sequence[AdaptiveEpisodeResult], name: str) -> float:
    return float(np.mean([float(getattr(result, name)) for result in results]))


def file_sha256(path: Path) -> str:
    """Return a stable checkpoint fingerprint without loading model weights."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(checkpoints: Sequence[Path]) -> list[dict[str, Any]]:
    """Capture the exact neural artifacts used by a run-level config record."""
    records: list[dict[str, Any]] = []
    for path in checkpoints:
        resolved = path.resolve()
        records.append(
            {
                "path": str(resolved),
                "bytes": int(resolved.stat().st_size),
                "sha256": file_sha256(resolved),
            }
        )
    return records


def assert_trace_targets_unused(directory: Path, seed_start: int, episodes: int) -> None:
    """Fail before creating results if a requested trace run would overwrite data."""
    conflicts = [
        directory / f"seed_{seed:05d}.jsonl.gz"
        for seed in range(seed_start, seed_start + episodes)
        if (directory / f"seed_{seed:05d}.jsonl.gz").exists()
    ]
    if conflicts:
        preview = ", ".join(str(path) for path in conflicts[:3])
        suffix = "" if len(conflicts) <= 3 else f" (+{len(conflicts) - 3} more)"
        raise FileExistsError(f"Refusing to overwrite existing trace file(s): {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/15x15/learned_frontier_ranker/evaluation.csv"))
    parser.add_argument("--frontier-checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=EnvConfig.max_steps)
    parser.add_argument("--collect-phase1-trajectories", type=Path, default=None)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional directory for one agent-observable JSONL.gz trace per seed.",
    )
    parser.add_argument(
        "--trace-observations",
        action="store_true",
        help="Include raw 5x5 agent sensory patches in traces (larger files).",
    )
    parser.add_argument(
        "--environment-set",
        choices=environment_set_names(),
        default="standard",
        help="Stationary profile family; every seed still creates a different layout.",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be positive")
    if args.trace_observations and args.trace_dir is None:
        parser.error("--trace-observations requires --trace-dir")
    if args.trace_dir is not None:
        try:
            assert_trace_targets_unused(args.trace_dir, args.seed, args.episodes)
        except FileExistsError as exc:
            parser.error(str(exc))

    controller = AdaptiveExplorerConfig()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "controller": asdict(controller),
                "environment_set": args.environment_set,
                "environment_profiles": environment_profile_configs(args.environment_set, max_steps=args.max_steps),
                "max_steps": args.max_steps,
                "frontier_checkpoints": [str(path) for path in args.frontier_checkpoint],
                "frontier_checkpoint_metadata": checkpoint_metadata(args.frontier_checkpoint),
                "device": args.device,
                "trace_dir": None if args.trace_dir is None else str(args.trace_dir),
                "trace_observations": bool(args.trace_observations),
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "seed_start": args.seed,
                "episodes": args.episodes,
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[AdaptiveEpisodeResult] = []
    frontier_models, frontier_device = load_frontier_models(args.frontier_checkpoint, args.device)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AdaptiveEpisodeResult.__dataclass_fields__))
        writer.writeheader()
        for offset in range(args.episodes):
            seed = args.seed + offset
            environment_variant, env_config = environment_config_for_seed(
                seed,
                args.environment_set,
                max_steps=args.max_steps,
            )
            trajectory = Phase1TrajectoryWriter(seed, adaptive_channels=True) if args.collect_phase1_trajectories else None
            result = run_episode(
                seed,
                env_config=env_config,
                controller_config=controller,
                frontier_checkpoints=args.frontier_checkpoint,
                device_name=args.device,
                frontier_models=frontier_models,
                frontier_device=frontier_device,
                phase1_trajectory=trajectory,
                environment_set=args.environment_set,
                environment_variant=environment_variant,
                trace_dir=args.trace_dir,
                trace_observations=args.trace_observations,
            )
            results.append(result)
            writer.writerow(asdict(result))
            handle.flush()
            if trajectory is not None:
                trajectory.save(args.collect_phase1_trajectories)
            print(
                "episode={:03d}/{:03d} seed={} env={} seen={:.3f} coverage={:.3f} mode={} "
                "surprises={} replans={} hazards={} status={}".format(
                    offset + 1,
                    args.episodes,
                    seed,
                    result.environment_variant,
                    result.seen_fraction,
                    result.coverage,
                    result.final_mode,
                    result.observation_surprises,
                    result.replans,
                    result.direct_hazard_entries,
                    "terminated" if result.terminated else "truncated" if result.truncated else "stopped",
                ),
                flush=True,
            )
    print(
        "episodes={} seen_mean={:.3f} coverage_mean={:.3f} surprises_mean={:.2f} "
        "hazard_entries={} neural_decisions={}".format(
            len(results),
            mean(results, "seen_fraction"),
            mean(results, "coverage"),
            mean(results, "observation_surprises"),
            sum(result.direct_hazard_entries for result in results),
            sum(result.neural_phase1_decisions for result in results),
        )
    )
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

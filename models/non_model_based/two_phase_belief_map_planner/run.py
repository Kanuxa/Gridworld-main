"""Run the deterministic two-phase v11 expert planner on seeded environments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, EnvConfig, ObservationSwitches, SensoryGridEnv
from models.shared.belief_map_tools import BeliefMap, HeadingAwareAStar, Phase1TrajectoryWriter, Phase2TrajectoryWriter, PhasePlan, select_phase1_plan, select_phase2_plan
from models.shared.belief_map_tools.environment_profiles import (
    environment_config_for_seed,
    environment_profile_configs,
    environment_set_names,
)


@dataclass
class EpisodeResult:
    seed: int
    environment_set: str
    environment_variant: str
    seen_fraction: float
    coverage: float
    phase: int
    reached_phase2: bool
    actions: int
    turns: int
    replans: int
    phase1_replans: int
    phase2_replans: int
    survival_mode_actions: int
    phase1_actions: int
    phase1_turns: int
    phase1_end_seen_fraction: float
    phase1_end_health: float
    phase1_end_energy: float
    phase1_temperature_discomfort_total: float
    phase1_expert_decisions: int
    phase2_expert_decisions: int
    phase1_forbidden_entries: int
    direct_hazard_entries: int
    meat_collected: int
    meat_health_restored: float
    meat_health_wasted: float
    flower_contacts: int
    fatigue_health_losses: int
    direct_hazard_health_losses: int
    temperature_discomfort_total: float
    health: float
    energy: float
    terminated: bool
    truncated: bool


Phase1Selector = Callable[[BeliefMap, HeadingAwareAStar], PhasePlan | None]
Phase2Selector = Callable[[BeliefMap, HeadingAwareAStar], PhasePlan | None]


def _new_plan(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    phase: int,
    phase1_selector: Phase1Selector | None = None,
    phase2_selector: Phase2Selector | None = None,
) -> PhasePlan | None:
    if phase == 1:
        return phase1_selector(belief, router) if phase1_selector is not None else select_phase1_plan(belief, router)
    return phase2_selector(belief, router) if phase2_selector is not None else select_phase2_plan(belief, router)


def run_episode(
    seed: int,
    config: EnvConfig | None = None,
    phase1_trajectory: Phase1TrajectoryWriter | None = None,
    phase2_trajectory: Phase2TrajectoryWriter | None = None,
    phase1_selector: Phase1Selector | None = None,
    phase2_selector: Phase2Selector | None = None,
    environment_set: str = "standard",
    environment_variant: str = "standard",
) -> EpisodeResult:
    env = SensoryGridEnv(config or EnvConfig())
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
    phase = 1
    plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
    if plan is not None and phase1_trajectory is not None:
        phase1_trajectory.record(belief, plan)
    route_actions = list(plan.route.actions) if plan else []
    actions = turns = replans = phase1_replans = phase2_replans = 0
    phase1_actions = phase1_turns = phase1_expert_decisions = 0
    survival_mode_actions = 0  # retained in CSV for comparison with prior runs; now disabled.
    phase1_forbidden_entries = direct_hazard_entries = 0
    meat_collected = flower_contacts = fatigue_health_losses = direct_hazard_health_losses = 0
    meat_health_restored = meat_health_wasted = 0.0
    phase1_end_seen_fraction = phase1_end_health = phase1_end_energy = float("nan")
    phase1_temperature_discomfort_total = 0.0
    discomfort_total = 0.0

    while not env.terminated and not env.truncated:
        if not route_actions:
            plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
            if plan is None:
                break
            if phase == 1 and phase1_trajectory is not None:
                phase1_trajectory.record(belief, plan)
            elif phase == 2 and phase2_trajectory is not None:
                phase2_trajectory.record(belief, plan)
            route_actions = list(plan.route.actions)
            replans += 1
            if phase == 1:
                phase1_replans += 1
            else:
                phase2_replans += 1

        action = route_actions.pop(0)
        phase_before_action = phase
        health_before = float(env.health)
        observation, _, terminated, truncated, info = env.step(action, switches)
        newly_seen = belief.update_after_action(action, observation, info)
        actions += 1
        turns += int(action != ACTION_FORWARD)
        if phase_before_action == 1:
            phase1_actions += 1
            phase1_turns += int(action != ACTION_FORWARD)
        discomfort_total += float(info.get("discomfort", 0.0))
        if phase_before_action == 1:
            phase1_temperature_discomfort_total += float(info.get("discomfort", 0.0))
        contacted = str(info.get("contacted_label", ""))
        meat_collected += int(contacted == "Meat")
        if contacted == "Meat":
            restored = min(float(env.config.meat_heal), max(0.0, float(env.config.max_health) - health_before))
            meat_health_restored += restored
            meat_health_wasted += float(env.config.meat_heal) - restored
        flower_contacts += int(contacted == "Flower")
        health_delta = int(info.get("health_delta", 0))
        if contacted in {"Fire", "Ice", "Glass"}:
            direct_hazard_health_losses += max(0, -health_delta)
        # With no direct contact damage in the same transition, any health
        # decrease is fatigue damage caused by an exhausted energy budget.
        elif health_delta < 0:
            fatigue_health_losses += -health_delta

        if action == ACTION_FORWARD:
            row, col = belief.position
            if phase == 1:
                phase1_forbidden_entries += int(belief.phase1_forbidden[row, col])
            direct_hazard_entries += int(belief.direct_hazard[row, col])
            if phase == 1 and belief.seen_fraction >= 0.80:
                phase1_end_seen_fraction = belief.seen_fraction
                phase1_end_health = float(env.health)
                phase1_end_energy = float(env.energy)
                phase = 2
                plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
                route_actions = list(plan.route.actions) if plan else []
                if plan is not None and phase2_trajectory is not None:
                    phase2_trajectory.record(belief, plan)
                replans += 1
                phase2_replans += 1
            elif phase == 1:
                # Phase 1 deliberately replans after every physical move.
                plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
                if plan is not None and phase1_trajectory is not None:
                    phase1_trajectory.record(belief, plan)
                route_actions = list(plan.route.actions) if plan else []
                replans += 1
                phase1_replans += 1
            elif newly_seen > 0:
                # Phase 2 replans only when a forward move produced new information.
                plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
                if plan is not None and phase2_trajectory is not None:
                    phase2_trajectory.record(belief, plan)
                route_actions = list(plan.route.actions) if plan else []
                replans += 1
                phase2_replans += 1

            if phase == 2 and contacted == "Meat":
                # Food changes the route survival budget even if it was already
                # visible, so it is a permitted phase-2 survival replan.
                plan = _new_plan(belief, router, phase, phase1_selector, phase2_selector)
                if plan is not None and phase2_trajectory is not None:
                    phase2_trajectory.record(belief, plan)
                route_actions = list(plan.route.actions) if plan else []
                replans += 1
                phase2_replans += 1

        if terminated or truncated:
            break

    return EpisodeResult(
        seed=seed,
        environment_set=environment_set,
        environment_variant=environment_variant,
        seen_fraction=belief.seen_fraction,
        coverage=float(env.current_scalars()["coverage"]),
        phase=phase,
        reached_phase2=phase == 2,
        actions=actions,
        turns=turns,
        replans=replans,
        phase1_replans=phase1_replans,
        phase2_replans=phase2_replans,
        survival_mode_actions=survival_mode_actions,
        phase1_actions=phase1_actions,
        phase1_turns=phase1_turns,
        phase1_end_seen_fraction=phase1_end_seen_fraction,
        phase1_end_health=phase1_end_health,
        phase1_end_energy=phase1_end_energy,
        phase1_temperature_discomfort_total=phase1_temperature_discomfort_total,
        phase1_expert_decisions=len(phase1_trajectory.next_actions) if phase1_trajectory is not None else 0,
        phase2_expert_decisions=len(phase2_trajectory.next_actions) if phase2_trajectory is not None else 0,
        phase1_forbidden_entries=phase1_forbidden_entries,
        direct_hazard_entries=direct_hazard_entries,
        meat_collected=meat_collected,
        meat_health_restored=meat_health_restored,
        meat_health_wasted=meat_health_wasted,
        flower_contacts=flower_contacts,
        fatigue_health_losses=fatigue_health_losses,
        direct_hazard_health_losses=direct_hazard_health_losses,
        temperature_discomfort_total=discomfort_total,
        health=float(env.health),
        energy=float(env.energy),
        terminated=bool(env.terminated),
        truncated=bool(env.truncated),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the deterministic two-phase belief-map planner.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/15x15/two_phase_belief_map_planner/evaluation.csv"))
    parser.add_argument(
        "--environment-set",
        choices=environment_set_names(),
        default="standard",
        help="Stationary profile family; standard exactly preserves the original configuration.",
    )
    parser.add_argument("--max-steps", type=int, default=EnvConfig.max_steps)
    parser.add_argument(
        "--phase1-trajectory-dir",
        type=Path,
        default=Path("runs/15x15/two_phase_belief_map_planner/phase1_trajectories"),
        help="Directory for one compressed phase-1 expert trajectory per seed.",
    )
    parser.add_argument(
        "--phase2-trajectory-dir",
        type=Path,
        default=Path("runs/15x15/two_phase_belief_map_planner/phase2_trajectories"),
        help="Directory for one compressed phase-2 expert trajectory per seed.",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max-steps must be at least one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".config.json").write_text(
        json.dumps(
            {
                "environment_set": args.environment_set,
                "environment_profiles": environment_profile_configs(args.environment_set, max_steps=args.max_steps),
                "seed_start": args.seed,
                "episodes": args.episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results: List[EpisodeResult] = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EpisodeResult.__dataclass_fields__))
        writer.writeheader()
        handle.flush()
        for offset in range(args.episodes):
            seed = args.seed + offset
            environment_variant, config = environment_config_for_seed(
                seed,
                args.environment_set,
                max_steps=args.max_steps,
            )
            phase1_trajectory = Phase1TrajectoryWriter(seed)
            phase2_trajectory = Phase2TrajectoryWriter(seed)
            result = run_episode(
                seed,
                config=config,
                phase1_trajectory=phase1_trajectory,
                phase2_trajectory=phase2_trajectory,
                environment_set=args.environment_set,
                environment_variant=environment_variant,
            )
            phase1_trajectory_path = phase1_trajectory.save(args.phase1_trajectory_dir)
            phase2_trajectory_path = phase2_trajectory.save(args.phase2_trajectory_dir)
            results.append(result)
            writer.writerow(asdict(result))
            handle.flush()
            print(
                "episode={:03d}/{:03d} seed={} env={} seen={:.3f} coverage={:.3f} "
                "phase={} p1_actions={} p1_turns={} p1_decisions={} p2_decisions={} meat={} fatigue_loss={} hazards={} status={}".format(
                    offset + 1,
                    args.episodes,
                    result.seed,
                    result.environment_variant,
                    result.seen_fraction,
                    result.coverage,
                    result.phase,
                    result.phase1_actions,
                    result.phase1_turns,
                    result.phase1_expert_decisions,
                    result.phase2_expert_decisions,
                    result.meat_collected,
                    result.fatigue_health_losses,
                    result.direct_hazard_entries,
                    "terminated" if result.terminated else "truncated" if result.truncated else "stopped",
                ),
                flush=True,
            )
            print(f"  phase1_trajectory={phase1_trajectory_path}", flush=True)
            print(f"  phase2_trajectory={phase2_trajectory_path}", flush=True)
    print(
        "episodes={} seen_mean={:.3f} coverage_mean={:.3f} phase2={}/{} "
        "hazard_entries={} forbidden_entries={} meat_mean={:.2f} fatigue_loss_mean={:.2f} terminated={}".format(
            args.episodes,
            float(np.mean([result.seen_fraction for result in results])),
            float(np.mean([result.coverage for result in results])),
            sum(result.reached_phase2 for result in results),
            args.episodes,
            sum(result.direct_hazard_entries for result in results),
            sum(result.phase1_forbidden_entries for result in results),
            float(np.mean([result.meat_collected for result in results])),
            float(np.mean([result.fatigue_health_losses for result in results])),
            sum(result.terminated for result in results),
        )
    )
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

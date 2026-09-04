"""Stress-test named planner components on the same held-out maps.

The variants are deliberately subtractive: each one starts with the released
frontier planner configuration and removes one decision component.  This is an
ablation of the deterministic reference, not a hyperparameter sweep and not a
comparison with the learned policies.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from models.non_model_based.partial_observation_frontier_planner.run import (
    BASELINE_PRESET,
    PlannerLabConfig,
    build_env,
    run_episode,
)


def environment_signature(seed: int) -> str:
    env, _ = build_env(BASELINE_PRESET)
    env.reset(seed=seed)
    digest = sha256()
    digest.update(env.reveal_world_ids().tobytes())
    digest.update(env.reveal_temperature_field_c().tobytes())
    digest.update(str(int(env.direction)).encode("ascii"))
    return digest.hexdigest()


def mean_and_p10(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(rows)
    result = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    result["coverage_p10"] = float(np.quantile([row["coverage"] for row in records], 0.10))
    return result


def two_sided_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(comb(trials, value) for value in range(min(wins, losses) + 1)) / (2**trials)
    return float(min(1.0, 2.0 * tail))


def paired_summary(deltas: np.ndarray, bootstrap_seed: int, resamples: int) -> Dict[str, float | int]:
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.mean(rng.choice(deltas, size=(resamples, len(deltas)), replace=True), axis=1)
    wins = int(np.sum(deltas > 0.0))
    losses = int(np.sum(deltas < 0.0))
    ties = int(np.sum(deltas == 0.0))
    return {
        "mean_coverage_delta_variant_minus_full": float(np.mean(deltas)),
        "median_coverage_delta_variant_minus_full": float(np.median(deltas)),
        "bootstrap_95_ci_low": float(np.quantile(samples, 0.025)),
        "bootstrap_95_ci_high": float(np.quantile(samples, 0.975)),
        "variant_wins": wins,
        "full_planner_wins": losses,
        "ties": ties,
        "two_sided_sign_test_p": two_sided_sign_test(wins, losses),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out paired ablation of frontier-planner components")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=65_000, help="First generated map seed.")
    parser.add_argument(
        "--save-dir",
        default="runs/15x15/benchmarks/frontier_planner_component_ablation_heldout",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=25_218_029)
    args = parser.parse_args()
    episodes, resamples = max(1, args.episodes), max(1, args.bootstrap_resamples)
    full = PlannerLabConfig(episodes=episodes, seed=args.seed - 1, environment_preset=BASELINE_PRESET)
    variants = {
        "full_frontier_planner": full,
        "without_route_revisit_costs": replace(full, revisit_cost=0.0, repeat_visit_cost=0.0),
        "without_thermal_route_cost": replace(full, thermal_extra_energy_max=0.0),
        "without_resource_recovery": replace(full, reserve_health_norm=0.0),
        "without_safe_frontier_forward": replace(full, force_safe_frontier_forward=False),
    }
    metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in variants}
    rows: List[Dict[str, object]] = []
    for index in range(episodes):
        episode, seed = index + 1, args.seed + index
        outcomes = {name: run_episode(config, episode, seed) for name, config in variants.items()}
        for name, outcome in outcomes.items():
            metrics[name].append(outcome)
        full_coverage = outcomes["full_frontier_planner"]["coverage"]
        row: Dict[str, object] = {
            "episode": episode,
            "seed": seed,
            "environment_signature": environment_signature(seed),
        }
        line = [f"episode={episode:03d}", f"seed={seed}"]
        for name, outcome in outcomes.items():
            coverage = outcome["coverage"]
            row[f"{name}_coverage"] = coverage
            if name != "full_frontier_planner":
                row[f"{name}_delta_minus_full"] = coverage - full_coverage
            line.append(f"{name}={coverage:.4f}")
        rows.append(row)
        print(" ".join(line), flush=True)

    comparisons = {
        name: paired_summary(
            np.asarray([float(row[f"{name}_delta_minus_full"]) for row in rows]),
            args.bootstrap_seed + index,
            resamples,
        )
        for index, name in enumerate(variants)
        if name != "full_frontier_planner"
    }
    summary = {
        "config": {
            "episodes": episodes,
            "seed": args.seed,
            "save_dir": args.save_dir,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "environment_preset": BASELINE_PRESET,
        },
        "scope": {
            "evaluation_type": "paired held-out map component ablation of the deterministic frontier planner",
            "same_seed_same_environment": True,
            "interpretation": (
                "Each variant removes one named component from the released planner. "
                "This isolates component stress tests within this implementation; it does not "
                "establish a globally optimal planner parameterisation."
            ),
        },
        "variant_configs": {name: asdict(config) for name, config in variants.items()},
        "controller_metrics": {name: mean_and_p10(values) for name, values in metrics.items()},
        "paired_comparisons": comparisons,
    }
    output = Path(args.save_dir)
    write_csv(output / "comparison_by_episode.csv", rows)
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

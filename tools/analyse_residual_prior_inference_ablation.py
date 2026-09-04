#!/usr/bin/env python3
"""Analyse the matched inference-time persistent-prior diagnostic.

This compares two evaluations of the same saved residual-PPO checkpoint on
identical maps: the released controller and a diagnostic version in which the
persistent planner target scores and tie bonus are zeroed at inference. The
checkpoint was trained with the prior, so this is a dependency check rather
than a no-prior training ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL_DEFAULT = ROOT / "runs/15x15/benchmarks/frontier_planner_vs_residual_prior_full_heldout/comparison_by_episode.csv"
WITHOUT_DEFAULT = ROOT / "runs/15x15/benchmarks/frontier_planner_vs_residual_without_persistent_prior_heldout/comparison_by_episode.csv"
OUTPUT_DEFAULT = ROOT / "runs/15x15/benchmarks/residual_persistent_prior_inference_ablation_heldout"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    lower_tail = sum(comb(trials, index) for index in range(min(wins, losses) + 1))
    return float(min(1.0, 2.0 * lower_tail / (2**trials)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse residual persistent-prior inference ablation")
    parser.add_argument("--full", type=Path, default=FULL_DEFAULT)
    parser.add_argument("--without-prior", type=Path, default=WITHOUT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=25_218_029)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_rows, without_rows = read_rows(args.full), read_rows(args.without_prior)
    if len(full_rows) != len(without_rows) or not full_rows:
        raise ValueError("Expected non-empty, equally sized matched CSV files.")

    paired: list[dict[str, object]] = []
    for full, without in zip(full_rows, without_rows):
        matching = ("episode", "seed", "environment_signature", "planner_coverage")
        if any(full[field] != without[field] for field in matching):
            raise ValueError("The two evaluations are not aligned on identical map episodes.")
        full_coverage = float(full["residual_ppo_seed_7_coverage"])
        without_coverage = float(without["residual_ppo_seed_7_without_persistent_prior_coverage"])
        paired.append(
            {
                "episode": int(full["episode"]),
                "seed": int(full["seed"]),
                "environment_signature": full["environment_signature"],
                "planner_coverage": float(full["planner_coverage"]),
                "full_prior_coverage": full_coverage,
                "without_prior_coverage": without_coverage,
                "coverage_delta_full_minus_without_prior": full_coverage - without_coverage,
            }
        )

    deltas = np.asarray([float(row["coverage_delta_full_minus_without_prior"]) for row in paired])
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = np.mean(
        rng.choice(deltas, size=(max(1, args.bootstrap_resamples), len(deltas)), replace=True),
        axis=1,
    )
    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    summary = {
        "scope": {
            "evaluation_type": "matched inference-time persistent-planner-prior dependency diagnostic",
            "same_checkpoint": True,
            "same_map_seed_and_fingerprint": True,
            "full_prior": "Released evaluation weights: planner-score weight 1.10 and teacher tie bonus 0.25.",
            "without_prior": "Evaluation-only ablation: planner-score weight and teacher tie bonus both zero.",
            "interpretation": "The checkpoint was trained with the persistent planner prior. This estimates its inference-time dependence on that prior, not the effect of training an architecture without one.",
        },
        "inputs": {"full_prior_csv": str(args.full), "without_prior_csv": str(args.without_prior)},
        "episodes": len(paired),
        "full_prior_mean_coverage": float(np.mean([row["full_prior_coverage"] for row in paired])),
        "full_prior_coverage_p10": float(np.quantile([row["full_prior_coverage"] for row in paired], 0.10)),
        "without_prior_mean_coverage": float(np.mean([row["without_prior_coverage"] for row in paired])),
        "without_prior_coverage_p10": float(np.quantile([row["without_prior_coverage"] for row in paired], 0.10)),
        "mean_coverage_delta_full_minus_without_prior": float(np.mean(deltas)),
        "median_coverage_delta_full_minus_without_prior": float(np.median(deltas)),
        "bootstrap_resamples": max(1, args.bootstrap_resamples),
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_95_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_ci_high": float(np.quantile(bootstrap, 0.975)),
        "full_prior_wins": wins,
        "without_prior_wins": losses,
        "ties": ties,
        "two_sided_sign_test_p": sign_test(wins, losses),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison_by_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

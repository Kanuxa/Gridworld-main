"""Reproduce the paired planner--oracle statistical summary in the report."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean


DEFAULT_INPUT = Path(
    "results/artifacts/15x15/benchmarks/frontier_planner_vs_oracle/"
    "comparison_by_episode.csv"
)


def percentile_from_sorted(values: list[float], fraction: float) -> float:
    """Return the report's fixed-order bootstrap percentile."""
    if not values:
        raise ValueError("Cannot take a percentile of an empty sequence.")
    index = int(fraction * len(values))
    return values[min(index, len(values) - 1)]


def two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign-test p-value, ignoring ties."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    smaller_tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(trials, count) for count in range(smaller_tail + 1)) / (2**trials)
    return min(1.0, probability)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise paired coverage differences for the planner--oracle benchmark."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=25_918_029)
    args = parser.parse_args()

    if args.resamples < 1:
        raise ValueError("--resamples must be positive.")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    deltas = [float(row["coverage_delta_god_minus_planner"]) for row in rows]
    if not deltas:
        raise ValueError(f"No paired rows found in {args.input}.")

    generator = random.Random(args.seed)
    bootstrap_means = sorted(
        mean(generator.choices(deltas, k=len(deltas))) for _ in range(args.resamples)
    )
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - wins - losses

    print(f"paired episodes: {len(deltas)}")
    print(f"mean coverage delta (oracle - planner): {mean(deltas):.6f}")
    print(
        "bootstrap 95% CI: "
        f"[{percentile_from_sorted(bootstrap_means, 0.025):.6f}, "
        f"{percentile_from_sorted(bootstrap_means, 0.975):.6f}]"
    )
    print(f"oracle wins / planner wins / ties: {wins} / {losses} / {ties}")
    print(f"exact two-sided sign-test p-value: {two_sided_sign_test(wins, losses):.3e}")


if __name__ == "__main__":
    main()

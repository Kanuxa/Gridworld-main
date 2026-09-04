#!/usr/bin/env python3
"""Create the dissertation's reproducible Matplotlib result figures.

The script only reads the curated archive under ``results/artifacts`` and the
normalised aggregate table ``results/summary_metrics.csv``.  Each PNG is
written both to ``results/figures`` (the archived result) and ``report/images``
(the copy embedded by LaTeX).  The two locations deliberately contain the same
version of each graphic.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURE_DIRS = (RESULTS / "figures", ROOT / "report" / "images")

PLANNER_BLUE = "#2B5F97"
LEARNED_ORANGE = "#D6792A"
ORACLE_GREY = "#777777"
GRID_GREY = "#D9DEE7"
TEXT_GREY = "#374151"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or not value.strip() or value.strip().lower() == "nan":
        return None
    return float(value)


def required_number(row: dict[str, str], field: str) -> float:
    value = number(row.get(field))
    if value is None or not math.isfinite(value):
        raise ValueError(f"Missing finite {field!r} in {row}")
    return value


def summary_row(
    rows: Iterable[dict[str, str]], source: str, metric_set: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["source_summary"] == source and row["metric_set"] == metric_set
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary row for {source!r} / {metric_set!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "semibold",
            "axes.labelcolor": TEXT_GREY,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.7,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / filename, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def add_value_labels(ax: plt.Axes, bars: Iterable, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT_GREY,
        )


def heldout_learned_summary() -> dict:
    path = RESULTS / "artifacts/15x15/benchmarks/frontier_planner_vs_learned_heldout/comparison_summary.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def benchmark_coverage(rows: list[dict[str, str]]) -> None:
    heldout = heldout_learned_summary()["controller_metrics"]
    data_15 = [
        (
            "Frontier\nplanner",
            100 * heldout["frontier_planner"]["coverage"],
            100 * heldout["frontier_planner"]["coverage_p10"],
            False,
        ),
        (
            "Spatial-target\nPPO",
            100 * heldout["spatial_target_ppo"]["coverage"],
            100 * heldout["spatial_target_ppo"]["coverage_p10"],
            False,
        ),
        (
            "Residual\nPPO",
            100 * heldout["planner_residual_ppo"]["coverage"],
            100 * heldout["planner_residual_ppo"]["coverage_p10"],
            False,
        ),
    ]
    planner_31 = summary_row(
        rows,
        "31x31/non_model_based/partial_observation_frontier_planner/baseline_50/summary.json",
        "aggregate",
    )
    residual_31 = summary_row(
        rows,
        "31x31/model_based/planner_residual_target_ppo/summary.json",
        "best_evaluation",
    )
    data_31 = [
        (
            "Frontier\nplanner",
            100 * required_number(planner_31, "coverage"),
            100 * required_number(planner_31, "coverage_p10"),
            False,
        ),
        (
            "Residual\nPPO",
            100 * required_number(residual_31, "coverage"),
            100 * required_number(residual_31, "coverage_p10"),
            False,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.45), sharey=True)
    for ax, title, data in zip(axes, ("15×15 held-out paired test", "31×31 archived summaries"), (data_15, data_31)):
        labels = [item[0] for item in data]
        mean = [item[1] for item in data]
        p10 = [item[2] for item in data]
        positions = list(range(len(data)))
        width = 0.34
        mean_bars = ax.bar(
            [position - width / 2 for position in positions],
            mean,
            width,
            color=PLANNER_BLUE,
            label="Mean coverage",
            zorder=3,
        )
        p10_bars = ax.bar(
            [position + width / 2 for position in positions],
            p10,
            width,
            color=LEARNED_ORANGE,
            label="P10 coverage",
            zorder=3,
        )
        for index, (_, _, _, privileged) in enumerate(data):
            if privileged:
                ax.axvspan(index - 0.52, index + 0.52, color=ORACLE_GREY, alpha=0.06, zorder=0)
                for bar in (mean_bars[index], p10_bars[index]):
                    bar.set_hatch("///")
                    bar.set_edgecolor(ORACLE_GREY)
                    bar.set_linewidth(0.8)

        add_value_labels(ax, mean_bars, mean)
        add_value_labels(ax, p10_bars, p10)
        ax.set_title(title, pad=10)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4)

    axes[0].set_ylabel("Coverage (%)")
    handles = [
        Patch(facecolor=PLANNER_BLUE, label="Mean coverage"),
        Patch(facecolor=LEARNED_ORANGE, label="P10 coverage"),
    ]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.04), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.24, wspace=0.18)
    save_figure(fig, "final_benchmark_coverage.png")


def paired_oracle_map_comparison() -> None:
    rows = read_csv(
        RESULTS
        / "artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_by_episode.csv"
    )
    planner = [100 * required_number(row, "planner_coverage") for row in rows]
    oracle = [100 * required_number(row, "god_coverage") for row in rows]
    deltas = [oracle_value - planner_value for planner_value, oracle_value in zip(planner, oracle)]
    lower = math.floor(min(planner + oracle) / 5) * 5 - 1
    upper = math.ceil(max(planner + oracle) / 5) * 5 + 1
    wins = sum(delta > 0 for delta in deltas)

    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    ax.scatter(planner, oracle, s=31, color=ORACLE_GREY, alpha=0.85, edgecolor="white", linewidth=0.45, zorder=3)
    ax.plot([lower, upper], [lower, upper], color=TEXT_GREY, linestyle="--", linewidth=1.0, label="Equal coverage", zorder=2)
    ax.fill_between([lower, upper], [lower, upper], upper, color=PLANNER_BLUE, alpha=0.055, zorder=0)
    ax.text(
        0.025,
        0.965,
        f"Oracle wins {wins}/50 matched maps\nMean advantage: {sum(deltas) / len(deltas):.2f} pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=TEXT_GREY,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": GRID_GREY},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Frontier-planner coverage (%)")
    ax.set_ylabel("Full-information-oracle coverage (%)")
    ax.grid(color=GRID_GREY, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "paired_oracle_map_comparison.png")


def evaluation_points(path: Path) -> tuple[list[int], list[float], list[float]]:
    episodes: list[int] = []
    means: list[float] = []
    p10s: list[float] = []
    for row in read_csv(path):
        mean = number(row.get("eval_coverage"))
        p10 = number(row.get("eval_coverage_p10"))
        if mean is not None and p10 is not None and math.isfinite(mean) and math.isfinite(p10):
            episodes.append(int(required_number(row, "episode")))
            means.append(100 * mean)
            p10s.append(100 * p10)
    if len(episodes) < 2:
        raise ValueError(f"Expected at least two evaluation points in {path}")
    return episodes, means, p10s


def ppo_checkpoint_trajectories(rows: list[dict[str, str]]) -> None:
    planner = heldout_learned_summary()["controller_metrics"]["frontier_planner"]
    planner_mean = 100 * float(planner["coverage"])
    planner_p10 = 100 * float(planner["coverage_p10"])
    runs = (
        (
            "Spatial-target PPO",
            RESULTS / "artifacts/15x15/model_based/spatial_target_ppo_with_planner/episode_metrics.csv",
        ),
        (
            "Planner-residual PPO",
            RESULTS / "artifacts/15x15/model_based/planner_residual_target_ppo/episode_metrics.csv",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.25), sharey=True)
    for ax, (title, path) in zip(axes, runs):
        episodes, means, p10s = evaluation_points(path)
        ax.plot(episodes, means, marker="o", markersize=3.0, linewidth=1.6, color=PLANNER_BLUE, label="Evaluation mean")
        ax.plot(episodes, p10s, marker="o", markersize=3.0, linewidth=1.6, color=LEARNED_ORANGE, label="Evaluation P10")
        ax.axhline(planner_mean, color=PLANNER_BLUE, linewidth=1.1, linestyle="--", alpha=0.85, label="Planner mean")
        ax.axhline(planner_p10, color=LEARNED_ORANGE, linewidth=1.1, linestyle="--", alpha=0.85, label="Planner P10")
        ax.set_title(title, pad=9)
        ax.set_xlim(0, 3000)
        ax.set_xticks((0, 1000, 2000, 3000))
        ax.set_ylim(35, 82)
        ax.grid(color=GRID_GREY, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Training episode")
    axes[0].set_ylabel("Coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.045), frameon=False, fontsize=7.7)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.25, wspace=0.16)
    save_figure(fig, "ppo_checkpoint_trajectories.png")


def paired_oracle_delta_distribution() -> None:
    rows = read_csv(
        RESULTS
        / "artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_by_episode.csv"
    )
    deltas = [100 * required_number(row, "coverage_delta_god_minus_planner") for row in rows]
    mean_delta = sum(deltas) / len(deltas)

    fig, ax = plt.subplots(figsize=(6.7, 3.15))
    bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ax.hist(deltas, bins=bins, color=ORACLE_GREY, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvspan(6.02, 7.96, color=PLANNER_BLUE, alpha=0.12, label="95% bootstrap CI for mean")
    ax.axvline(0, color=TEXT_GREY, linewidth=1.0, linestyle="--", label="No difference")
    ax.axvline(mean_delta, color=PLANNER_BLUE, linewidth=1.5, label=f"Mean: {mean_delta:.2f} pp")
    ax.set_xlabel("Oracle minus frontier-planner coverage (percentage points)")
    ax.set_ylabel("Matched maps")
    ax.set_xlim(2, 13)
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "paired_oracle_delta_distribution.png")


def paired_learned_delta_distribution() -> None:
    """Plot the held-out paired effects for the two saved PPO checkpoints."""
    rows = read_csv(
        RESULTS
        / "artifacts/15x15/benchmarks/frontier_planner_vs_learned_heldout/comparison_by_episode.csv"
    )
    summary = heldout_learned_summary()["paired_comparisons"]
    series = (
        ("Spatial-target PPO", "spatial_target_ppo_delta_minus_planner", "spatial_target_ppo_vs_planner", "#D6792A"),
        ("Planner-residual PPO", "planner_residual_ppo_delta_minus_planner", "planner_residual_ppo_vs_planner", "#B45309"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 4.15), sharex=True)
    for ax, (title, field, comparison_key, colour) in zip(axes, series):
        deltas = np.asarray([100 * required_number(row, field) for row in rows])
        comparison = summary[comparison_key]
        ci_low, ci_high = 100 * comparison["bootstrap_95_ci_low"], 100 * comparison["bootstrap_95_ci_high"]
        lower, upper = math.floor(min(deltas) - 1), math.ceil(max(deltas) + 1)
        bins = np.arange(lower, upper + 1.01, 1.0)
        ax.hist(deltas, bins=bins, color=colour, edgecolor="white", linewidth=0.75, zorder=3)
        ax.axvspan(ci_low, ci_high, color=PLANNER_BLUE, alpha=0.13, label="95% bootstrap CI for mean")
        ax.axvline(0, color=TEXT_GREY, linewidth=1.0, linestyle="--", label="No difference")
        ax.axvline(100 * comparison["mean_coverage_delta_learned_minus_planner"], color=PLANNER_BLUE, linewidth=1.4, label="Mean difference")
        ax.set_title(title, loc="left", fontsize=9, pad=6)
        ax.text(
            0.985, 0.92,
            f"mean {100 * comparison['mean_coverage_delta_learned_minus_planner']:.2f} pp; p={comparison['two_sided_sign_test_p']:.2g}\n"
            f"wins: learned {comparison['learned_wins']}, planner {comparison['planner_wins']}, ties {comparison['ties']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color=TEXT_GREY,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID_GREY},
        )
        ax.set_ylabel("Matched maps")
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Learned-controller minus frontier-planner coverage (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.92, bottom=0.16, hspace=0.42)
    save_figure(fig, "paired_learned_delta_distribution.png")


def planner_component_ablation() -> None:
    """Plot paired effects of subtracting one planner component at a time."""
    path = (
        RESULTS
        / "artifacts/15x15/benchmarks/frontier_planner_component_ablation_heldout"
        / "comparison_summary.json"
    )
    with path.open(encoding="utf-8") as handle:
        comparisons = json.load(handle)["paired_comparisons"]
    series = (
        ("No resource recovery", "without_resource_recovery", LEARNED_ORANGE),
        ("No safe-frontier forward rule", "without_safe_frontier_forward", LEARNED_ORANGE),
        ("No route revisit costs", "without_route_revisit_costs", LEARNED_ORANGE),
        ("No thermal route cost", "without_thermal_route_cost", ORACLE_GREY),
    )
    labels = [item[0] for item in series]
    means = np.asarray([
        100 * comparisons[key]["mean_coverage_delta_variant_minus_full"]
        for _, key, _ in series
    ])
    lower = np.asarray([
        100 * comparisons[key]["bootstrap_95_ci_low"]
        for _, key, _ in series
    ])
    upper = np.asarray([
        100 * comparisons[key]["bootstrap_95_ci_high"]
        for _, key, _ in series
    ])
    errors = np.vstack((means - lower, upper - means))

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    positions = np.arange(len(series))
    for index, (_, key, colour) in enumerate(series):
        ax.errorbar(
            means[index], positions[index], xerr=errors[:, index:index + 1],
            fmt="o", color=colour, ecolor=colour, capsize=3.2, markersize=5.8,
            linewidth=1.45, zorder=3,
        )
        ax.text(
            0.985, positions[index],
            f"p={comparisons[key]['two_sided_sign_test_p']:.3g}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.4, color=TEXT_GREY,
        )
    ax.axvline(0, color=TEXT_GREY, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(-14.5, 1.5)
    ax.set_xticks((-14, -10, -6, -2, 0))
    ax.set_xlabel("Coverage difference: component removed minus full planner (pp)")
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, "frontier_planner_component_ablation.png")


def residual_robustness_stress_tests() -> None:
    """Plot pre-specified matched shifts for the fixed residual checkpoint.

    The point estimate is deliberately presented with its map-resampling
    interval and sign-test value: these two tests assess sensitivity of a
    saved controller to a changed task condition, not training replication or
    real-robot robustness.
    """
    series = (
        (
            "15% object-detection dropout",
            "frontier_planner_vs_residual_object_dropout_heldout",
        ),
        (
            "One restorative resource",
            "frontier_planner_vs_residual_resource_scarce_heldout",
        ),
    )
    values = []
    for title, directory in series:
        path = RESULTS / "artifacts/15x15/benchmarks" / directory / "comparison_summary.json"
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        comparison = summary["paired_comparisons"]["residual_ppo_seed_7"]
        values.append((title, comparison))

    means = np.asarray([
        100 * item[1]["mean_coverage_delta_residual_minus_planner"] for item in values
    ])
    lower = np.asarray([
        100 * item[1]["bootstrap_95_ci_low"] for item in values
    ])
    upper = np.asarray([
        100 * item[1]["bootstrap_95_ci_high"] for item in values
    ])
    errors = np.vstack((means - lower, upper - means))
    positions = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(6.8, 1.85))
    ax.errorbar(
        means, positions, xerr=errors, fmt="o", color=LEARNED_ORANGE,
        ecolor=LEARNED_ORANGE, capsize=3.2, markersize=6.0, linewidth=1.5,
        zorder=3,
    )
    ax.axvline(0, color=TEXT_GREY, linewidth=1.0, linestyle="--", zorder=1)
    for index, (_, comparison) in enumerate(values):
        ax.text(
            0.985, positions[index],
            f"p={comparison['two_sided_sign_test_p']:.3g}; "
            f"wins {comparison['residual_wins']}--{comparison['planner_wins']}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.2, color=TEXT_GREY,
        )
    ax.set_yticks(positions, [item[0] for item in values])
    ax.invert_yaxis()
    ax.set_xlim(-3.1, 6.1)
    ax.set_xticks((-3, -1, 0, 1, 3, 5))
    ax.set_xlabel("Residual PPO minus frontier-planner coverage (percentage points)")
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, "residual_robustness_stress_tests.png")


def residual_prior_inference_ablation() -> None:
    """Plot the evaluation-only persistent-prior dependency diagnostic."""
    path = (
        RESULTS
        / "artifacts/15x15/benchmarks/residual_persistent_prior_inference_ablation_heldout"
        / "summary.json"
    )
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    mean = 100 * summary["mean_coverage_delta_full_minus_without_prior"]
    lower = 100 * summary["bootstrap_95_ci_low"]
    upper = 100 * summary["bootstrap_95_ci_high"]

    fig, ax = plt.subplots(figsize=(6.8, 1.55))
    ax.errorbar(
        [mean], [0], xerr=[[mean - lower], [upper - mean]], fmt="o",
        color=LEARNED_ORANGE, ecolor=LEARNED_ORANGE, capsize=3.3,
        markersize=6.0, linewidth=1.5, zorder=3,
    )
    ax.axvline(0, color=TEXT_GREY, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks([0], ["Persistent planner prior retained"])
    ax.set_xlim(-1.2, 3.2)
    ax.set_xticks((-1, 0, 1, 2, 3))
    ax.set_xlabel("Full prior minus evaluation-only no-prior coverage (percentage points)")
    ax.grid(axis="x", color=GRID_GREY, linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save_figure(fig, "residual_prior_inference_ablation.png")


def main() -> None:
    configure_style()
    summary_rows = read_csv(RESULTS / "summary_metrics.csv")
    benchmark_coverage(summary_rows)
    paired_oracle_map_comparison()
    ppo_checkpoint_trajectories(summary_rows)
    paired_oracle_delta_distribution()
    paired_learned_delta_distribution()
    planner_component_ablation()
    residual_robustness_stress_tests()
    residual_prior_inference_ablation()
    print("Wrote eight reproducible report figures to results/figures and report/images.")


if __name__ == "__main__":
    main()

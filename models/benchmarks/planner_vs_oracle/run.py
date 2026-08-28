"""Run the plain planner and full-map God Mode on identical seeded maps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from models.benchmarks.full_information_oracle.run import GodModeRunConfig, build_env as build_god_env, run_episode as run_god_episode
from models.non_model_based.partial_observation_frontier_planner.run import PlannerLabConfig, build_env as build_planner_env, run_episode as run_planner_episode


@dataclass(frozen=True)
class ComparisonConfig:
    episodes: int = 50
    seed: int = 50_000
    save_dir: str = "runs/runs_15x15/runs_planner_vs_god"
    trace_every: int = 10
    god_search_beam_width: int = 8192


def environment_signature(seed: int) -> str:
    """Fingerprint the deterministic world created for a seed.

    Both competitors construct a fresh environment from this same seed. The
    signature is saved alongside each comparison row as an auditable check
    that the placement and temperature field are identical.
    """
    env, _ = build_planner_env()
    env.reset(seed=seed)
    digest = hashlib.sha256()
    digest.update(env.reveal_world_ids().tobytes())
    digest.update(env.reveal_temperature_field_c().tobytes())
    digest.update(str(int(env.direction)).encode("ascii"))
    return digest.hexdigest()


def metric_summary(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(rows)
    result = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    result["coverage_p10"] = float(np.quantile([row["coverage"] for row in records], 0.10))
    return result


def add_derived_action_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    result = dict(metrics)
    result.setdefault("turn_actions", float(result["steps"] - result["forward_actions"]))
    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(cfg: ComparisonConfig) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    output = Path(cfg.save_dir)
    planner_dir = output / "planner"
    god_dir = output / "god_mode"
    planner_cfg = PlannerLabConfig(
        episodes=cfg.episodes,
        seed=cfg.seed,
        save_dir=str(planner_dir),
        trace_every=cfg.trace_every,
    )
    god_cfg = GodModeRunConfig(
        episodes=cfg.episodes,
        seed=cfg.seed,
        save_dir=str(god_dir),
        trace_every=cfg.trace_every,
        search_beam_width=cfg.god_search_beam_width,
    )

    planner_rows: List[Dict[str, float]] = []
    god_rows: List[Dict[str, float]] = []
    comparison_rows: List[Dict[str, object]] = []
    for index in range(cfg.episodes):
        episode, seed = index + 1, cfg.seed + index + 1
        trace_planner = planner_dir / "traces" if episode % cfg.trace_every == 0 else None
        trace_god = god_dir / "traces" if episode % cfg.trace_every == 0 else None
        planner_metrics = add_derived_action_metrics(run_planner_episode(planner_cfg, episode, seed, trace_planner))
        god_metrics = add_derived_action_metrics(run_god_episode(god_cfg, episode, seed, trace_god))
        planner_rows.append(planner_metrics)
        god_rows.append(god_metrics)
        coverage_delta = float(god_metrics["coverage"] - planner_metrics["coverage"])
        comparison_rows.append({
            "episode": episode,
            "seed": seed,
            "environment_signature": environment_signature(seed),
            "planner_coverage": planner_metrics["coverage"],
            "god_coverage": god_metrics["coverage"],
            "coverage_delta_god_minus_planner": coverage_delta,
            "winner": "god_mode" if coverage_delta > 0 else ("planner" if coverage_delta < 0 else "tie"),
            "planner_steps": planner_metrics["steps"],
            "god_steps": god_metrics["steps"],
            "planner_forward_actions": planner_metrics["forward_actions"],
            "god_forward_actions": god_metrics["forward_actions"],
            "planner_turn_actions": planner_metrics["turn_actions"],
            "god_turn_actions": god_metrics["turn_actions"],
            "planner_repeat_forwards": planner_metrics["repeat_forwards"],
            "god_repeat_forwards": god_metrics["repeat_forwards"],
            "planner_health_loss": planner_metrics["health_loss"],
            "god_health_loss": god_metrics["health_loss"],
        })

    planner_summary = metric_summary(planner_rows)
    god_summary = metric_summary(god_rows)
    deltas = np.asarray([float(row["coverage_delta_god_minus_planner"]) for row in comparison_rows], dtype=np.float64)
    comparison_summary = {
        "episodes": cfg.episodes,
        "same_seed_same_environment": True,
        "planner": planner_summary,
        "god_mode": god_summary,
        "mean_coverage_delta_god_minus_planner": float(np.mean(deltas)),
        "median_coverage_delta_god_minus_planner": float(np.median(deltas)),
        "coverage_delta_p10_god_minus_planner": float(np.quantile(deltas, 0.10)),
        "god_mode_wins": int(sum(row["winner"] == "god_mode" for row in comparison_rows)),
        "planner_wins": int(sum(row["winner"] == "planner" for row in comparison_rows)),
        "ties": int(sum(row["winner"] == "tie" for row in comparison_rows)),
        "god_mode_win_rate": float(sum(row["winner"] == "god_mode" for row in comparison_rows) / len(comparison_rows)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (planner_dir / "planner_config.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in asdict(planner_cfg).items()) + "\n", encoding="utf-8"
    )
    (god_dir / "god_mode_config.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in asdict(god_cfg).items()) + "\n", encoding="utf-8"
    )
    write_csv(planner_dir / "planner_episode_metrics.csv", [
        {"episode": index + 1, "seed": cfg.seed + index + 1, **row} for index, row in enumerate(planner_rows)
    ])
    write_csv(god_dir / "god_mode_episode_metrics.csv", [
        {"episode": index + 1, "seed": cfg.seed + index + 1, **row} for index, row in enumerate(god_rows)
    ])
    write_csv(output / "comparison_by_episode.csv", comparison_rows)
    (planner_dir / "summary.json").write_text(json.dumps({"config": asdict(planner_cfg), "metrics": planner_summary}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (god_dir / "summary.json").write_text(json.dumps({"config": asdict(god_cfg), "metrics": god_summary}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "comparison_summary.json").write_text(json.dumps({"config": asdict(cfg), "comparison": comparison_summary}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return comparison_summary, comparison_rows


def main() -> None:
    defaults = ComparisonConfig()
    parser = argparse.ArgumentParser(description="Matched-map benchmark: partial-observation planner versus full-map God Mode")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="First comparison uses seed + 1")
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--trace-every", type=int, default=defaults.trace_every)
    parser.add_argument("--god-search-beam-width", type=int, default=defaults.god_search_beam_width)
    args = parser.parse_args()
    cfg = ComparisonConfig(
        episodes=max(1, args.episodes),
        seed=args.seed,
        save_dir=args.save_dir,
        trace_every=max(1, args.trace_every),
        god_search_beam_width=max(1, args.god_search_beam_width),
    )
    summary, _ = run_comparison(cfg)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

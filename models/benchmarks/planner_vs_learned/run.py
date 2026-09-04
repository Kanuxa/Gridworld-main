"""Evaluate saved learned checkpoints against the frontier planner on matched maps.

This benchmark intentionally uses a held-out seed range.  It estimates the
episode-level difference conditional on the saved training runs; it does not
replace replication across independent training seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from models.model_based.ppo.planner_residual_target_ppo.model import PlannerResidualTargetActorCritic
from models.model_based.ppo.planner_residual_target_ppo.train import (
    PlannerResidualTrainConfig,
    build_env as build_residual_env,
    run_episode as run_residual_episode,
)
from models.model_based.ppo.spatial_target_ppo_with_planner.model import TargetMapActorCritic
from models.model_based.ppo.spatial_target_ppo_with_planner.train import (
    TargetTrainConfig,
    build_env as build_spatial_env,
    run_episode as run_spatial_episode,
)
from models.non_model_based.partial_observation_frontier_planner.run import (
    BASELINE_PRESET,
    PlannerLabConfig,
    build_env as build_planner_env,
    run_episode as run_planner_episode,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESIDUAL_CHECKPOINT = REPOSITORY_ROOT / "runs/15x15/model_based/planner_residual_target_ppo/best_coverage.pt"
DEFAULT_SPATIAL_CHECKPOINT = REPOSITORY_ROOT / "runs/15x15/model_based/spatial_target_ppo_with_planner/best_coverage.pt"


@dataclass(frozen=True)
class PairedBenchmarkConfig:
    episodes: int = 50
    seed: int = 60_000
    save_dir: str = "runs/15x15/benchmarks/frontier_planner_vs_learned_heldout"
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 25_218_029
    residual_checkpoint: str = str(DEFAULT_RESIDUAL_CHECKPOINT)
    spatial_checkpoint: str = str(DEFAULT_SPATIAL_CHECKPOINT)


def environment_signature(seed: int) -> str:
    """Return a hash of the hidden world and temperature field for one seed."""
    env, _ = build_planner_env(BASELINE_PRESET)
    env.reset(seed=seed)
    digest = sha256()
    digest.update(env.reveal_world_ids().tobytes())
    digest.update(env.reveal_temperature_field_c().tobytes())
    digest.update(str(int(env.direction)).encode("ascii"))
    return digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_from_payload(config_type: type, payload: Dict[str, Any]):
    allowed = {field.name for field in fields(config_type)}
    values = payload.get("train_config", {})
    return config_type(**{key: value for key, value in values.items() if key in allowed})


def load_residual(path: Path) -> Tuple[PlannerResidualTargetActorCritic, PlannerResidualTrainConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_payload(PlannerResidualTrainConfig, payload)
    model = PlannerResidualTargetActorCritic()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload


def load_spatial(path: Path) -> Tuple[TargetMapActorCritic, TargetTrainConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_payload(TargetTrainConfig, payload)
    model = TargetMapActorCritic()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload


def mean_and_p10(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(rows)
    result = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    result["coverage_p10"] = float(np.quantile([row["coverage"] for row in records], 0.10))
    return result


def two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact binomial sign test over non-tied paired outcomes."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(comb(trials, index) for index in range(min(wins, losses) + 1)) / (2**trials)
    return float(min(1.0, 2.0 * tail))


def paired_summary(deltas: np.ndarray, bootstrap_seed: int, resamples: int) -> Dict[str, float | int]:
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.mean(rng.choice(deltas, size=(resamples, len(deltas)), replace=True), axis=1)
    wins = int(np.sum(deltas > 0.0))
    losses = int(np.sum(deltas < 0.0))
    ties = int(np.sum(deltas == 0.0))
    return {
        "mean_coverage_delta_learned_minus_planner": float(np.mean(deltas)),
        "median_coverage_delta_learned_minus_planner": float(np.median(deltas)),
        "bootstrap_95_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_ci_high": float(np.quantile(bootstrap, 0.975)),
        "learned_wins": wins,
        "planner_wins": losses,
        "ties": ties,
        "two_sided_sign_test_p": two_sided_sign_test(wins, losses),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    defaults = PairedBenchmarkConfig()
    parser = argparse.ArgumentParser(description="Held-out matched-map planner-versus-learned benchmark")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="First episode uses this seed.")
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--bootstrap-resamples", type=int, default=defaults.bootstrap_resamples)
    parser.add_argument("--bootstrap-seed", type=int, default=defaults.bootstrap_seed)
    parser.add_argument("--residual-checkpoint", default=defaults.residual_checkpoint)
    parser.add_argument("--spatial-checkpoint", default=defaults.spatial_checkpoint)
    args = parser.parse_args()

    cfg = PairedBenchmarkConfig(
        episodes=max(1, args.episodes),
        seed=args.seed,
        save_dir=args.save_dir,
        bootstrap_resamples=max(1, args.bootstrap_resamples),
        bootstrap_seed=args.bootstrap_seed,
        residual_checkpoint=args.residual_checkpoint,
        spatial_checkpoint=args.spatial_checkpoint,
    )
    residual_path, spatial_path = Path(cfg.residual_checkpoint), Path(cfg.spatial_checkpoint)
    for path in (residual_path, spatial_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    residual_model, residual_cfg, residual_payload = load_residual(residual_path)
    spatial_model, spatial_cfg, spatial_payload = load_spatial(spatial_path)
    residual_env, residual_switches = build_residual_env(residual_cfg.environment_preset)
    spatial_env, spatial_switches = build_spatial_env()
    planner_cfg = PlannerLabConfig(episodes=cfg.episodes, seed=cfg.seed - 1, environment_preset=BASELINE_PRESET)

    runners: Dict[str, Callable[[int, int], Dict[str, float]]] = {
        "frontier_planner": lambda episode, seed: run_planner_episode(planner_cfg, episode, seed),
        "spatial_target_ppo": lambda episode, seed: run_spatial_episode(
            spatial_model, spatial_env, spatial_switches, torch.device("cpu"), spatial_cfg,
            -episode, seed, False, guidance_episode=0,
        )[1],
        "planner_residual_ppo": lambda episode, seed: run_residual_episode(
            residual_model, residual_env, residual_switches, torch.device("cpu"), residual_cfg,
            -episode, seed, False, schedule_episode=0,
        )[1],
    }
    metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in runners}
    rows: List[Dict[str, object]] = []
    for index in range(cfg.episodes):
        episode, seed = index + 1, cfg.seed + index
        episode_metrics = {name: runner(episode, seed) for name, runner in runners.items()}
        for name, result in episode_metrics.items():
            metrics[name].append(result)
        planner_coverage = episode_metrics["frontier_planner"]["coverage"]
        rows.append({
            "episode": episode,
            "seed": seed,
            "environment_signature": environment_signature(seed),
            "planner_coverage": planner_coverage,
            "spatial_target_ppo_coverage": episode_metrics["spatial_target_ppo"]["coverage"],
            "planner_residual_ppo_coverage": episode_metrics["planner_residual_ppo"]["coverage"],
            "spatial_target_ppo_delta_minus_planner": episode_metrics["spatial_target_ppo"]["coverage"] - planner_coverage,
            "planner_residual_ppo_delta_minus_planner": episode_metrics["planner_residual_ppo"]["coverage"] - planner_coverage,
        })
        print(f"episode={episode:03d} seed={seed} planner={planner_coverage:.4f} spatial={episode_metrics['spatial_target_ppo']['coverage']:.4f} residual={episode_metrics['planner_residual_ppo']['coverage']:.4f}", flush=True)

    output = Path(cfg.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": asdict(cfg),
        "scope": {
            "evaluation_type": "paired held-out map evaluation conditional on two saved single-seed PPO training runs",
            "shared_environment_preset": BASELINE_PRESET,
            "same_seed_same_environment": True,
            "training_replication": "not addressed; each learned controller is one saved training run",
            "checkpoint_selection": "checkpoints were selected on the archived 50,000--50,049 range; this benchmark evaluates fresh seeds",
        },
        "checkpoints": {
            "spatial_target_ppo": {
                "path": str(spatial_path), "sha256": checkpoint_digest(spatial_path),
                "model_type": spatial_payload.get("model_type"), "checkpoint_episode": spatial_payload.get("episode"),
                "training_seed": spatial_cfg.seed,
            },
            "planner_residual_ppo": {
                "path": str(residual_path), "sha256": checkpoint_digest(residual_path),
                "model_type": residual_payload.get("model_type"), "checkpoint_episode": residual_payload.get("episode"),
                "training_seed": residual_cfg.seed,
            },
        },
        "controller_metrics": {name: mean_and_p10(controller_rows) for name, controller_rows in metrics.items()},
        "paired_comparisons": {
            "spatial_target_ppo_vs_planner": paired_summary(
                np.asarray([float(row["spatial_target_ppo_delta_minus_planner"]) for row in rows]),
                cfg.bootstrap_seed, cfg.bootstrap_resamples,
            ),
            "planner_residual_ppo_vs_planner": paired_summary(
                np.asarray([float(row["planner_residual_ppo_delta_minus_planner"]) for row in rows]),
                cfg.bootstrap_seed, cfg.bootstrap_resamples,
            ),
        },
    }
    write_csv(output / "comparison_by_episode.csv", rows)
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

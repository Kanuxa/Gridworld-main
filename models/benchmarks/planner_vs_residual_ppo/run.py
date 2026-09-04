"""Evaluate planner-residual PPO checkpoints against the frontier planner.

This runner is deliberately limited to the residual architecture.  Unlike the
earlier three-controller runner, it obtains the environment preset from the
checkpoint and can therefore evaluate the 15x15 and 31x31 residual models, or
several independently trained residual checkpoints, on identical fresh maps.
It records the map signature, checkpoint hashes, and every paired outcome.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from gui.current_environment.sensory_grid_env import EnvConfig, SensoryGridEnv
from models.model_based.ppo.planner_residual_target_ppo.model import PlannerResidualTargetActorCritic
from models.model_based.ppo.planner_residual_target_ppo.train import (
    PlannerResidualTrainConfig,
    build_env as build_residual_env,
    run_episode as run_residual_episode,
)
from models.non_model_based.partial_observation_frontier_planner.run import (
    BASELINE_PRESET,
    ENVIRONMENT_PRESETS,
    PlannerLabConfig,
    build_env as build_planner_env,
    environment_config,
    run_episode as run_planner_episode,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "runs/15x15/model_based/planner_residual_target_ppo/best_coverage.pt"
)


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 50
    seed: int = 70_000
    save_dir: str = "runs/15x15/benchmarks/frontier_planner_vs_residual_fresh"
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 25_218_029
    environment_preset: str = BASELINE_PRESET
    scenario: str = "nominal"


@dataclass(frozen=True)
class ObservationNoise:
    """Deterministic perception perturbations applied to public observations."""

    object_dropout_probability: float = 0.0
    temperature_noise_std_c: float = 0.0


SCENARIOS = {
    "nominal": ("Released environment and unmodified observations.", ObservationNoise(), lambda config: config),
    "object-dropout-15": (
        "Fifteen-percent dropout of non-empty visual object detections in every public 5x5 patch.",
        ObservationNoise(object_dropout_probability=0.15),
        lambda config: config,
    ),
    "temperature-noise-3c": (
        "Independent zero-mean 3 C Gaussian noise on each public temperature-patch value.",
        ObservationNoise(temperature_noise_std_c=3.0),
        lambda config: config,
    ),
    "resource-scarce": (
        "One restorative meat item rather than three; all other 15x15 baseline parameters are unchanged.",
        ObservationNoise(),
        lambda config: replace(config, n_meat=1),
    ),
}


class PerturbedObservationEnv:
    """Expose a deterministic noisy public-observation interface around an environment.

    The hidden grid is never changed after reset. For a given map seed and time
    step, both controllers receive the same noise draw; their different action
    histories determine which local patch is observed. This mirrors a shared
    sensor model without granting either controller privileged state.
    """

    def __init__(self, base: SensoryGridEnv, noise: ObservationNoise):
        self.base = base
        self.noise = noise
        self.episode_seed = 0

    @property
    def config(self) -> EnvConfig:
        return self.base.config

    def reset(self, seed: int | None = None):
        self.episode_seed = 0 if seed is None else int(seed)
        observation, info = self.base.reset(seed=seed)
        return self._perturb(observation), info

    def step(self, action: int, switches=None):
        observation, reward, terminated, truncated, info = self.base.step(action, switches)
        return self._perturb(observation), reward, terminated, truncated, info

    def _rng(self) -> np.random.Generator:
        # Use only the frozen map seed and elapsed time, so the two controllers
        # share a deterministic sensor-noise schedule on a matched map.
        value = (self.episode_seed * 1_000_003 + int(self.base.steps) * 104_729 + 25_218_029) % (2**63 - 1)
        return np.random.default_rng(value)

    def _perturb(self, observation: Dict[str, object]) -> Dict[str, object]:
        if self.noise == ObservationNoise():
            return observation
        result = dict(observation)
        rng = self._rng()
        if self.noise.object_dropout_probability:
            vision = np.asarray(observation["vision"], dtype=np.int64).copy()
            drop = (vision != 0) & (rng.random(vision.shape) < self.noise.object_dropout_probability)
            vision[drop] = 0
            result["vision"] = vision
        if self.noise.temperature_noise_std_c:
            temperature = np.asarray(observation["temperature_patch_c"], dtype=np.float32).copy()
            temperature += rng.normal(0.0, self.noise.temperature_noise_std_c, size=temperature.shape).astype(np.float32)
            result["temperature_patch_c"] = temperature
        return result

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def checkpoint_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_signature(seed: int, config: EnvConfig) -> str:
    """Hash the hidden fields which define a generated map instance."""
    env = SensoryGridEnv(config)
    env.reset(seed=seed)
    digest = sha256()
    digest.update(env.reveal_world_ids().tobytes())
    digest.update(env.reveal_temperature_field_c().tobytes())
    digest.update(str(int(env.direction)).encode("ascii"))
    return digest.hexdigest()


def config_from_payload(config_type: type, payload: Dict[str, Any]):
    allowed = {field.name for field in fields(config_type)}
    values = payload.get("train_config", {})
    return config_type(**{key: value for key, value in values.items() if key in allowed})


def load_checkpoint(
    path: Path,
) -> Tuple[PlannerResidualTargetActorCritic, PlannerResidualTrainConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_payload(PlannerResidualTrainConfig, payload)
    model = PlannerResidualTargetActorCritic()
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
        "mean_coverage_delta_residual_minus_planner": float(np.mean(deltas)),
        "median_coverage_delta_residual_minus_planner": float(np.median(deltas)),
        "bootstrap_95_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_ci_high": float(np.quantile(bootstrap, 0.975)),
        "residual_wins": wins,
        "planner_wins": losses,
        "ties": ties,
        "two_sided_sign_test_p": two_sided_sign_test(wins, losses),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="Held-out matched planner-versus-residual-PPO benchmark")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="First generated map seed.")
    parser.add_argument("--save-dir", default=defaults.save_dir)
    parser.add_argument("--bootstrap-resamples", type=int, default=defaults.bootstrap_resamples)
    parser.add_argument("--bootstrap-seed", type=int, default=defaults.bootstrap_seed)
    parser.add_argument(
        "--preset", choices=ENVIRONMENT_PRESETS,
        help="Optional check on the preset stored in every checkpoint.",
    )
    parser.add_argument(
        "--scenario", choices=tuple(SCENARIOS), default=defaults.scenario,
        help="Fixed environment or perception stress condition for every matched map.",
    )
    parser.add_argument(
        "--residual-checkpoint", action="append", dest="checkpoints",
        help="Residual PPO checkpoint. Repeat to compare independently trained checkpoints.",
    )
    parser.add_argument(
        "--remove-persistent-prior", action="store_true",
        help=(
            "Diagnostic inference ablation: set planner target-score weight and "
            "teacher tie bonus to zero. This does not retrain the checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in (args.checkpoints or [str(DEFAULT_CHECKPOINT)])]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    loaded = [load_checkpoint(path) for path in paths]
    if args.remove_persistent_prior:
        loaded = [
            (
                model,
                replace(
                    train_cfg,
                    planner_score_weight=0.0,
                    planner_teacher_tie_bonus=0.0,
                ),
                payload,
            )
            for model, train_cfg, payload in loaded
        ]
    presets = {config.environment_preset for _, config, _ in loaded}
    if len(presets) != 1:
        raise ValueError(f"All checkpoints must use one environment preset; found {sorted(presets)}.")
    preset = presets.pop()
    if args.preset and args.preset != preset:
        raise ValueError(f"Checkpoint preset is {preset}, not requested {args.preset}.")
    scenario_description, noise, scenario_builder = SCENARIOS[args.scenario]
    scenario_env_config = scenario_builder(environment_config(preset))
    cfg = BenchmarkConfig(
        episodes=max(1, args.episodes),
        seed=args.seed,
        save_dir=args.save_dir,
        bootstrap_resamples=max(1, args.bootstrap_resamples),
        bootstrap_seed=args.bootstrap_seed,
        environment_preset=preset,
        scenario=args.scenario,
    )

    labels: List[str] = []
    for _, train_cfg, _ in loaded:
        label = f"residual_ppo_seed_{train_cfg.seed}"
        if args.remove_persistent_prior:
            label += "_without_persistent_prior"
        if label in labels:
            raise ValueError(f"Duplicate training seed label {label}; provide checkpoints from distinct runs.")
        labels.append(label)
    _, switches = build_residual_env(preset)
    residual_envs = [
        (PerturbedObservationEnv(SensoryGridEnv(replace(scenario_env_config)), noise), switches)
        for _ in loaded
    ]
    planner_cfg = PlannerLabConfig(
        episodes=cfg.episodes, seed=cfg.seed - 1, environment_preset=preset,
    )
    metrics: Dict[str, List[Dict[str, float]]] = {"frontier_planner": []}
    metrics.update({label: [] for label in labels})
    rows: List[Dict[str, object]] = []
    for index in range(cfg.episodes):
        episode, seed = index + 1, cfg.seed + index
        planner_env = PerturbedObservationEnv(SensoryGridEnv(replace(scenario_env_config)), noise)
        planner = run_planner_episode(planner_cfg, episode, seed, env=planner_env)
        metrics["frontier_planner"].append(planner)
        row: Dict[str, object] = {
            "episode": episode,
            "seed": seed,
            "environment_signature": environment_signature(seed, scenario_env_config),
            "planner_coverage": planner["coverage"],
        }
        line = [f"episode={episode:03d}", f"seed={seed}", f"planner={planner['coverage']:.4f}"]
        for checkpoint_index, (label, (model, train_cfg, _), (env, switches)) in enumerate(
            zip(labels, loaded, residual_envs)
        ):
            result = run_residual_episode(
                model, env, switches, torch.device("cpu"), train_cfg,
                -episode, seed, False, schedule_episode=0,
            )[1]
            metrics[label].append(result)
            delta = result["coverage"] - planner["coverage"]
            row[f"{label}_coverage"] = result["coverage"]
            row[f"{label}_delta_minus_planner"] = delta
            line.append(f"{label}={result['coverage']:.4f}")
        rows.append(row)
        print(" ".join(line), flush=True)

    comparisons = {
        label: paired_summary(
            np.asarray([float(row[f"{label}_delta_minus_planner"]) for row in rows]),
            cfg.bootstrap_seed + index,
            cfg.bootstrap_resamples,
        )
        for index, label in enumerate(labels)
    }
    checkpoint_records = {
        label: {
            "path": str(path),
            "sha256": checkpoint_digest(path),
            "model_type": payload.get("model_type"),
            "checkpoint_episode": payload.get("episode"),
            "training_seed": train_cfg.seed,
            "evaluation_planner_score_weight": train_cfg.planner_score_weight,
            "evaluation_planner_teacher_tie_bonus": train_cfg.planner_teacher_tie_bonus,
        }
        for label, path, (_, train_cfg, payload) in zip(labels, paths, loaded)
    }
    summary = {
        "config": asdict(cfg),
        "scope": {
            "evaluation_type": "paired held-out map evaluation conditional on saved residual-PPO checkpoints",
            "same_seed_same_environment": True,
            "scenario": args.scenario,
            "scenario_description": scenario_description,
            "perception_noise": asdict(noise),
            "training_replication": (
                "The individual checkpoint comparisons estimate map variation. "
                "When several training seeds are supplied, their separate estimates document "
                "replication but do not create a population-level confidence interval."
            ),
            "checkpoint_selection": "The supplied checkpoints are evaluated on fresh map seeds selected by this command.",
            "inference_ablation": (
                "Planner target-score weight and teacher tie bonus are set to zero only at evaluation; "
                "the checkpoint was trained with its persistent planner prior."
                if args.remove_persistent_prior
                else "No evaluation-time controller ablation is applied."
            ),
        },
        "scenario_environment_config": asdict(scenario_env_config),
        "checkpoints": checkpoint_records,
        "controller_metrics": {name: mean_and_p10(controller_rows) for name, controller_rows in metrics.items()},
        "paired_comparisons": comparisons,
        "across_checkpoint_descriptives": {
            "number_of_checkpoints": len(labels),
            "mean_of_checkpoint_mean_deltas": float(np.mean([
                result["mean_coverage_delta_residual_minus_planner"] for result in comparisons.values()
            ])),
            "minimum_checkpoint_mean_delta": float(min(
                result["mean_coverage_delta_residual_minus_planner"] for result in comparisons.values()
            )),
            "maximum_checkpoint_mean_delta": float(max(
                result["mean_coverage_delta_residual_minus_planner"] for result in comparisons.values()
            )),
        },
    }
    output = Path(cfg.save_dir)
    write_csv(output / "comparison_by_episode.csv", rows)
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

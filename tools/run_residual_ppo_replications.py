"""Run the pre-specified independent-seed residual-PPO replication protocol.

The script intentionally trains sequentially so that one accelerator is not
oversubscribed.  It writes the protocol before starting any training and then
evaluates all completed checkpoints on one fresh, shared map range.  The
script does not reinterpret an interrupted run as evidence: a missing
checkpoint stops the evaluation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


DEFAULT_SEEDS = (13, 29, 53)
DEFAULT_TEST_SEED = 86_000
DEFAULT_TEST_EPISODES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate independent residual-PPO replications."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-root",
        default="runs/15x15/replications/planner_residual_target_ppo",
    )
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--test-episodes", type=int, default=DEFAULT_TEST_EPISODES)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=25_218_029)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the immutable protocol and print commands without running them.",
    )
    return parser.parse_args()


def command_for_training(args: argparse.Namespace, seed: int, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "models.model_based.ppo.planner_residual_target_ppo.train",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--save-dir",
        str(root / f"seed_{seed}"),
        "--no-traces",
    ]


def command_for_evaluation(
    args: argparse.Namespace, root: Path, checkpoints: Sequence[Path]
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "models.benchmarks.planner_vs_residual_ppo.run",
        "--episodes",
        str(args.test_episodes),
        "--seed",
        str(args.test_seed),
        "--bootstrap-resamples",
        str(args.bootstrap_resamples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--save-dir",
        str(root / "heldout_aggregate"),
    ]
    for checkpoint in checkpoints:
        command.extend(("--residual-checkpoint", str(checkpoint)))
    return command


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide one or more distinct training seeds.")
    if args.episodes < 1 or args.test_episodes < 1:
        raise ValueError("Episode counts must be positive.")

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "replication_protocol.json"
    protocol = {
        "purpose": "Independent-training-seed replication of planner-residual PPO.",
        "training_seeds": list(args.seeds),
        "training_episodes_per_seed": args.episodes,
        "environment_preset": "15x15-baseline",
        "checkpoint_selection": "best_coverage.pt selected only from each run's fixed validation evaluation.",
        "heldout_evaluation": {
            "map_seed_start": args.test_seed,
            "episodes": args.test_episodes,
            "controllers": "one unchanged frontier planner and every completed residual-PPO seed",
            "statistics": "paired mean coverage difference, bootstrap 95% CI, exact two-sided sign test, and P10",
        },
        "exclusions": [
            "The fresh held-out map range must not be used for checkpoint selection.",
            "Interrupted training or partial seed sets are not reported as a completed replication.",
            "No changes to reward, architecture, or planner settings after the protocol is written.",
        ],
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError(
                f"Existing protocol differs at {protocol_path}; choose a new --out-root."
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    training_commands = [command_for_training(args, seed, root) for seed in args.seeds]
    for command in training_commands:
        print(" ".join(command), flush=True)
    checkpoints = [root / f"seed_{seed}" / "best_coverage.pt" for seed in args.seeds]
    print(" ".join(command_for_evaluation(args, root, checkpoints)), flush=True)
    if args.dry_run:
        return

    for command in training_commands:
        subprocess.run(command, check=True)
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise RuntimeError(f"Replication training did not produce expected checkpoints: {missing}")
    subprocess.run(command_for_evaluation(args, root, checkpoints), check=True)


if __name__ == "__main__":
    main()

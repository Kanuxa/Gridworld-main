"""Generate planner-residual checkpoint traces with the persistent planner-score baseline."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from models.model_based.ppo.planner_residual_target_ppo.model import PlannerResidualTargetActorCritic
from models.non_model_based.partial_observation_frontier_planner.run import ENVIRONMENT_PRESETS
from models.model_based.ppo.planner_residual_target_ppo.train import (
    LEGACY_MODEL_TYPE,
    MODEL_TYPE,
    PlannerResidualTrainConfig,
    build_env,
    choose_device,
    run_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a planner-residual spatial-target checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/15x15/planner_residual_target_ppo/rollouts")
    parser.add_argument("--preset", choices=ENVIRONMENT_PRESETS, default=None, help="Override the environment preset stored in the checkpoint.")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("model_type") not in {MODEL_TYPE, LEGACY_MODEL_TYPE}:
        raise ValueError("Checkpoint is not a planner-residual spatial-target PPO model.")
    allowed = {field.name for field in fields(PlannerResidualTrainConfig)}
    cfg = PlannerResidualTrainConfig(**{key: value for key, value in payload.get("train_config", {}).items() if key in allowed})
    net = PlannerResidualTargetActorCritic().to(device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    env, switches = build_env(args.preset or cfg.environment_preset)
    output = Path(args.output_dir)
    for index in range(args.episodes):
        _, summary = run_episode(net, env, switches, device, cfg, index + 1, args.seed + index, args.stochastic, output, schedule_episode=0)
        print(f"episode={index + 1:03d} seed={args.seed + index} {summary}", flush=True)


if __name__ == "__main__":
    main()

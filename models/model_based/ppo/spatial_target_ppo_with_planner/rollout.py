"""Generate traces from a released spatial-target hierarchy checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from models.model_based.ppo.spatial_target_ppo_with_planner.model import TargetMapActorCritic
from models.model_based.ppo.spatial_target_ppo_with_planner.train import (
    LEGACY_MODEL_TYPE,
    MODEL_TYPE,
    TargetTrainConfig,
    build_env,
    choose_device,
    run_episode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a spatial target PPO checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="runs/15x15/spatial_target_ppo_with_planner/rollouts")
    parser.add_argument("--stochastic", action="store_true", help="Sample targets instead of using greedy inference")
    args = parser.parse_args()

    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("model_type") not in {MODEL_TYPE, LEGACY_MODEL_TYPE}:
        raise ValueError("Checkpoint is not a spatial-target PPO-with-planner model.")
    allowed = {field.name for field in fields(TargetTrainConfig)}
    cfg = TargetTrainConfig(**{key: value for key, value in payload.get("train_config", {}).items() if key in allowed})
    net = TargetMapActorCritic().to(device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    env, switches = build_env()
    output_dir = Path(args.output_dir)

    for index in range(args.episodes):
        _, summary = run_episode(
            net, env, switches, device, cfg, index + 1, args.seed + index,
            stochastic=args.stochastic, trace_dir=output_dir,
            guidance_episode=0,
        )
        print(f"episode={index + 1:03d} seed={args.seed + index} {summary}", flush=True)


if __name__ == "__main__":
    main()

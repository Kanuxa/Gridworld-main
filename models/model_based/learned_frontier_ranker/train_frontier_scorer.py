"""Supervised training for FrontierScoreNet from phase-1 expert trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.shared.belief_map_tools.models import FrontierScoreNet
from models.shared.belief_map_tools.phase1_dataset import Phase1DecisionDataset, discover_trajectory_files, split_files_by_seed


@dataclass
class TrainingConfig:
    trajectory_dir: str = "runs/15x15/two_phase_belief_map_planner/phase1_trajectories"
    output_dir: str = "runs/15x15/learned_frontier_ranker/frontier_scorer"
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_channels: int = 48
    seed: int = 7
    device: str = "auto"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = total = 0
    distance_total = 0.0
    with torch.no_grad():
        for maps, scalars, targets in loader:
            maps, scalars, targets = maps.to(device), scalars.to(device), targets.to(device)
            logits = model(maps, scalars).flatten(1)
            losses.append(float(nn.functional.cross_entropy(logits, targets).item()))
            predicted = torch.argmax(logits, dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.numel())
            width = maps.shape[-1]
            distance_total += float((torch.abs(predicted // width - targets // width) + torch.abs(predicted % width - targets % width)).sum().item())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(correct / max(1, total)),
        "mean_manhattan_error": float(distance_total / max(1, total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FrontierScoreNet from phase-1 expert trajectories.")
    parser.add_argument("--trajectory-dir", type=Path, default=Path(TrainingConfig.trajectory_dir))
    parser.add_argument("--output-dir", type=Path, default=Path(TrainingConfig.output_dir))
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--hidden-channels", type=int, default=TrainingConfig.hidden_channels)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", type=str, default=TrainingConfig.device)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    cfg = TrainingConfig(
        trajectory_dir=str(args.trajectory_dir), output_dir=str(args.output_dir), epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, hidden_channels=args.hidden_channels,
        seed=args.seed, device=args.device,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    files = discover_trajectory_files(args.trajectory_dir)
    split = split_files_by_seed(files, cfg.seed)
    if not split["validation"] or not split["test"]:
        raise ValueError("At least seven seed files are required for train/validation/test splitting.")
    datasets = {name: Phase1DecisionDataset(paths) for name, paths in split.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=cfg.batch_size),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size),
    }
    map_channels, height, width = datasets["train"].map_shape
    model = FrontierScoreNet(map_channels, datasets["train"].scalar_dim, cfg.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_split.json").write_text(json.dumps({name: [path.name for path in paths] for name, paths in split.items()}, indent=2))
    with (args.output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump({**asdict(cfg), "map_shape": [map_channels, height, width]}, handle, indent=2)

    best_validation = float("inf")
    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_mean_manhattan_error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            losses: list[float] = []
            for maps, scalars, targets in loaders["train"]:
                maps, scalars, targets = maps.to(device), scalars.to(device), targets.to(device)
                logits = model(maps, scalars).flatten(1)
                loss = nn.functional.cross_entropy(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))
            validation = evaluate(model, loaders["validation"], device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_mean_manhattan_error": validation["mean_manhattan_error"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                "epoch={:03d}/{:03d} train_loss={:.4f} val_loss={:.4f} val_acc={:.3f} val_manhattan={:.2f}".format(
                    epoch, cfg.epochs, row["train_loss"], row["validation_loss"], row["validation_accuracy"], row["validation_mean_manhattan_error"]
                ),
                flush=True,
            )
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_kwargs": {"map_channels": map_channels, "scalar_dim": datasets["train"].scalar_dim, "hidden_channels": cfg.hidden_channels},
                        "training_config": asdict(cfg),
                        "best_validation": validation,
                    },
                    args.output_dir / "best_model.pt",
                )
    payload = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    test = evaluate(model, loaders["test"], device)
    print("test_loss={loss:.4f} test_acc={accuracy:.3f} test_manhattan={mean_manhattan_error:.2f}".format(**test))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test, handle, indent=2)


if __name__ == "__main__":
    main()

"""Supervised phase-2 target and route-quality training for ExplorationScoreNet."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.shared.belief_map_tools.models import ExplorationScoreNet
from models.shared.belief_map_tools.phase1_dataset import discover_trajectory_files, split_files_by_seed
from models.shared.belief_map_tools.phase2_dataset import Phase2DecisionDataset
from models.model_based.learned_frontier_ranker.train_frontier_scorer import choose_device


@dataclass
class TrainingConfig:
    trajectory_dir: str = "runs/15x15/two_phase_belief_map_planner/phase2_trajectories"
    output_dir: str = "runs/15x15/learned_frontier_ranker/exploration_scorer"
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_channels: int = 48
    auxiliary_weight: float = 0.25
    survival_weight: float = 0.10
    seed: int = 7
    device: str = "auto"


def metrics(model: nn.Module, loader: DataLoader, device: torch.device, pos_weight: torch.Tensor) -> Dict[str, float]:
    model.eval()
    target_losses=[]; aux_losses=[]; survival_losses=[]; correct=total=0; distance_total=0.0; survival_correct=0
    with torch.no_grad():
        for maps, scalars, targets, auxiliary in loader:
            maps, scalars, targets, auxiliary = maps.to(device), scalars.to(device), targets.to(device), auxiliary.to(device)
            logits, predicted_aux = model(maps, scalars)
            flattened = logits.flatten(1)
            target_losses.append(float(nn.functional.cross_entropy(flattened, targets).item()))
            aux_losses.append(float(nn.functional.smooth_l1_loss(predicted_aux[:, :6], auxiliary[:, :6]).item()))
            survival_losses.append(float(nn.functional.binary_cross_entropy_with_logits(predicted_aux[:, 6], auxiliary[:, 6], pos_weight=pos_weight).item()))
            predicted = torch.argmax(flattened, dim=1)
            correct += int((predicted == targets).sum().item())
            total += int(targets.numel())
            width = maps.shape[-1]
            distance_total += float((torch.abs(predicted // width - targets // width) + torch.abs(predicted % width - targets % width)).sum().item())
            survival_correct += int(((torch.sigmoid(predicted_aux[:, 6]) >= 0.5) == (auxiliary[:, 6] >= 0.5)).sum().item())
    return {
        "target_loss": float(np.mean(target_losses)),
        "auxiliary_loss": float(np.mean(aux_losses)),
        "survival_loss": float(np.mean(survival_losses)),
        "target_accuracy": float(correct / max(1, total)),
        "mean_manhattan_error": float(distance_total / max(1, total)),
        "survival_accuracy": float(survival_correct / max(1, total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ExplorationScoreNet from phase-2 expert trajectories.")
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
    cfg = TrainingConfig(trajectory_dir=str(args.trajectory_dir), output_dir=str(args.output_dir), epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, hidden_channels=args.hidden_channels, seed=args.seed, device=args.device)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    split = split_files_by_seed(discover_trajectory_files(args.trajectory_dir), cfg.seed)
    datasets = {name: Phase2DecisionDataset(paths) for name, paths in split.items()}
    loaders = {"train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True), "validation": DataLoader(datasets["validation"], batch_size=cfg.batch_size), "test": DataLoader(datasets["test"], batch_size=cfg.batch_size)}
    map_channels, height, width = datasets["train"].map_shape
    model = ExplorationScoreNet(map_channels, datasets["train"].scalar_dim, cfg.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    positive_rate = float(np.mean(datasets["train"].auxiliary[:, 6]))
    pos_weight = torch.tensor([(1.0 - positive_rate) / max(positive_rate, 1e-6)], device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_split.json").write_text(json.dumps({name: [path.name for path in paths] for name, paths in split.items()}, indent=2))
    (args.output_dir / "train_config.json").write_text(json.dumps({**asdict(cfg), "map_shape": [map_channels, height, width], "survival_positive_rate": positive_rate}, indent=2))
    best_distance = float("inf")
    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields=["epoch","train_total_loss","validation_target_loss","validation_auxiliary_loss","validation_survival_loss","validation_target_accuracy","validation_mean_manhattan_error","validation_survival_accuracy"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for epoch in range(1,cfg.epochs+1):
            model.train(); total_losses=[]
            for maps, scalars, targets, auxiliary in loaders["train"]:
                maps, scalars, targets, auxiliary=maps.to(device),scalars.to(device),targets.to(device),auxiliary.to(device)
                logits,predicted_aux=model(maps,scalars)
                target_loss=nn.functional.cross_entropy(logits.flatten(1),targets)
                auxiliary_loss=nn.functional.smooth_l1_loss(predicted_aux[:,:6],auxiliary[:,:6])
                survival_loss=nn.functional.binary_cross_entropy_with_logits(predicted_aux[:,6],auxiliary[:,6],pos_weight=pos_weight)
                loss=target_loss+cfg.auxiliary_weight*auxiliary_loss+cfg.survival_weight*survival_loss
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
                total_losses.append(float(loss.item()))
            validation=metrics(model,loaders["validation"],device,pos_weight)
            row={"epoch":epoch,"train_total_loss":float(np.mean(total_losses)),**{f"validation_{name}":value for name,value in validation.items()}}
            writer.writerow(row); handle.flush()
            print("epoch={:03d}/{:03d} train={:.4f} val_acc={:.3f} val_manhattan={:.2f} val_survival_acc={:.3f}".format(epoch,cfg.epochs,row["train_total_loss"],row["validation_target_accuracy"],row["validation_mean_manhattan_error"],row["validation_survival_accuracy"]),flush=True)
            if validation["mean_manhattan_error"] < best_distance:
                best_distance=validation["mean_manhattan_error"]
                torch.save({"model_state_dict":model.state_dict(),"model_kwargs":{"map_channels":map_channels,"scalar_dim":datasets["train"].scalar_dim,"hidden_channels":cfg.hidden_channels},"training_config":asdict(cfg),"best_validation":validation},args.output_dir/"best_target_model.pt")
    payload=torch.load(args.output_dir/"best_target_model.pt",map_location=device,weights_only=False); model.load_state_dict(payload["model_state_dict"])
    test=metrics(model,loaders["test"],device,pos_weight)
    print("test_target_acc={target_accuracy:.3f} test_manhattan={mean_manhattan_error:.2f} test_survival_acc={survival_accuracy:.3f}".format(**test))
    (args.output_dir/"test_metrics.json").write_text(json.dumps(test,indent=2))


if __name__ == "__main__":
    main()

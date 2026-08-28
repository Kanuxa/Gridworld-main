"""Train an early-stopped bootstrap ensemble for safe frontier selection.

Unlike the original single-model classifier, this trainer masks invalid target
cells before the ranking loss and trains several seed/bootstrap members.  The
deployed controller can use their disagreement as an uncertainty signal and
fall back to the deterministic V11 expert when the committee is unreliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from models.shared.belief_map_tools.models import FrontierScoreNet
from models.shared.belief_map_tools.phase1_dataset import Phase1DecisionDataset, discover_trajectory_files, split_files_by_seed


@dataclass(frozen=True)
class TrainingConfig:
    trajectory_dir: str = "runs/15x15/two_phase_belief_map_planner/phase1_trajectories"
    output_dir: str = "runs/15x15/learned_frontier_ranker/frontier_ensemble"
    members: int = 5
    epochs: int = 12
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_channels: int = 48
    seed: int = 2026
    device: str = "auto"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def candidate_mask(maps: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
    """Mask cells the planner could never select in phase 1.

    The first six channels intentionally preserve V11's schema even when a
    newer extended-belief trajectory adds extra channels.  Valid targets are
    seen, temperature-known, non-forbidden viewpoints with at least one unseen
    cell in their 5x5 observation window.
    """
    if maps.ndim != 4 or maps.shape[1] < 6:
        raise ValueError("Expected map tensor with at least the V11 six leading channels")
    seen = maps[:, 0] > 0.5
    temperature_known = maps[:, 2] > 0.5
    forbidden = maps[:, 5] > 0.5
    unseen = (~seen).to(dtype=maps.dtype).unsqueeze(1)
    window = F.conv2d(unseen, torch.ones((1, 1, 5, 5), device=maps.device, dtype=maps.dtype), padding=2).squeeze(1)
    valid = seen & temperature_known & ~forbidden & (window > 0.0)
    if targets is not None:
        # Expert labels are the authoritative candidate set for a training
        # record; retaining the label also makes malformed legacy examples
        # fail safely rather than creating an all--inf cross-entropy row.
        flat = valid.flatten(1)
        flat.scatter_(1, targets.unsqueeze(1), True)
        valid = flat.reshape_as(valid)
    return valid


def masked_target_loss(logits: torch.Tensor, maps: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = candidate_mask(maps, targets)
    masked_logits = logits.masked_fill(~valid, -1e9)
    return F.cross_entropy(masked_logits.flatten(1), targets), masked_logits


def evaluate(model: FrontierScoreNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    distances: list[float] = []
    with torch.no_grad():
        for maps, scalars, targets in loader:
            maps, scalars, targets = maps.to(device), scalars.to(device), targets.to(device)
            logits = model(maps, scalars)
            loss, masked_logits = masked_target_loss(logits, maps, targets)
            predictions = masked_logits.flatten(1).argmax(dim=1)
            width = maps.shape[-1]
            distances.extend(
                (
                    (predictions // width - targets // width).abs()
                    + (predictions % width - targets % width).abs()
                )
                .detach()
                .cpu()
                .tolist()
            )
            losses.append(float(loss.item()) * int(targets.shape[0]))
            correct += int((predictions == targets).sum().item())
            total += int(targets.shape[0])
    if total == 0:
        raise ValueError("Evaluation loader contains no examples")
    return {
        "loss": float(sum(losses) / total),
        "accuracy": float(correct / total),
        "mean_manhattan_error": float(np.mean(distances)),
    }


def bootstrap_paths(paths: Sequence[Path], seed: int) -> list[Path]:
    if not paths:
        raise ValueError("Cannot bootstrap an empty trajectory split")
    rng = np.random.default_rng(seed)
    return [paths[int(index)] for index in rng.integers(0, len(paths), size=len(paths))]


def train_member(
    member_index: int,
    train_paths: Sequence[Path],
    validation_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
) -> dict:
    member_seed = cfg.seed + member_index
    random.seed(member_seed)
    np.random.seed(member_seed)
    torch.manual_seed(member_seed)
    train_dataset = Phase1DecisionDataset(train_paths)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    map_channels, _, _ = train_dataset.map_shape
    model = FrontierScoreNet(map_channels=map_channels, scalar_dim=train_dataset.scalar_dim, hidden_channels=cfg.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    output_dir = Path(cfg.output_dir)
    checkpoint = output_dir / f"member_{member_index:02d}.pt"
    log_path = output_dir / f"member_{member_index:02d}_training.csv"
    best_loss = float("inf")
    stale_epochs = 0
    epochs_completed = 0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_mean_manhattan_error"])
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            count = 0
            for maps, scalars, targets in train_loader:
                maps, scalars, targets = maps.to(device), scalars.to(device), targets.to(device)
                logits = model(maps, scalars)
                loss, _ = masked_target_loss(logits, maps, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += float(loss.item()) * int(targets.shape[0])
                count += int(targets.shape[0])
            validation = evaluate(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, count),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_mean_manhattan_error": validation["mean_manhattan_error"],
            }
            writer.writerow(row)
            handle.flush()
            epochs_completed = epoch
            print(
                "member={:02d} epoch={:02d}/{:02d} train={:.4f} val={:.4f} acc={:.3f} distance={:.2f}".format(
                    member_index,
                    epoch,
                    cfg.epochs,
                    row["train_loss"],
                    row["validation_loss"],
                    row["validation_accuracy"],
                    row["validation_mean_manhattan_error"],
                ),
                flush=True,
            )
            if validation["loss"] < best_loss - 1e-5:
                best_loss = validation["loss"]
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_kwargs": {
                            "map_channels": map_channels,
                            "scalar_dim": train_dataset.scalar_dim,
                            "hidden_channels": cfg.hidden_channels,
                        },
                        "member_index": member_index,
                        "member_seed": member_seed,
                        "best_validation": validation,
                        "training_config": asdict(cfg),
                    },
                    checkpoint,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    test = evaluate(model, test_loader, device)
    return {
        "member": member_index,
        "seed": member_seed,
        "epochs_completed": epochs_completed,
        "best_validation": payload["best_validation"],
        "test": test,
        "checkpoint": checkpoint.name,
        "bootstrap_files": [path.name for path in train_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=Path(TrainingConfig.trajectory_dir))
    parser.add_argument("--output-dir", type=Path, default=Path(TrainingConfig.output_dir))
    parser.add_argument("--members", type=int, default=TrainingConfig.members)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--hidden-channels", type=int, default=TrainingConfig.hidden_channels)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", type=str, default=TrainingConfig.device)
    args = parser.parse_args()
    if min(args.members, args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("members, epochs, patience, and batch-size must be positive")
    cfg = TrainingConfig(
        trajectory_dir=str(args.trajectory_dir),
        output_dir=str(args.output_dir),
        members=args.members,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        seed=args.seed,
        device=args.device,
    )
    files = discover_trajectory_files(args.trajectory_dir)
    split = split_files_by_seed(files, cfg.seed)
    if not split["validation"] or not split["test"]:
        raise ValueError("Need enough distinct trajectory seeds for validation and test splits")
    validation_dataset = Phase1DecisionDataset(split["validation"])
    test_dataset = Phase1DecisionDataset(split["test"])
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    device = choose_device(cfg.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for member_index in range(cfg.members):
        member_paths = bootstrap_paths(split["train"], cfg.seed + member_index)
        reports.append(train_member(member_index, member_paths, validation_loader, test_loader, cfg, device))
    report = {
        "training_config": asdict(cfg),
        "device": str(device),
        "seed_split": {name: [path.name for path in paths] for name, paths in split.items()},
        "members": reports,
        "ensemble_test_accuracy_mean": float(np.mean([member["test"]["accuracy"] for member in reports])),
        "ensemble_test_manhattan_mean": float(np.mean([member["test"]["mean_manhattan_error"] for member in reports])),
    }
    (args.output_dir / "ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "members={} test_accuracy_mean={:.3f} test_manhattan_mean={:.2f} output={}".format(
            cfg.members,
            report["ensemble_test_accuracy_mean"],
            report["ensemble_test_manhattan_mean"],
            args.output_dir.resolve(),
        )
    )


if __name__ == "__main__":
    main()

"""Seed-aware loading of phase-1 expert trajectory records."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def discover_trajectory_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("seed_*.npz"))
    if not files:
        raise FileNotFoundError(f"No phase-1 trajectory files found in {directory}")
    return files


def split_files_by_seed(files: Sequence[Path], seed: int) -> Dict[str, List[Path]]:
    """Create a deterministic 70/15/15 train/validation/test seed split."""
    rng = np.random.default_rng(seed)
    ordered = list(files)
    rng.shuffle(ordered)
    count = len(ordered)
    train_end = max(1, int(round(count * 0.70)))
    valid_end = max(train_end + 1, int(round(count * 0.85)))
    valid_end = min(valid_end, count)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:valid_end],
        "test": ordered[valid_end:],
    }


class Phase1DecisionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Individual phase-1 expert decisions, while splitting at the seed level."""

    def __init__(self, files: Iterable[Path]):
        maps: List[np.ndarray] = []
        scalars: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        self.seed_count = 0
        for path in files:
            with np.load(path) as data:
                map_channels = np.asarray(data["map_channels"], dtype=np.float32)
                scalar_values = np.asarray(data["scalars"], dtype=np.float32)
                target_values = np.asarray(data["target"], dtype=np.int64)
            if map_channels.shape[0] == 0:
                continue
            if map_channels.shape[0] != scalar_values.shape[0] or map_channels.shape[0] != target_values.shape[0]:
                raise ValueError(f"Mismatched decision counts in {path}")
            maps.append(map_channels)
            scalars.append(scalar_values)
            targets.append(target_values)
            self.seed_count += 1
        if not maps:
            raise ValueError("No non-empty phase-1 decision records were loaded.")
        self.maps = np.concatenate(maps, axis=0)
        self.scalars = np.concatenate(scalars, axis=0)
        target_rows = np.concatenate(targets, axis=0)
        rows, cols = self.maps.shape[-2:]
        if np.any(target_rows[:, 0] < 0) or np.any(target_rows[:, 0] >= rows) or np.any(target_rows[:, 1] < 0) or np.any(target_rows[:, 1] >= cols):
            raise ValueError("Target cells fall outside trajectory map dimensions.")
        self.targets = target_rows[:, 0] * cols + target_rows[:, 1]

    @property
    def map_shape(self) -> Tuple[int, int, int]:
        return tuple(int(value) for value in self.maps.shape[1:])

    @property
    def scalar_dim(self) -> int:
        return int(self.scalars.shape[1])

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.maps[index]),
            torch.from_numpy(self.scalars[index]),
            torch.tensor(self.targets[index], dtype=torch.long),
        )

"""Loading of phase-2 expert decisions and route-quality labels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Phase2DecisionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Phase-2 targets plus normalised auxiliary route-quality labels."""

    def __init__(self, files: Iterable[Path]):
        maps: List[np.ndarray] = []
        scalars: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        self.seed_count = 0
        required = (
            "map_channels", "scalars", "target", "new_visited_along_route", "visibility_gain",
            "route_cost", "route_turns", "projected_health_after_route", "projected_energy_cost", "survival_feasible",
        )
        for path in files:
            with np.load(path) as data:
                missing = [name for name in required if name not in data]
                if missing:
                    raise ValueError(f"Missing phase-2 labels in {path}: {missing}")
                map_values = np.asarray(data["map_channels"], dtype=np.float32)
                scalar_values = np.asarray(data["scalars"], dtype=np.float32)
                target_values = np.asarray(data["target"], dtype=np.int64)
                # Fixed normalisation keeps all Huber regression heads on a
                # comparable scale without leaking statistics from test seeds.
                aux_values = np.column_stack([
                    np.asarray(data["new_visited_along_route"], dtype=np.float32) / 15.0,
                    np.asarray(data["visibility_gain"], dtype=np.float32) / 25.0,
                    np.asarray(data["route_cost"], dtype=np.float32) / 50.0,
                    np.asarray(data["route_turns"], dtype=np.float32) / 10.0,
                    np.asarray(data["projected_health_after_route"], dtype=np.float32) / 10.0,
                    np.asarray(data["projected_energy_cost"], dtype=np.float32) / 50.0,
                    np.asarray(data["survival_feasible"], dtype=np.float32),
                ])
            count = map_values.shape[0]
            if count == 0:
                continue
            if any(values.shape[0] != count for values in (scalar_values, target_values, aux_values)):
                raise ValueError(f"Mismatched decision counts in {path}")
            maps.append(map_values)
            scalars.append(scalar_values)
            targets.append(target_values)
            labels.append(aux_values)
            self.seed_count += 1
        if not maps:
            raise ValueError("No non-empty phase-2 decision records were loaded.")
        self.maps = np.concatenate(maps, axis=0)
        self.scalars = np.concatenate(scalars, axis=0)
        target_rows = np.concatenate(targets, axis=0)
        rows, cols = self.maps.shape[-2:]
        if np.any(target_rows[:, 0] < 0) or np.any(target_rows[:, 0] >= rows) or np.any(target_rows[:, 1] < 0) or np.any(target_rows[:, 1] >= cols):
            raise ValueError("Target cells fall outside trajectory map dimensions.")
        self.targets = target_rows[:, 0] * cols + target_rows[:, 1]
        self.auxiliary = np.concatenate(labels, axis=0)

    @property
    def map_shape(self) -> Tuple[int, int, int]:
        return tuple(int(value) for value in self.maps.shape[1:])

    @property
    def scalar_dim(self) -> int:
        return int(self.scalars.shape[1])

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.maps[index]),
            torch.from_numpy(self.scalars[index]),
            torch.tensor(self.targets[index], dtype=torch.long),
            torch.from_numpy(self.auxiliary[index]),
        )

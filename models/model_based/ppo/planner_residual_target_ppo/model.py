"""Spatial target actor-critic for the V8 planner-residual policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class PlannerResidualTargetActorCritic(nn.Module):
    """Scores a map target; planner scores are added outside the network.

    The actor deliberately retains spatial positions through the target head.
    It emits only the learned residual; the deterministic planner contributes
    its persistent target-score baseline in ``coverage_v8.train``.
    """

    def __init__(self, map_channels: int = 15, scalar_dim: int = 6, width: int = 64):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(map_channels, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            ResidualBlock(width),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 96),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 48), nn.SiLU())
        self.policy_context = nn.Sequential(nn.Linear(96 + 48, 64), nn.SiLU())
        self.target_head = nn.Sequential(
            nn.Conv2d(width + 64, width, 1),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(width + 96 + 48, 160),
            nn.SiLU(),
            nn.Linear(160, 1),
        )

    def forward(self, centred_map: torch.Tensor, local_sensor: torch.Tensor, scalars: torch.Tensor):
        spatial = self.spatial_encoder(centred_map)
        local = self.local_encoder(local_sensor)
        scalar = self.scalar_encoder(scalars)
        context = self.policy_context(torch.cat([local, scalar], dim=-1))
        context = context[:, :, None, None].expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        target_logits = self.target_head(torch.cat([spatial, context], dim=1)).flatten(1)
        value = self.value_head(torch.cat([spatial.mean(dim=(-2, -1)), local, scalar], dim=-1)).squeeze(-1)
        return target_logits, value

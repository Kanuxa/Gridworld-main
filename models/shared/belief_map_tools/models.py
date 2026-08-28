"""Neural target-scoring models for the planner hierarchy."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


class FrontierScoreNet(nn.Module):
    """Scores every map cell as a phase-1 visibility target.

    The model predicts target utility only. Valid-target masking and safe route
    construction remain deterministic responsibilities of the planner/A*.
    """

    def __init__(self, map_channels: int = 11, scalar_dim: int = 7, hidden_channels: int = 48):
        super().__init__()
        self.map_channels = int(map_channels)
        self.scalar_dim = int(scalar_dim)
        self.hidden_channels = int(hidden_channels)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.map_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(self.hidden_channels),
            ResidualBlock(self.hidden_channels),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )

    def forward(self, map_channels: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        spatial = self.map_encoder(map_channels)
        scalar_embedding = self.scalar_encoder(scalars).unsqueeze(-1).unsqueeze(-1)
        scalar_embedding = scalar_embedding.expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        return self.head(torch.cat([spatial, scalar_embedding], dim=1)).squeeze(1)


class ExplorationScoreNet(nn.Module):
    """Phase-2 target heatmap plus route-quality auxiliary prediction heads."""

    def __init__(self, map_channels: int = 11, scalar_dim: int = 7, hidden_channels: int = 48):
        super().__init__()
        self.map_channels = int(map_channels)
        self.scalar_dim = int(scalar_dim)
        self.hidden_channels = int(hidden_channels)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.map_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(self.hidden_channels),
            ResidualBlock(self.hidden_channels),
            ResidualBlock(self.hidden_channels),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.target_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 7),
        )

    def forward(self, map_channels: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.map_encoder(map_channels)
        scalar_embedding = self.scalar_encoder(scalars)
        expanded_scalars = scalar_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        target_logits = self.target_head(torch.cat([spatial, expanded_scalars], dim=1)).squeeze(1)
        pooled = spatial.mean(dim=(-2, -1))
        auxiliary = self.auxiliary_head(torch.cat([pooled, scalar_embedding], dim=1))
        return target_logits, auxiliary

"""Two-stream neural actor-critic matching the project-plan observation design."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.model_based.ppo.spatial_target_ppo_with_planner.memory import MAP_CHANNELS


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class NeuralMapActorCritic(nn.Module):
    """CNN encoders for the local 8x5x5 sensor view and persistent map.

    A GRU is deliberately not included in v7's first baseline. The explicit map
    already retains full-episode spatial state; this makes PPO training and the
    subsequent ablation easier to validate. A recurrent layer can be added only
    if the map-only baseline demonstrably needs short-term context.
    """

    def __init__(self, map_channels: int = MAP_CHANNELS, scalar_dim: int = 6, width: int = 64):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 96),
            nn.SiLU(),
        )
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            ResidualBlock(width),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 48), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(width + 96 + 48, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.policy_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

    def forward(self, world_map: torch.Tensor, local_sensor: torch.Tensor, scalars: torch.Tensor):
        map_features = self.map_encoder(world_map).mean(dim=(-2, -1))
        local_features = self.local_encoder(local_sensor)
        scalar_features = self.scalar_encoder(scalars)
        features = self.fusion(torch.cat([map_features, local_features, scalar_features], dim=-1))
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class TargetMapActorCritic(nn.Module):
    """Spatial target selector for the planner-executed coverage hierarchy.

    It outputs a score at every cell of an agent-centred 29x29 frame.  The
    policy head deliberately preserves spatial positions: it does *not* use
    global average pooling before scoring possible frontiers.  A separate
    value head may pool globally because value is a whole-state prediction.
    """

    def __init__(self, map_channels: int = MAP_CHANNELS + 2, scalar_dim: int = 6, width: int = 64):
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
        local_features = self.local_encoder(local_sensor)
        scalar_features = self.scalar_encoder(scalars)
        policy_context = self.policy_context(torch.cat([local_features, scalar_features], dim=-1))
        context_grid = policy_context[:, :, None, None].expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        target_logits = self.target_head(torch.cat([spatial, context_grid], dim=1)).flatten(1)
        map_features = spatial.mean(dim=(-2, -1))
        value = self.value_head(torch.cat([map_features, local_features, scalar_features], dim=-1)).squeeze(-1)
        return target_logits, value

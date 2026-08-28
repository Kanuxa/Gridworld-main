"""Per-episode expert trajectory artifacts for supervised target-score training."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .belief_map import BeliefMap
from .experts import PhasePlan


class Phase1TrajectoryWriter:
    """Collect selected phase-1 expert decisions and save a compact NPZ file.

    Each record is a supervised example for a future `FrontierScoreNet`: the
    belief-map tensor and scalar state are inputs; selected target, route score,
    visibility gain, and next route action are expert labels.
    """

    def __init__(self, seed: int, *, adaptive_channels: bool = False):
        self.seed = int(seed)
        self.adaptive_channels = bool(adaptive_channels)
        self.map_channels: List[np.ndarray] = []
        self.scalars: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.next_actions: List[int] = []
        self.expert_scores: List[float] = []
        self.visibility_gains: List[int] = []
        self.route_costs: List[float] = []
        self.route_turns: List[int] = []
        self.route_lengths: List[int] = []

    def record(self, belief: BeliefMap, plan: PhasePlan) -> None:
        if plan.phase != 1 or not plan.route.actions:
            return
        map_values = belief.export_adaptive_channels(phase=1) if self.adaptive_channels else belief.export_channels(phase=1)
        self.map_channels.append(map_values.copy())
        self.scalars.append(
            np.asarray(
                [
                    belief.position[0] / max(1, belief.grid_size - 1),
                    belief.position[1] / max(1, belief.grid_size - 1),
                    belief.heading / 3.0,
                    belief.health / max(1.0, belief.max_health),
                    belief.energy / max(1e-6, belief.max_energy),
                    belief.seen_fraction,
                    belief.steps / max(1, belief.max_steps),
                ],
                dtype=np.float32,
            )
        )
        self.targets.append(np.asarray(plan.target, dtype=np.int16))
        self.next_actions.append(int(plan.route.actions[0]))
        self.expert_scores.append(float(plan.score))
        self.visibility_gains.append(int(plan.visibility_gain))
        self.route_costs.append(float(plan.route.cost))
        self.route_turns.append(int(plan.route.turns))
        self.route_lengths.append(int(len(plan.route.actions)))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"seed_{self.seed:05d}.npz"
        channel_shape = (0, 16 if self.adaptive_channels else 11, 0, 0)
        maps = np.stack(self.map_channels).astype(np.float32) if self.map_channels else np.empty(channel_shape, dtype=np.float32)
        scalars = np.stack(self.scalars).astype(np.float32) if self.scalars else np.empty((0, 7), dtype=np.float32)
        np.savez_compressed(
            path,
            seed=np.asarray(self.seed, dtype=np.int32),
            adaptive_channels=np.asarray(self.adaptive_channels, dtype=bool),
            map_channels=maps,
            scalars=scalars,
            target=np.asarray(self.targets, dtype=np.int16).reshape(-1, 2),
            next_action=np.asarray(self.next_actions, dtype=np.int8),
            expert_score=np.asarray(self.expert_scores, dtype=np.float32),
            visibility_gain=np.asarray(self.visibility_gains, dtype=np.int16),
            route_cost=np.asarray(self.route_costs, dtype=np.float32),
            route_turns=np.asarray(self.route_turns, dtype=np.int16),
            route_length=np.asarray(self.route_lengths, dtype=np.int16),
        )
        return path


class Phase2TrajectoryWriter:
    """Collect selected phase-2 expert decisions for ExplorationScoreNet.

    Phase 2 uses the same belief-map representation as phase 1, with the phase
    channel set to two. Extra labels describe physical exploration, resource
    value, and survival feasibility of the selected target route.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.map_channels: List[np.ndarray] = []
        self.scalars: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.next_actions: List[int] = []
        self.expert_scores: List[float] = []
        self.visibility_gains: List[int] = []
        self.new_visited_along_route: List[int] = []
        self.route_costs: List[float] = []
        self.route_turns: List[int] = []
        self.route_lengths: List[int] = []
        self.projected_health_after_route: List[float] = []
        self.projected_energy_cost: List[float] = []
        self.projected_health_at_horizon: List[float] = []
        self.survival_feasible: List[bool] = []

    def record(self, belief: BeliefMap, plan: PhasePlan) -> None:
        if plan.phase != 2 or not plan.route.actions:
            return
        self.map_channels.append(belief.export_channels(phase=2).copy())
        self.scalars.append(
            np.asarray(
                [
                    belief.position[0] / max(1, belief.grid_size - 1),
                    belief.position[1] / max(1, belief.grid_size - 1),
                    belief.heading / 3.0,
                    belief.health / max(1.0, belief.max_health),
                    belief.energy / max(1e-6, belief.max_energy),
                    belief.seen_fraction,
                    belief.steps / max(1, belief.max_steps),
                ],
                dtype=np.float32,
            )
        )
        self.targets.append(np.asarray(plan.target, dtype=np.int16))
        self.next_actions.append(int(plan.route.actions[0]))
        self.expert_scores.append(float(plan.score))
        self.visibility_gains.append(int(plan.visibility_gain))
        self.new_visited_along_route.append(int(plan.new_visited_along_route))
        self.route_costs.append(float(plan.route.cost))
        self.route_turns.append(int(plan.route.turns))
        self.route_lengths.append(int(len(plan.route.actions)))
        self.projected_health_after_route.append(float(plan.projected_health_after_route))
        self.projected_energy_cost.append(float(plan.projected_energy_cost))
        self.projected_health_at_horizon.append(float(plan.projected_health_at_horizon))
        self.survival_feasible.append(bool(plan.survival_feasible))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"seed_{self.seed:05d}.npz"
        maps = np.stack(self.map_channels).astype(np.float32) if self.map_channels else np.empty((0, 11, 0, 0), dtype=np.float32)
        scalars = np.stack(self.scalars).astype(np.float32) if self.scalars else np.empty((0, 7), dtype=np.float32)
        np.savez_compressed(
            path,
            seed=np.asarray(self.seed, dtype=np.int32),
            map_channels=maps,
            scalars=scalars,
            target=np.asarray(self.targets, dtype=np.int16).reshape(-1, 2),
            next_action=np.asarray(self.next_actions, dtype=np.int8),
            expert_score=np.asarray(self.expert_scores, dtype=np.float32),
            visibility_gain=np.asarray(self.visibility_gains, dtype=np.int16),
            new_visited_along_route=np.asarray(self.new_visited_along_route, dtype=np.int16),
            route_cost=np.asarray(self.route_costs, dtype=np.float32),
            route_turns=np.asarray(self.route_turns, dtype=np.int16),
            route_length=np.asarray(self.route_lengths, dtype=np.int16),
            projected_health_after_route=np.asarray(self.projected_health_after_route, dtype=np.float32),
            projected_energy_cost=np.asarray(self.projected_energy_cost, dtype=np.float32),
            projected_health_at_horizon=np.asarray(self.projected_health_at_horizon, dtype=np.float32),
            survival_feasible=np.asarray(self.survival_feasible, dtype=bool),
        )
        return path

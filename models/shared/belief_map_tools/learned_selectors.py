"""Safe integration of learned target scorers with deterministic route planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .belief_map import BeliefMap
from .experts import (
    PhasePlan,
    estimate_health_at_horizon,
    estimate_route_survival,
    select_phase1_plan,
    select_phase2_plan,
)
from .models import ExplorationScoreNet, FrontierScoreNet
from .route_planner import HeadingAwareAStar


@dataclass
class LearnedSelectorStats:
    model_decisions: int = 0
    expert_fallbacks: int = 0
    emergency_meat_decisions: int = 0


def _planner_scalars(belief: BeliefMap) -> np.ndarray:
    """Return the scalar feature vector shared by the two target scorers."""
    return np.asarray([
        belief.position[0] / max(1, belief.grid_size - 1),
        belief.position[1] / max(1, belief.grid_size - 1),
        belief.heading / 3.0,
        belief.health / max(1.0, belief.max_health),
        belief.energy / max(1e-6, belief.max_energy),
        belief.seen_fraction,
        belief.steps / max(1, belief.max_steps),
    ], dtype=np.float32)


class LearnedPhase1Selector:
    """Choose a valid high-scoring learned target, then let A* enforce safety."""

    def __init__(self, model: FrontierScoreNet, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device
        self.stats = LearnedSelectorStats()

    def __call__(self, belief: BeliefMap, router: HeadingAwareAStar) -> PhasePlan | None:
        with torch.no_grad():
            map_tensor = torch.from_numpy(belief.export_channels(phase=1)).unsqueeze(0).to(self.device)
            scalar_tensor = torch.from_numpy(_planner_scalars(belief)).unsqueeze(0).to(self.device)
            scores = self.model(map_tensor, scalar_tensor)[0].detach().cpu().numpy()

        valid = belief.phase1_safe.copy()
        for row, col in zip(*np.where(valid)):
            if belief.visibility_gain((int(row), int(col))) <= 0:
                valid[row, col] = False
        ranked = np.argsort(np.where(valid, scores, -np.inf).reshape(-1))[::-1]
        for flat_index in ranked:
            if not np.isfinite(scores.reshape(-1)[flat_index]) or not valid.reshape(-1)[flat_index]:
                break
            target = (int(flat_index // belief.grid_size), int(flat_index % belief.grid_size))
            route = router.plan(belief, target=target, phase=1, allow_fallback=False)
            if route is None or not route.actions:
                continue
            energy_cost, health_after, energy_after = estimate_route_survival(belief, route)
            self.stats.model_decisions += 1
            return PhasePlan(
                phase=1,
                target=target,
                route=route,
                score=float(scores[target]),
                visibility_gain=belief.visibility_gain(target),
                new_visited_along_route=sum(not bool(belief.visited[cell]) for cell in route.cells[1:]),
                projected_health_after_route=health_after,
                projected_energy_cost=energy_cost,
                projected_health_at_horizon=health_after,
                survival_feasible=health_after >= 1.0,
            )

        # This should be rare. The deterministic expert fallback keeps the
        # rollout safe and records where the learned mask had no usable choice.
        self.stats.expert_fallbacks += 1
        return select_phase1_plan(belief, router)


class LearnedPhase2Selector:
    """Safely rank phase-2 targets with a learned model.

    The network never authorises movement: targets must still be known,
    non-hazard cells and must have a normal (non-fallback) A* route.  If its
    ranked shortlist has no route that can be completed with the available
    health, the deterministic phase-2 expert takes over.  Low-health meat
    rescue also remains deterministic because it is a safety requirement,
    rather than an exploration preference.
    """

    def __init__(self, model: ExplorationScoreNet, device: torch.device, candidate_limit: int = 24):
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.model = model.to(device).eval()
        self.device = device
        self.candidate_limit = int(candidate_limit)
        self.stats = LearnedSelectorStats()

    def __call__(self, belief: BeliefMap, router: HeadingAwareAStar) -> PhasePlan | None:
        # Preserve the expert's deliberate survival behaviour.  It evaluates
        # known meat first only when the health budget is already constrained.
        if belief.health / max(1.0, belief.max_health) <= 0.70 and np.any(belief.meat & belief.seen):
            emergency_plan = select_phase2_plan(belief, router)
            if emergency_plan is not None and belief.meat[emergency_plan.target]:
                self.stats.emergency_meat_decisions += 1
                return emergency_plan

        with torch.no_grad():
            map_tensor = torch.from_numpy(belief.export_channels(phase=2)).unsqueeze(0).to(self.device)
            scalar_tensor = torch.from_numpy(_planner_scalars(belief)).unsqueeze(0).to(self.device)
            scores, _ = self.model(map_tensor, scalar_tensor)
            scores = scores[0].detach().cpu().numpy()

        # Match the expert's target universe.  The learned model supplies an
        # ordering only; safe route feasibility remains deterministic.
        valid = belief.seen & ~belief.direct_hazard
        for row, col in zip(*np.where(valid)):
            target = (int(row), int(col))
            if belief.visited[target] and belief.visibility_gain(target) <= 0 and not belief.meat[target]:
                valid[target] = False
        ranked = np.argsort(np.where(valid, scores, -np.inf).reshape(-1))[::-1]
        evaluated = 0
        for flat_index in ranked:
            if evaluated >= self.candidate_limit:
                break
            if not valid.reshape(-1)[flat_index]:
                break
            evaluated += 1
            target = (int(flat_index // belief.grid_size), int(flat_index % belief.grid_size))
            route = router.plan(belief, target=target, phase=2, allow_fallback=False)
            if route is None or not route.actions:
                continue
            energy_cost, health_after, energy_after = estimate_route_survival(belief, route)
            if health_after < 1.0:
                continue
            self.stats.model_decisions += 1
            return PhasePlan(
                phase=2,
                target=target,
                route=route,
                score=float(scores[target]),
                visibility_gain=belief.visibility_gain(target),
                new_visited_along_route=sum(not bool(belief.visited[cell]) for cell in route.cells[1:]),
                projected_health_after_route=health_after,
                projected_energy_cost=energy_cost,
                projected_health_at_horizon=estimate_health_at_horizon(
                    belief,
                    health=health_after,
                    energy=energy_after,
                    steps_after_route=belief.steps + len(route.actions),
                ),
                survival_feasible=health_after >= 1.0,
            )

        self.stats.expert_fallbacks += 1
        return select_phase2_plan(belief, router)

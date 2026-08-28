"""Heading-aware A* planning over the agent-owned belief map."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, DIR_VECTORS

from .belief_map import BeliefMap


Pose = Tuple[int, int, int]


@dataclass(frozen=True)
class RoutePlan:
    target: Tuple[int, int]
    actions: Tuple[int, ...]
    cells: Tuple[Tuple[int, int], ...]
    cost: float
    used_fallback: bool = False

    @property
    def turns(self) -> int:
        return sum(action in (ACTION_LEFT, ACTION_RIGHT) for action in self.actions)


class HeadingAwareAStar:
    """A* with turns included in state and cost."""

    def plan(
        self,
        belief: BeliefMap,
        target: Tuple[int, int],
        phase: int,
        allow_fallback: bool = True,
        *,
        max_staleness: float | None = None,
        staleness_cost: float = 0.0,
        predicted_hazard_risk: np.ndarray | None = None,
        hazard_risk_threshold: float | None = None,
        hazard_risk_cost: float = 0.0,
    ) -> RoutePlan | None:
        """Plan a safe route using only agent-side state.

        The keyword-only freshness/risk controls are opt-in and leave V11's
        historical planner behavior unchanged.  They let an uncertainty-aware
        controller reject stale or high-risk cells in normal search while
        retaining the existing fallback behavior when no safe alternative is
        available.
        """
        if predicted_hazard_risk is not None and predicted_hazard_risk.shape != (belief.grid_size, belief.grid_size):
            raise ValueError("predicted_hazard_risk must match the belief-map shape")
        result = self._search(
            belief,
            target,
            phase=phase,
            fallback=False,
            max_staleness=max_staleness,
            staleness_cost=staleness_cost,
            predicted_hazard_risk=predicted_hazard_risk,
            hazard_risk_threshold=hazard_risk_threshold,
            hazard_risk_cost=hazard_risk_cost,
        )
        if result is not None or not allow_fallback:
            return result
        return self._search(
            belief,
            target,
            phase=phase,
            fallback=True,
            max_staleness=max_staleness,
            staleness_cost=staleness_cost,
            predicted_hazard_risk=predicted_hazard_risk,
            hazard_risk_threshold=hazard_risk_threshold,
            hazard_risk_cost=hazard_risk_cost,
        )

    def _search(
        self,
        belief: BeliefMap,
        target: Tuple[int, int],
        phase: int,
        fallback: bool,
        max_staleness: float | None,
        staleness_cost: float,
        predicted_hazard_risk: np.ndarray | None,
        hazard_risk_threshold: float | None,
        hazard_risk_cost: float,
    ) -> RoutePlan | None:
        start: Pose = (belief.position[0], belief.position[1], belief.heading)
        frontier: List[Tuple[float, float, int, Pose]] = []
        counter = 0
        heapq.heappush(frontier, (self._heuristic(start, target), 0.0, counter, start))
        best_cost: Dict[Pose, float] = {start: 0.0}
        previous: Dict[Pose, Tuple[Pose, int]] = {}

        while frontier:
            _, current_cost, _, pose = heapq.heappop(frontier)
            if current_cost != best_cost.get(pose):
                continue
            row, col, heading = pose
            if (row, col) == target:
                return self._reconstruct(target, pose, previous, current_cost, fallback)

            for action, nxt, action_cost in self._neighbours(
                belief,
                pose,
                phase,
                fallback,
                max_staleness=max_staleness,
                staleness_cost=staleness_cost,
                predicted_hazard_risk=predicted_hazard_risk,
                hazard_risk_threshold=hazard_risk_threshold,
                hazard_risk_cost=hazard_risk_cost,
            ):
                candidate = current_cost + action_cost
                if candidate >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = candidate
                previous[nxt] = (pose, action)
                counter += 1
                estimate = candidate + self._heuristic(nxt, target)
                heapq.heappush(frontier, (estimate, candidate, counter, nxt))
        return None

    @staticmethod
    def _heuristic(pose: Pose, target: Tuple[int, int]) -> float:
        return float(abs(pose[0] - target[0]) + abs(pose[1] - target[1]))

    def _neighbours(
        self,
        belief: BeliefMap,
        pose: Pose,
        phase: int,
        fallback: bool,
        *,
        max_staleness: float | None,
        staleness_cost: float,
        predicted_hazard_risk: np.ndarray | None,
        hazard_risk_threshold: float | None,
        hazard_risk_cost: float,
    ):
        row, col, heading = pose
        yield ACTION_LEFT, (row, col, (heading - 1) % 4), 0.20 if phase == 1 else 0.25
        yield ACTION_RIGHT, (row, col, (heading + 1) % 4), 0.20 if phase == 1 else 0.25

        dr, dc = DIR_VECTORS[heading]
        nxt = (row + dr, col + dc)
        if not belief.in_bounds(nxt):
            return
        next_row, next_col = nxt
        known = bool(belief.seen[next_row, next_col])
        forbidden = bool(belief.phase1_forbidden[next_row, next_col])
        direct_hazard = bool(belief.direct_hazard[next_row, next_col])
        # Observation age builds a full map, so leave the V11/default route
        # path untouched unless a caller explicitly enables a freshness term.
        staleness = 0.0
        if max_staleness is not None or staleness_cost > 0.0:
            staleness = float(belief.observation_age[next_row, next_col])
        predicted_risk = 0.0 if predicted_hazard_risk is None else float(predicted_hazard_risk[next_row, next_col])
        if not fallback and max_staleness is not None and known and staleness > max_staleness:
            return
        if not fallback and hazard_risk_threshold is not None and predicted_risk >= hazard_risk_threshold:
            return
        if phase == 1:
            if (not known or forbidden) and not fallback:
                return
            cost = 1.0 + (0.30 if belief.visited[next_row, next_col] else 0.0)
            if not known:
                cost += 200.0
            if forbidden:
                temperature = float(belief.temperature[next_row, next_col])
                severity = max(0.0, 13.0 - temperature, temperature - 29.0) if known else 1.0
                cost += 1_000.0 + 100.0 * severity
        else:
            if direct_hazard and not fallback:
                return
            if not known and not fallback:
                return
            cost = 1.0 + (0.40 if belief.visited[next_row, next_col] else 0.0)
            if known:
                cost += float(belief.phase2_discomfort_cost[next_row, next_col])
                if belief.flower[next_row, next_col]:
                    cost += 0.20
                if belief.meat[next_row, next_col]:
                    health_norm = belief.health / 10.0
                    energy_norm = belief.energy / 10.0
                    need = max(0.0, 0.60 - health_norm) + max(0.0, 0.40 - energy_norm)
                    cost -= min(0.75, need)
            else:
                cost += 200.0
            if direct_hazard:
                cost += 1_000.0
        cost += max(0.0, float(staleness_cost)) * staleness
        cost += max(0.0, float(hazard_risk_cost)) * max(0.0, predicted_risk)
        yield ACTION_FORWARD, (next_row, next_col, heading), cost

    @staticmethod
    def _reconstruct(
        target: Tuple[int, int], end: Pose, previous: Dict[Pose, Tuple[Pose, int]], cost: float, fallback: bool) -> RoutePlan:
        actions: List[int] = []
        poses = [end]
        current = end
        while current in previous:
            prior, action = previous[current]
            actions.append(action)
            poses.append(prior)
            current = prior
        actions.reverse()
        poses.reverse()
        cells: List[Tuple[int, int]] = [(poses[0][0], poses[0][1])]
        for prior, nxt in zip(poses, poses[1:]):
            if prior[:2] != nxt[:2]:
                cells.append((nxt[0], nxt[1]))
        return RoutePlan(target=target, actions=tuple(actions), cells=tuple(cells), cost=float(cost), used_fallback=fallback)

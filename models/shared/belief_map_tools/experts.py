"""Deterministic phase-specific target selection for v11 expert trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
from gui.current_environment.sensory_grid_env import ACTION_FORWARD

from .belief_map import BeliefMap
from .route_planner import HeadingAwareAStar, RoutePlan

MAX_ROUTE_EVALUATIONS = 24

@dataclass(frozen=True)
class PhasePlan:
    phase: int
    target: Tuple[int, int]
    route: RoutePlan
    score: float
    visibility_gain: int
    new_visited_along_route: int
    projected_health_after_route: float
    projected_energy_cost: float
    projected_health_at_horizon: float
    survival_feasible: bool


def _known_cells(mask: np.ndarray) -> Iterable[Tuple[int, int]]:
    for row, col in zip(*np.where(mask)):
        yield int(row), int(col)


def select_phase1_plan(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    *,
    route_options: dict[str, Any] | None = None,
) -> PhasePlan | None:
    """Select a safe frontier with maximal expected visibility per route cost."""
    candidates = []
    for target in _known_cells(belief.phase1_safe):
        gain = belief.visibility_gain(target)
        if gain > 0:
            candidates.append((target, gain))
    candidates.sort(key=lambda item: (-item[1], _manhattan(belief.position, item[0]), item[0]))
    return _select(belief, router, phase=1, candidates=candidates[:MAX_ROUTE_EVALUATIONS], route_options=route_options)


def select_phase2_plan(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    *,
    route_options: dict[str, Any] | None = None,
) -> PhasePlan | None:
    """Select known non-hazard targets for safe physical exploration."""
    traversable = belief.seen & ~belief.direct_hazard
    candidates = []
    for target in _known_cells(traversable):
        is_unvisited = not bool(belief.visited[target])
        gain = belief.visibility_gain(target)
        if is_unvisited or gain > 0 or bool(belief.meat[target]):
            candidates.append((target, gain))
    def optimistic_value(item: Tuple[Tuple[int, int], int]) -> Tuple[float, int, Tuple[int, int]]:
        target, gain = item
        meat_bonus = 2.0 if belief.meat[target] else 0.0
        unvisited_bonus = 8.0 if not belief.visited[target] else 0.0
        value = unvisited_bonus + 3.0 * gain + meat_bonus - _manhattan(belief.position, target)
        return -value, _manhattan(belief.position, target), target

    candidates.sort(key=optimistic_value)
    shortlist = candidates[:MAX_ROUTE_EVALUATIONS]

    # Survival has priority over coverage.  Once the agent is below 70% health,
    # any known reachable meat is evaluated before ordinary exploration targets.
    # This occurs early enough to pay the route energy cost before a fatigue
    # cascade consumes the remaining health.
    if belief.health / max(1.0, belief.max_health) <= 0.70:
        meat_candidates = [(target, gain) for target, gain in candidates if belief.meat[target]]
        meat_plan = _select(belief, router, phase=2, candidates=meat_candidates, route_options=route_options)
        if meat_plan is not None and meat_plan.projected_health_after_route >= 1.0:
            return meat_plan
    return _select(belief, router, phase=2, candidates=shortlist, route_options=route_options)


def _manhattan(source: Tuple[int, int], target: Tuple[int, int]) -> int:
    return abs(source[0] - target[0]) + abs(source[1] - target[1])


def _select(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    phase: int,
    candidates: Iterable[Tuple[Tuple[int, int], int]],
    route_options: dict[str, Any] | None,
) -> PhasePlan | None:
    # Never mix risky fallback routes with ordinary routes.  First require a
    # completely valid route; only if no such target exists may the caller use
    # the least-risk fallback behaviour.
    options = route_options or {}
    candidates = list(candidates)
    normal = _select_from_candidates(belief, router, phase, candidates, allow_fallback=False, route_options=options)
    if normal is not None:
        return normal
    return _select_from_candidates(belief, router, phase, candidates, allow_fallback=True, route_options=options)


def _select_from_candidates(
    belief: BeliefMap,
    router: HeadingAwareAStar,
    phase: int,
    candidates: Iterable[Tuple[Tuple[int, int], int]],
    allow_fallback: bool,
    route_options: dict[str, Any],
) -> PhasePlan | None:
    best: PhasePlan | None = None
    for target, visibility_gain in candidates:
        route = router.plan(belief, target=target, phase=phase, allow_fallback=allow_fallback, **route_options)
        if route is None or not route.actions:
            continue
        route_cells = route.cells[1:]
        new_visited = sum(not bool(belief.visited[cell]) for cell in route_cells)
        revisit_count = sum(bool(belief.visited[cell]) for cell in route_cells)
        projected_energy_cost, projected_health_after, projected_energy_after = estimate_route_survival(belief, route)
        projected_health_at_horizon = estimate_health_at_horizon(
            belief,
            health=projected_health_after,
            energy=projected_energy_after,
            steps_after_route=belief.steps + len(route.actions),
        )
        if phase == 1:
            score = 10.0 * visibility_gain - route.cost - 0.30 * revisit_count
        else:
            meat_bonus = 0.0
            if belief.meat[target]:
                health_norm = belief.health / 10.0
                energy_norm = belief.energy / 10.0
                meat_bonus = 2.0 * (max(0.0, 0.60 - health_norm) + max(0.0, 0.40 - energy_norm))
            # A route which would exhaust all remaining health is not a viable
            # exploration route.  A large penalty makes safe/meat routes win
            # while still permitting the route as a last-resort fallback.
            # The horizon forecast is logged for diagnostics, but using it as
            # a target penalty is too conservative: it can prevent discovery
            # of remaining meat. Reject only routes that cannot be completed
            # with the current health budget.
            survival_penalty = 0.0 if projected_health_after >= 1.0 else 100.0 * (1.0 - projected_health_after)
            score = 8.0 * new_visited + 3.0 * visibility_gain + meat_bonus - route.cost - survival_penalty
        candidate = PhasePlan(
            phase=phase,
            target=target,
            route=route,
            score=float(score),
            visibility_gain=int(visibility_gain),
            new_visited_along_route=int(new_visited),
            projected_health_after_route=float(projected_health_after),
            projected_energy_cost=float(projected_energy_cost),
            projected_health_at_horizon=float(projected_health_at_horizon),
            survival_feasible=bool(projected_health_at_horizon >= 1.0),
        )
        if best is None or candidate.score > best.score or (candidate.score == best.score and candidate.target < best.target):
            best = candidate
    return best


def estimate_route_survival(belief: BeliefMap, route: RoutePlan) -> Tuple[float, float, float]:
    """Estimate energy and fatigue health loss for a route using known cells.

    The environment consumes a time cost on every action, extra cost for
    forward/turn actions, and up to a thermal extra cost from discomfort.  A
    negative energy balance consumes health and restores one max-energy block.
    Known meat encountered on the route restores two health under the current
    environment rules.
    """
    energy_cost = 0.0
    health = float(belief.health)
    energy = float(belief.energy)
    cell_iter = iter(route.cells[1:])
    for action in route.actions:
        action_energy = belief.time_energy_cost
        if action == ACTION_FORWARD:
            cell = next(cell_iter)
            action_energy += belief.forward_energy_cost
            action_energy += float(belief.phase2_discomfort_cost[cell]) * belief.thermal_extra_energy_max
            if belief.meat[cell]:
                health = min(belief.max_health, health + 2.0)
        else:
            action_energy += belief.turn_energy_cost
        energy_cost += action_energy
        health, energy = _apply_energy_cost(health, energy, action_energy, belief.max_energy)
    return float(energy_cost), float(health), float(energy)


def estimate_health_at_horizon(belief: BeliefMap, health: float, energy: float, steps_after_route: int) -> float:
    """Conservative survival estimate if all remaining actions are low-energy turns."""
    remaining_steps = max(0, int(belief.max_steps) - int(steps_after_route))
    row, col = belief.position
    current_discomfort = float(belief.phase2_discomfort_cost[row, col])
    turn_cost = belief.time_energy_cost + belief.turn_energy_cost + current_discomfort * belief.thermal_extra_energy_max
    for _ in range(remaining_steps):
        health, energy = _apply_energy_cost(health, energy, turn_cost, belief.max_energy)
        if health <= 0:
            return 0.0
    return float(health)


def _apply_energy_cost(health: float, energy: float, cost: float, max_energy: float) -> Tuple[float, float]:
    energy -= cost
    while energy <= 0.0 and health > 0.0:
        health -= 1.0
        energy += max_energy
    return health, energy

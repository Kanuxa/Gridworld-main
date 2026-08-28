"""Deterministic coverage baseline for the sensory Gridworld.

``dumb_run`` is intentionally not a learned policy.  It begins with a simple
coverage route, then updates that route whenever its vision reveals danger.
Every observed fire, ice, or glass tile marks its surrounding 3x3 area as
avoided.  The planner chooses a route with the fewest danger-zone entries and
only then the shortest path length.

It evaluates this behaviour over many normal, hazard-filled randomly seeded
environments.  An episode can still end from energy loss before total coverage;
that is a genuine baseline result, not a route-planning failure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import heapq
from typing import Dict, List, Tuple

import numpy as np

from gui.current_environment.sensory_grid_env import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    DIR_VECTORS,
    EnvConfig,
    OBJ_FIRE,
    OBJ_GLASS,
    OBJ_ICE,
    ObservationSwitches,
    SensoryGridEnv,
)


TRAINER_DISPLAY_NAME = "dumb_run"
MODEL_ARCH = "observation_driven_hazard_avoiding_coverage_sweep"

# Directions use the same encoding as sensory_grid_env_v5: up, right, down,
# left.  The route is independent of the randomly sampled starting direction.
UP, RIGHT, DOWN, LEFT = range(4)


def spiral_directions(grid_size: int) -> List[int]:
    """Return world directions for a no-revisit, centre-out spiral.

    The start cell is the centre.  For a 2m+1 grid the route has exactly
    (2m+1)^2 - 1 forward moves, so start + moves visits every cell once.
    """
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError("dumb_run requires a positive odd grid size.")

    directions: List[int] = []
    for radius in range(1, grid_size // 2 + 1):
        directions.extend([UP] * 1)
        directions.extend([LEFT] * (2 * radius - 1))
        directions.extend([DOWN] * (2 * radius))
        directions.extend([RIGHT] * (2 * radius))
        directions.extend([UP] * (2 * radius))
    return directions


def route_cells(grid_size: int) -> List[Tuple[int, int]]:
    """Materialise the route, primarily for validation and reporting."""
    centre = grid_size // 2
    row, col = centre, centre
    cells = [(row, col)]
    for direction in spiral_directions(grid_size):
        dr, dc = DIR_VECTORS[direction]
        row, col = row + dr, col + dc
        cells.append((row, col))
    return cells


class DumbRunPolicy:
    """Coverage planner which replans after observing a fire, ice, or glass cell.

    It keeps only an agent-side map: positions, visited cells and danger zones
    inferred from the egocentric vision patch.  A shortest-path search assigns
    a cost of one to entering a known danger-zone cell and zero elsewhere, so
    it uses a safe route whenever one exists.  Distance is only the secondary
    objective; therefore unavoidable danger-zone crossings are minimised.
    """

    def __init__(self, grid_size: int):
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError("dumb_run requires a positive odd grid size.")
        self.grid_size = grid_size
        centre = grid_size // 2
        self.position = (centre, centre)
        self.visited = {self.position}
        self.danger_sources: set[Tuple[int, int]] = set()
        self.avoid_cells: set[Tuple[int, int]] = set()
        self.replan_count = 0
        self.danger_zone_entries = 0
        self.forward_moves = 0

    @property
    def complete(self) -> bool:
        return len(self.visited) == self.grid_size * self.grid_size

    def _in_bounds(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    @staticmethod
    def _ego_to_world(position: Tuple[int, int], direction: int, ego_row: int, ego_col: int) -> Tuple[int, int]:
        row, col = position
        if direction == UP:
            return row + ego_row, col + ego_col
        if direction == RIGHT:
            return row + ego_col, col - ego_row
        if direction == DOWN:
            return row - ego_row, col - ego_col
        return row - ego_col, col + ego_row

    def observe(self, observation: Dict[str, object]) -> None:
        """Incorporate newly visible hazards and expand their 3x3 avoid zones."""
        vision = np.asarray(observation["vision"], dtype=np.int32)
        direction = int(observation["direction"])
        half = vision.shape[0] // 2
        added_source = False
        for patch_row in range(vision.shape[0]):
            for patch_col in range(vision.shape[1]):
                if int(vision[patch_row, patch_col]) not in {OBJ_FIRE, OBJ_ICE, OBJ_GLASS}:
                    continue
                source = self._ego_to_world(self.position, direction, patch_row - half, patch_col - half)
                if not self._in_bounds(source) or source in self.danger_sources:
                    continue
                self.danger_sources.add(source)
                added_source = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        candidate = (source[0] + dr, source[1] + dc)
                        if self._in_bounds(candidate):
                            self.avoid_cells.add(candidate)
        if added_source:
            self.replan_count += 1

    def _path_to_best_unvisited(self) -> List[Tuple[int, int]]:
        """Find the lexicographically safest then shortest path to a new cell."""
        start = self.position
        frontier: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        costs: Dict[Tuple[int, int], Tuple[int, int]] = {start: (0, 0)}
        previous: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while frontier:
            danger_cost, steps, position = heapq.heappop(frontier)
            if costs[position] != (danger_cost, steps):
                continue
            for direction in (UP, RIGHT, DOWN, LEFT):
                dr, dc = DIR_VECTORS[direction]
                nxt = (position[0] + dr, position[1] + dc)
                if not self._in_bounds(nxt):
                    continue
                candidate_cost = (danger_cost + int(nxt in self.avoid_cells), steps + 1)
                if nxt not in costs or candidate_cost < costs[nxt]:
                    costs[nxt] = candidate_cost
                    previous[nxt] = position
                    heapq.heappush(frontier, (*candidate_cost, nxt))

        targets = [position for position in costs if position not in self.visited]
        if not targets:
            return []
        # The final tie-break is stable and makes the route reproducible once
        # observations are fixed.
        target = min(targets, key=lambda position: (*costs[position], position))
        path = [target]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def next_action(self, current_direction: int) -> int:
        if self.complete:
            raise StopIteration("The dumb_run route is complete.")
        path = self._path_to_best_unvisited()
        if len(path) < 2:
            raise RuntimeError("No path to an unvisited cell exists.")
        next_position = path[1]
        delta = (next_position[0] - self.position[0], next_position[1] - self.position[1])
        desired = next(direction for direction, vector in DIR_VECTORS.items() if vector == delta)
        turn = (desired - int(current_direction)) % 4
        if turn == 0:
            self.position = next_position
            self.visited.add(next_position)
            self.forward_moves += 1
            if next_position in self.avoid_cells:
                self.danger_zone_entries += 1
            return ACTION_FORWARD
        if turn == 1:
            return ACTION_RIGHT
        # A 180-degree turn is represented by two left turns.  The following
        # call will issue the second one, then move forward.
        return ACTION_LEFT


def action_budget(grid_size: int) -> int:
    """Minimum action budget for the original no-revisit spiral."""
    # There are four direction changes per spiral ring and at most two turns
    # to align from the environment's random initial orientation.
    rings = grid_size // 2
    return len(spiral_directions(grid_size)) + 4 * rings + 2


def build_env(grid_size: int) -> SensoryGridEnv:
    base = EnvConfig(grid_size=grid_size)
    # Preserve the regular object counts, health and energy dynamics.  The
    # default 250-step cap is up to four actions short of even the original
    # no-revisit 15x15 route, so extend it only to fit that minimum route.
    return SensoryGridEnv(replace(base, max_steps=action_budget(grid_size)))


def run(seed: int, grid_size: int) -> dict:
    """Run one normally configured random environment with adaptive avoidance."""
    env = build_env(grid_size)
    switches = ObservationSwitches()
    observation, _ = env.reset(seed=seed)
    policy = DumbRunPolicy(grid_size)
    policy.observe(observation)
    actions = 0

    while not policy.complete and not env.terminated and not env.truncated:
        action = policy.next_action(int(observation["direction"]))
        observation, _, _, _, _ = env.step(action, switches)
        policy.observe(observation)
        actions += 1

    return {
        "coverage": float(env.current_scalars()["coverage"]),
        "actions": actions,
        "forward_moves": policy.forward_moves,
        "unique_route_cells": len(policy.visited),
        "route_cells": grid_size * grid_size,
        "known_danger_sources": len(policy.danger_sources),
        "danger_zone_entries": policy.danger_zone_entries,
        "route_replans": policy.replan_count,
        "terminated": env.terminated,
        "truncated": env.truncated,
        "completed_route": policy.complete,
        "health": env.health,
        "energy": env.energy,
    }


def evaluate(seed_start: int, episodes: int, grid_size: int) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Evaluate the exact same dumb route on consecutive, distinct seeds."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    results = [run(seed_start + offset, grid_size) for offset in range(episodes)]
    numeric_means = {
        f"{name}_mean": float(np.mean([float(result[name]) for result in results]))
        for name in ("coverage", "actions", "health", "energy", "known_danger_sources", "danger_zone_entries", "route_replans")
    }
    numeric_means.update({
        "episodes": float(episodes),
        "route_completed_count": float(sum(result["completed_route"] for result in results)),
        "terminated_count": float(sum(result["terminated"] for result in results)),
        "truncated_count": float(sum(result["truncated"] for result in results)),
        "full_coverage_count": float(sum(result["coverage"] == 1.0 for result in results)),
    })
    return numeric_means, results


def write_results(results: List[Dict[str, object]], seed_start: int, output_path: Path) -> None:
    """Save one row per seeded environment for later comparison with DQN logs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "coverage", "actions", "forward_moves", "unique_route_cells", "route_cells",
              "known_danger_sources", "danger_zone_entries", "route_replans",
              "completed_route", "terminated", "truncated", "health", "energy"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, result in enumerate(results):
            writer.writerow({"seed": seed_start + offset, **result})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic sweep coverage baseline.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=15, help="Positive odd grid size (default: 15).")
    parser.add_argument("--episodes", type=int, default=100, help="Number of differently seeded environments (default: 100).")
    parser.add_argument("--output", type=Path, default=Path("runs/15x15/deterministic_sweep_baseline/evaluation.csv"))
    args = parser.parse_args()
    result, results = evaluate(args.seed, args.episodes, args.grid_size)
    write_results(results, args.seed, args.output)
    print(f"model={TRAINER_DISPLAY_NAME} arch={MODEL_ARCH}")
    print(
        "episodes={episodes:.0f} seeds={seed}-{last_seed} coverage_mean={coverage_mean:.3f} "
        "full_coverage={full_coverage_count:.0f}/{episodes:.0f} "
        "route_completed={route_completed_count:.0f}/{episodes:.0f} "
        "terminated={terminated_count:.0f} truncated={truncated_count:.0f} "
        "actions_mean={actions_mean:.1f} danger_entries_mean={danger_zone_entries_mean:.2f} "
        "replans_mean={route_replans_mean:.1f}".format(
            **result, seed=args.seed, last_seed=args.seed + args.episodes - 1
        )
    )
    print(f"Per-environment results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

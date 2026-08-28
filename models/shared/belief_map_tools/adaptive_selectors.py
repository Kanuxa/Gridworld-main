"""Uncertainty-gated neural target selection for adaptive exploration.

The ensemble predicts target preference, never raw actions.  Heading-aware A*
and the observation-derived hazard map remain the safety boundary, while the
deterministic frontier expert takes over whenever the committee is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import torch

from .belief_map import BeliefMap
from .experts import PhasePlan, estimate_route_survival, select_phase1_plan
from .models import FrontierScoreNet
from .route_planner import HeadingAwareAStar


@dataclass
class EnsembleSelectorStats:
    model_decisions: int = 0
    uncertainty_fallbacks: int = 0
    feasibility_fallbacks: int = 0
    mean_selected_disagreement: float = 0.0


@dataclass(frozen=True)
class EnsembleDecision:
    """Agent-observable details of the selector's most recent invocation."""

    source: str = "not_called"
    valid_candidates: int = 0
    candidates_evaluated: int = 0
    selected_target: Tuple[int, int] | None = None
    selected_mean_score: float | None = None
    selected_priority: float | None = None
    selected_disagreement: float | None = None
    uncertainty_threshold: float = 0.85
    top_candidates: tuple[dict[str, Any], ...] = ()


def planner_scalars(belief: BeliefMap) -> np.ndarray:
    """Scalar features shared with V11 score-net checkpoints."""
    return np.asarray(
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


def load_frontier_model(path: Path, device: torch.device) -> FrontierScoreNet:
    """Load a FrontierScoreNet checkpoint with its self-described dimensions."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "model_kwargs" not in payload:
        raise ValueError(f"{path} is not a FrontierScoreNet checkpoint")
    model = FrontierScoreNet(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


class EnsemblePhase1Selector:
    """Use a calibrated committee for phase-1 targets with safe fallback.

    Per-model logits are standardised across currently valid candidates before
    aggregation, so independently trained members need not share a raw-logit
    scale.  High disagreement causes a deterministic expert decision instead
    of an unsafe neural guess.
    """

    def __init__(
        self,
        models: Sequence[FrontierScoreNet],
        device: torch.device,
        *,
        uncertainty_threshold: float = 0.85,
        uncertainty_bonus: float = 0.10,
        candidate_limit: int = 32,
    ) -> None:
        if not models:
            raise ValueError("At least one ensemble model is required")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.models = [model.to(device).eval() for model in models]
        self.map_channels = int(self.models[0].map_channels)
        if any(int(model.map_channels) != self.map_channels for model in self.models):
            raise ValueError("All ensemble members must use the same map-channel schema")
        if self.map_channels not in (11, 16):
            raise ValueError("Expected an 11-channel V11 or 16-channel adaptive FrontierScoreNet")
        self.device = device
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.uncertainty_bonus = float(uncertainty_bonus)
        self.candidate_limit = int(candidate_limit)
        self.stats = EnsembleSelectorStats()
        self.last_decision = EnsembleDecision(uncertainty_threshold=self.uncertainty_threshold)

    @staticmethod
    def _valid_candidates(belief: BeliefMap) -> np.ndarray:
        valid = belief.phase1_safe.copy()
        for row, col in zip(*np.where(valid)):
            if belief.visibility_gain((int(row), int(col))) <= 0:
                valid[row, col] = False
        return valid

    def _committee_scores(self, belief: BeliefMap, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        map_values = belief.export_channels(phase=1) if self.map_channels == 11 else belief.export_adaptive_channels(phase=1)
        map_tensor = torch.from_numpy(map_values).unsqueeze(0).to(self.device)
        scalar_tensor = torch.from_numpy(planner_scalars(belief)).unsqueeze(0).to(self.device)
        member_scores = []
        with torch.no_grad():
            for model in self.models:
                scores = model(map_tensor, scalar_tensor)[0].detach().cpu().numpy().astype(np.float64)
                candidate_values = scores[valid]
                mean = float(np.mean(candidate_values)) if candidate_values.size else 0.0
                std = max(float(np.std(candidate_values)), 1e-6)
                member_scores.append((scores - mean) / std)
        stacked = np.stack(member_scores, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def __call__(
        self,
        belief: BeliefMap,
        router: HeadingAwareAStar,
        *,
        route_options: dict[str, Any] | None = None,
    ) -> PhasePlan | None:
        options = route_options or {}
        valid = self._valid_candidates(belief)
        valid_candidates = int(np.sum(valid))
        if not np.any(valid):
            self.stats.feasibility_fallbacks += 1
            self.last_decision = EnsembleDecision(
                source="feasibility_fallback_no_valid_candidate",
                valid_candidates=valid_candidates,
                uncertainty_threshold=self.uncertainty_threshold,
            )
            return select_phase1_plan(belief, router, route_options=options)

        mean_score, disagreement = self._committee_scores(belief, valid)
        # Uncertainty can encourage informative targets only when it is below
        # the reliability gate.  It never overrides route feasibility.
        priority = mean_score + self.uncertainty_bonus * disagreement
        ranked = np.argsort(np.where(valid, priority, -np.inf).reshape(-1))[::-1]
        top_candidates = tuple(
            {
                "target": [int(flat_index // belief.grid_size), int(flat_index % belief.grid_size)],
                "mean_score": float(mean_score.flat[flat_index]),
                "priority": float(priority.flat[flat_index]),
                "disagreement": float(disagreement.flat[flat_index]),
                "visibility_gain": int(
                    belief.visibility_gain((int(flat_index // belief.grid_size), int(flat_index % belief.grid_size)))
                ),
            }
            for flat_index in ranked[: min(5, valid_candidates)]
            if valid.reshape(-1)[flat_index]
        )
        evaluated = 0
        for flat_index in ranked:
            if evaluated >= self.candidate_limit or not valid.reshape(-1)[flat_index]:
                break
            evaluated += 1
            target = (int(flat_index // belief.grid_size), int(flat_index % belief.grid_size))
            target_disagreement = float(disagreement[target])
            if target_disagreement > self.uncertainty_threshold:
                self.stats.uncertainty_fallbacks += 1
                self.last_decision = EnsembleDecision(
                    source="uncertainty_fallback",
                    valid_candidates=valid_candidates,
                    candidates_evaluated=evaluated,
                    selected_target=target,
                    selected_mean_score=float(mean_score[target]),
                    selected_priority=float(priority[target]),
                    selected_disagreement=target_disagreement,
                    uncertainty_threshold=self.uncertainty_threshold,
                    top_candidates=top_candidates,
                )
                return select_phase1_plan(belief, router, route_options=options)
            route = router.plan(belief, target=target, phase=1, allow_fallback=False, **options)
            if route is None or not route.actions:
                continue
            energy_cost, health_after, _ = estimate_route_survival(belief, route)
            self.stats.model_decisions += 1
            count = max(1, self.stats.model_decisions)
            self.stats.mean_selected_disagreement += (target_disagreement - self.stats.mean_selected_disagreement) / count
            self.last_decision = EnsembleDecision(
                source="model",
                valid_candidates=valid_candidates,
                candidates_evaluated=evaluated,
                selected_target=target,
                selected_mean_score=float(mean_score[target]),
                selected_priority=float(priority[target]),
                selected_disagreement=target_disagreement,
                uncertainty_threshold=self.uncertainty_threshold,
                top_candidates=top_candidates,
            )
            return PhasePlan(
                phase=1,
                target=target,
                route=route,
                score=float(priority[target]),
                visibility_gain=belief.visibility_gain(target),
                new_visited_along_route=sum(not bool(belief.visited[cell]) for cell in route.cells[1:]),
                projected_health_after_route=health_after,
                projected_energy_cost=energy_cost,
                projected_health_at_horizon=health_after,
                survival_feasible=health_after >= 1.0,
            )

        self.stats.feasibility_fallbacks += 1
        self.last_decision = EnsembleDecision(
            source="feasibility_fallback_no_route",
            valid_candidates=valid_candidates,
            candidates_evaluated=evaluated,
            uncertainty_threshold=self.uncertainty_threshold,
            top_candidates=top_candidates,
        )
        return select_phase1_plan(belief, router, route_options=options)

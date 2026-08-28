"""Compare learned phase-1 target selection with the deterministic expert."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models.shared.belief_map_tools.learned_selectors import LearnedPhase1Selector
from models.shared.belief_map_tools.models import FrontierScoreNet
from models.non_model_based.two_phase_belief_map_planner.run import EpisodeResult, run_episode
from models.model_based.learned_frontier_ranker.train_frontier_scorer import choose_device


def load_model(path: Path, device: torch.device) -> FrontierScoreNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FrontierScoreNet(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def test_seeds(split_path: Path) -> List[int]:
    with split_path.open(encoding="utf-8") as handle:
        names = json.load(handle)["test"]
    return [int(Path(name).stem.split("_")[-1]) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FrontierScoreNet phase-1 rollouts against the expert planner.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/15x15/learned_frontier_ranker/frontier_scorer/best_model.pt"))
    parser.add_argument("--seed-split", type=Path, default=Path("runs/15x15/learned_frontier_ranker/frontier_scorer/seed_split.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/15x15/learned_frontier_ranker/frontier_scorer/rollout_comparison.csv"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of held-out seeds for a quick smoke test.")
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    seeds = test_seeds(args.seed_split)
    if args.limit is not None:
        seeds = seeds[: args.limit]
    if not seeds:
        raise ValueError("No held-out test seeds selected.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    result_fields = list(EpisodeResult.__dataclass_fields__)
    fieldnames = ["seed", "model_phase1_decisions", "expert_fallbacks"] + [f"learned_{name}" for name in result_fields] + [f"expert_{name}" for name in result_fields]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        for index, seed in enumerate(seeds, start=1):
            selector = LearnedPhase1Selector(model, device)
            learned = run_episode(seed, phase1_selector=selector)
            expert = run_episode(seed)
            learned_row = {f"learned_{name}": value for name, value in asdict(learned).items()}
            expert_row = {f"expert_{name}": value for name, value in asdict(expert).items()}
            row: Dict[str, object] = {
                "seed": seed,
                "model_phase1_decisions": selector.stats.model_decisions,
                "expert_fallbacks": selector.stats.expert_fallbacks,
                **learned_row,
                **expert_row,
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                "episode={:02d}/{:02d} seed={} learned_seen={:.3f} expert_seen={:.3f} "
                "learned_p1_actions={} expert_p1_actions={} learned_safe={} fallbacks={}".format(
                    index,
                    len(seeds),
                    seed,
                    learned.seen_fraction,
                    expert.seen_fraction,
                    learned.phase1_actions,
                    expert.phase1_actions,
                    learned.phase1_forbidden_entries == 0 and learned.direct_hazard_entries == 0,
                    selector.stats.expert_fallbacks,
                ),
                flush=True,
            )
    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in rows]))
    print(
        "learned_seen_mean={:.3f} expert_seen_mean={:.3f} learned_p1_actions_mean={:.1f} "
        "expert_p1_actions_mean={:.1f} learned_coverage_mean={:.3f} expert_coverage_mean={:.3f} "
        "learned_fallbacks={}".format(
            mean("learned_seen_fraction"), mean("expert_seen_fraction"),
            mean("learned_phase1_actions"), mean("expert_phase1_actions"),
            mean("learned_coverage"), mean("expert_coverage"),
            sum(int(row["expert_fallbacks"]) for row in rows),
        )
    )
    print(f"Comparison saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

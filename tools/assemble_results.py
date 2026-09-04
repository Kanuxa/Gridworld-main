#!/usr/bin/env python3
"""Build a curated, searchable archive of evaluations and metrics from runs/."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "results"
ARTIFACTS_ROOT = RESULTS_ROOT / "artifacts"

TEXT_SUFFIXES = {".csv", ".txt", ".md", ".tsv"}
JSON_BASENAMES = {
    "summary.json",
    "comparison_summary.json",
    "test_metrics.json",
    "train_config.json",
    "seed_split.json",
    "ensemble_report.json",
    "failure_analysis.json",
    "run_config.json",
}


def is_result_artifact(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in JSON_BASENAMES or path.name.endswith(".config.json")


def artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if "comparison" in name:
        return "comparison"
    if "evaluation" in name or name == "evaluation.csv":
        return "evaluation"
    if "metric" in name or name == "summary.json":
        return "metric"
    if "analysis" in name or name == "report.md" or name == "failure_cases.csv":
        return "analysis"
    if "config" in name or name in {"seed_split.json", "manifest.tsv", "sha256sums"}:
        return "configuration"
    if "training" in name or name == "episode_metrics.csv":
        return "training_metric"
    return "supporting_result"


def approach_columns(relative_path: Path) -> tuple[str, str, str]:
    parts = relative_path.parts
    map_size = parts[0] if parts else "unknown"
    family = parts[1] if len(parts) > 1 else "unknown"
    if family == "model_based" and len(parts) > 2:
        approach = "/".join(parts[2:-1]) or parts[2]
    elif family in {"non_model_based", "benchmarks"} and len(parts) > 2:
        approach = "/".join(parts[2:-1]) or parts[2]
    else:
        approach = "/".join(parts[1:-1]) or "unknown"
    return map_size, family, approach


def finite_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return str(value)


def metric_rows(relative_path: Path, payload: dict[str, Any]) -> Iterable[dict[str, str]]:
    map_size, family, approach = approach_columns(relative_path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return []
    candidates: list[tuple[str, dict[str, Any]]] = []
    if any(key in metrics for key in ("coverage", "coverage_p10", "steps")):
        candidates.append(("aggregate", metrics))
    for name in ("best_evaluation", "final_evaluation", "final_training_episode", "last_100_training_episodes_mean"):
        value = metrics.get(name)
        if isinstance(value, dict):
            candidates.append((name, value))
    rows: list[dict[str, str]] = []
    for label, metric in candidates:
        rows.append(
            {
                "source_summary": relative_path.as_posix(),
                "map_size": map_size,
                "decision_family": family,
                "approach": approach,
                "metric_set": label,
                "episode": finite_number(metric.get("episode")),
                "coverage": finite_number(metric.get("coverage")),
                "coverage_p10": finite_number(metric.get("coverage_p10")),
                "steps": finite_number(metric.get("steps")),
                "forward_actions": finite_number(metric.get("forward_actions")),
                "turn_actions": finite_number(metric.get("turn_actions")),
                "repeat_forwards": finite_number(metric.get("repeat_forwards")),
                "unique_per_forward": finite_number(metric.get("unique_per_forward")),
                "health_loss": finite_number(metric.get("health_loss")),
                "survived": finite_number(metric.get("survived")),
                "terminated": finite_number(metric.get("terminated")),
            }
        )
    return rows


def main() -> None:
    if not RUNS_ROOT.is_dir():
        raise FileNotFoundError(f"Missing run archive: {RUNS_ROOT}")
    if ARTIFACTS_ROOT.exists():
        shutil.rmtree(ARTIFACTS_ROOT)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    catalog_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for source in sorted(path for path in RUNS_ROOT.rglob("*") if path.is_file() and is_result_artifact(path)):
        relative_path = source.relative_to(RUNS_ROOT)
        destination = ARTIFACTS_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        map_size, family, approach = approach_columns(relative_path)
        catalog_rows.append(
            {
                "result_file": relative_path.as_posix(),
                "map_size": map_size,
                "decision_family": family,
                "approach": approach,
                "artifact_kind": artifact_kind(source),
                "bytes": str(source.stat().st_size),
            }
        )
        if source.name == "summary.json":
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                summary_rows.extend(metric_rows(relative_path, payload))

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULTS_ROOT / "results_catalog.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(catalog_rows[0]) if catalog_rows else ["result_file"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(catalog_rows)
    metric_fields = [
        "source_summary", "map_size", "decision_family", "approach", "metric_set", "episode", "coverage",
        "coverage_p10", "steps", "forward_actions", "turn_actions", "repeat_forwards", "unique_per_forward",
        "health_loss", "survived", "terminated",
    ]
    with (RESULTS_ROOT / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Copied {len(catalog_rows)} result artifacts into {ARTIFACTS_ROOT}")
    print(f"Indexed {len(summary_rows)} aggregate metric rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose V12 exploration runs from result CSVs and observable JSONL traces.

The runner writes one ``seed_*.jsonl.gz`` file per episode.  This tool is
purposefully tolerant of partially populated traces: result CSV values are
used when available and optional trace fields are reported as unavailable
instead of causing an analysis run to fail.

Typical use::

    python analyze_exploration_traces_v12.py \
      --results runs/15x15/learned_frontier_ranker/standard_ensemble.csv \
      --trace-dir runs/15x15/learned_frontier_ranker/standard_ensemble_traces \
      --output-dir runs/15x15/learned_frontier_ranker/standard_ensemble_analysis \
      --baseline-results runs/15x15/learned_frontier_ranker/standard_expert.csv

Outputs:

* ``failure_analysis.json`` – complete machine-readable aggregate and paired
  baseline deltas.
* ``failure_cases.csv`` – episodes worth inspecting first.
* ``report.md`` – compact human-readable diagnosis.

Only the Python standard library and NumPy are required.  The input traces are
agent-observable execution records; this script never needs to inspect a
hidden environment grid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_SCHEMA_VERSION = 1
_MISSING = object()
_HAZARD_LABELS = {"fire", "ice", "glass", "hazard", "lava", "pit", "wall"}


def _canonical_seed(value: Any) -> str | None:
    """Return a stable seed key without losing non-numeric experiment IDs."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _seed_sort_key(seed: str) -> tuple[int, Any]:
    try:
        return (0, int(seed))
    except (TypeError, ValueError):
        return (1, str(seed))


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan", "n/a", "na"}:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(round(number))


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0", "", "none", "null"}:
        return False
    return None


def _normalise_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "n/a", "na"}:
        return None
    return text


def _nested_get(mapping: Any, dotted_key: str) -> Any:
    """Read a literal or dotted key from a possibly nested JSON object."""
    if not isinstance(mapping, Mapping):
        return _MISSING
    if dotted_key in mapping:
        return mapping[dotted_key]
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _first(mapping: Any, *aliases: str) -> Any:
    for alias in aliases:
        value = _nested_get(mapping, alias)
        if value is not _MISSING and value is not None:
            if not isinstance(value, str) or value.strip():
                return value
    return None


def _first_number(mapping: Any, *aliases: str) -> float | None:
    for alias in aliases:
        value = _to_float(_nested_get(mapping, alias))
        if value is not None:
            return value
    return None


def _first_bool(mapping: Any, *aliases: str) -> bool | None:
    for alias in aliases:
        value = _to_bool(_nested_get(mapping, alias))
        if value is not None:
            return value
    return None


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def _number(value: float | int | None) -> float | int | None:
    """Convert NumPy scalars and non-finite values to JSON-safe primitives."""
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    if result.is_integer():
        return int(result)
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    numbers = [value for value in (_to_float(item) for item in values) if value is not None]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    array = np.asarray(numbers, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": _number(float(np.mean(array))),
        "median": _number(float(np.median(array))),
        "std": _number(float(np.std(array))),
        "min": _number(float(np.min(array))),
        "max": _number(float(np.max(array))),
        "p10": _number(float(np.percentile(array, 10))),
        "p25": _number(float(np.percentile(array, 25))),
        "p75": _number(float(np.percentile(array, 75))),
        "p90": _number(float(np.percentile(array, 90))),
    }


def _coordinate(value: Any) -> tuple[int, int] | None:
    """Normalise the common serialized target forms to ``(row, col)``."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        row = _first_number(value, "row", "r", "y", "position.row", "cell.row")
        col = _first_number(value, "col", "c", "x", "position.col", "cell.col")
        if row is not None and col is not None:
            return (int(round(row)), int(round(col)))
        for key in ("target", "target_cell", "target_position", "goal", "cell", "position", "coordinates"):
            nested = _nested_get(value, key)
            if nested is not _MISSING:
                result = _coordinate(nested)
                if result is not None:
                    return result
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        row = _to_float(value[0])
        col = _to_float(value[1])
        if row is not None and col is not None:
            return (int(round(row)), int(round(col)))
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?\s*[,;:]\s*[-+]?\d+(?:\.\d+)?", value)
        if match:
            left, right = re.split(r"\s*[,;:]\s*", match.group(0), maxsplit=1)
            row = _to_float(left)
            col = _to_float(right)
            if row is not None and col is not None:
                return (int(round(row)), int(round(col)))
    return None


def _extract_target(plan: Any) -> tuple[int, int] | None:
    if plan is None:
        return None
    direct = _coordinate(plan)
    if direct is not None:
        return direct
    if isinstance(plan, Mapping):
        for alias in (
            "target",
            "selected_target",
            "target_cell",
            "target_position",
            "goal",
            "candidate",
            "route.target",
            "route.goal",
        ):
            value = _nested_get(plan, alias)
            if value is not _MISSING:
                target = _coordinate(value)
                if target is not None:
                    return target
    return None


def _extract_route_actions(plan: Any) -> int | None:
    """Return a planned action count without confusing route *cost* for actions."""
    if not isinstance(plan, Mapping):
        return None
    for alias in (
        "route.actions",
        "route_actions",
        "actions",
        "route.action_sequence",
        "route.action_list",
    ):
        value = _nested_get(plan, alias)
        if isinstance(value, (list, tuple, np.ndarray)):
            return int(len(value))
    for alias in ("route.action_count", "route.length", "route.n_actions", "action_count", "n_actions"):
        number = _first_number(plan, alias)
        if number is not None:
            return max(0, int(round(number)))
    return None


def _extract_plan_record(record: Mapping[str, Any]) -> Any:
    """Find a selected plan even when trace writers nest it under planner."""
    for alias in ("selected_plan", "plan", "planner.selected_plan", "decision.selected_plan"):
        value = _nested_get(record, alias)
        if value is not _MISSING and value is not None:
            return value
    # Some small custom trace writers put target/route directly in the event.
    if any(key in record for key in ("target", "selected_target", "route", "route_actions")):
        return record
    return None


def _extract_expert_plan(record: Mapping[str, Any]) -> Any:
    for alias in ("expert_plan", "planner.expert_plan", "decision.expert_plan", "expert"):
        value = _nested_get(record, alias)
        if value is not _MISSING and value is not None:
            return value
    return None


def _is_forward_action(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        # Current V12 traces serialize actions as {"id": 0, "label":
        # "forward"}; accepting either form also keeps small custom traces
        # convenient to analyse.
        return _is_forward_action(_first(value, "id", "label", "action"))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"forward", "move_forward", "action_forward", "0"}:
            return True
        if text in {"left", "right", "turn_left", "turn_right", "1", "2"}:
            return False
        return None
    number = _to_int(value)
    if number is None:
        return None
    # SensoryGridEnv uses action 0 for forward and 1/2 for turns.
    return number == 0


def _contact_label(record: Mapping[str, Any]) -> str | None:
    value = _first(
        record,
        "contacted_label",
        "contact_label",
        "contact",
        "object_contact",
        "info.contacted_label",
        "info.contact",
        "transition.contacted_label",
    )
    return _normalise_text(value)


def _reason(record: Mapping[str, Any]) -> str | None:
    value = _first(
        record,
        "end_reason",
        "reason",
        "termination_reason",
        "status",
        "info.end_reason",
        "result.end_reason",
    )
    return _normalise_text(value)


def _endpoint_number(record: Mapping[str, Any], name: str) -> float | None:
    aliases: dict[str, tuple[str, ...]] = {
        "energy_before": (
            "energy_before",
            "state_before.energy",
            "before.energy",
            "pre_state.energy",
            "belief_before.energy",
        ),
        "energy_after": (
            "energy_after",
            "energy",
            "state.energy",
            "after.energy",
            "post_state.energy",
            "belief_after.energy",
            "initial_belief.energy",
            "final_belief.energy",
            "result.energy",
        ),
        "health_before": (
            "health_before",
            "state_before.health",
            "before.health",
            "pre_state.health",
            "belief_before.health",
        ),
        "health_after": (
            "health_after",
            "health",
            "state.health",
            "after.health",
            "post_state.health",
            "belief_after.health",
            "initial_belief.health",
            "final_belief.health",
            "result.health",
        ),
        "coverage": (
            "coverage",
            "coverage_fraction",
            "state.coverage",
            "metrics.coverage",
            "result.coverage",
            "belief_after.visited_fraction",
            "final_belief.visited_fraction",
        ),
        "seen_fraction": (
            "seen_fraction",
            "seen",
            "seen_coverage",
            "state.seen_fraction",
            "metrics.seen_fraction",
            "result.seen_fraction",
            "belief_after.seen_fraction",
            "initial_belief.seen_fraction",
            "final_belief.seen_fraction",
        ),
    }
    return _first_number(record, *aliases[name])


def _new_trace_summary(seed: str, path: Path) -> dict[str, Any]:
    return {
        "seed": seed,
        "path": str(path),
        "event_counts": Counter(),
        "invalid_lines": 0,
        "unknown_events": 0,
        "start": {},
        "end": {},
        "step_count": 0,
        "forward_attempts": 0,
        "turn_attempts": 0,
        "forward_moves": 0,
        "forward_move_observations": 0,
        "plan_count": 0,
        "plans_by_phase": Counter(),
        "plan_sources": Counter(),
        "fallback_count": 0,
        "fallback_reasons": Counter(),
        "model_plan_count": 0,
        "expert_plan_count": 0,
        "selected_targets": [],
        "planned_action_count": 0,
        "planned_action_observations": 0,
        "divergence_comparisons": 0,
        "target_divergences": 0,
        "target_manhattan_from_expert": [],
        "route_cost_delta_from_expert": [],
        "route_length_delta_from_expert": [],
        "phase2_start_step": None,
        "phase2_start_event": None,
        "last_phase": None,
        "initial_energy": None,
        "final_energy": None,
        "initial_health": None,
        "final_health": None,
        "last_coverage": None,
        "last_seen_fraction": None,
        "fatigue_health_loss": 0.0,
        "fatigue_observed": False,
        "hazard_entries": 0,
        "direct_hazard_health_loss": 0.0,
        "contacts": Counter(),
        "wall_bumps": 0,
        "time_energy_cost_total": 0.0,
        "forward_energy_cost_total": 0.0,
        "turn_energy_cost_total": 0.0,
        "thermal_energy_cost_total": 0.0,
        "termination": None,
        "truncation": None,
        "end_reason": None,
        "did_move_missing": 0,
    }


def _record_phase(trace: dict[str, Any], record: Mapping[str, Any], event_index: int) -> None:
    phase = _to_int(
        _first(
            record,
            "phase",
            "phase_before_action",
            "planner.phase",
            "selected_plan.phase",
            "decision.phase",
            "result.phase",
        )
    )
    if phase is None:
        return
    trace["last_phase"] = phase
    if phase >= 2 and trace["phase2_start_step"] is None:
        step = _to_int(
            _first(record, "episode_step", "step", "step_index", "action_index", "t", "planner.step", "result.phase2_entry_step")
        )
        trace["phase2_start_step"] = trace["step_count"] if step is None else max(0, step)
        trace["phase2_start_event"] = event_index


def _record_plan(trace: dict[str, Any], record: Mapping[str, Any]) -> None:
    trace["plan_count"] += 1
    phase = _to_int(_first(record, "phase", "planner.phase", "selected_plan.phase", "decision.phase"))
    if phase is not None:
        trace["plans_by_phase"][str(phase)] += 1

    source_value = _first(
        record,
        "source",
        "plan_source",
        "selector_source",
        "selection.source",
        "decision.source",
        "planner.source",
        "selected_plan.source",
    )
    source = (_normalise_text(source_value) or "unknown").lower()
    trace["plan_sources"][source] += 1
    if "model" in source or "neural" in source:
        trace["model_plan_count"] += 1
    if "expert" in source:
        trace["expert_plan_count"] += 1

    fallback_flag = _first_bool(
        record,
        "fallback",
        "used_fallback",
        "is_fallback",
        "selection.fallback",
        "decision.fallback",
        "selected_plan.route.used_fallback",
    )
    fallback_reason = _normalise_text(
        _first(record, "fallback_reason", "selection.fallback_reason", "decision.fallback_reason", "planner.fallback_reason")
    )
    is_fallback = bool(fallback_flag) or fallback_reason is not None or "fallback" in source
    if is_fallback:
        trace["fallback_count"] += 1
        trace["fallback_reasons"][(fallback_reason or source).lower()] += 1

    selected_plan = _extract_plan_record(record)
    expert_plan = _extract_expert_plan(record)
    selected_target = _extract_target(selected_plan)
    expert_target = _extract_target(expert_plan)
    if selected_target is not None:
        trace["selected_targets"].append(selected_target)
    if selected_target is not None and expert_target is not None:
        trace["divergence_comparisons"] += 1
        if selected_target != expert_target:
            trace["target_divergences"] += 1
    comparison = _nested_get(record, "comparison_to_expert")
    if isinstance(comparison, Mapping):
        target_distance = _first_number(comparison, "target_manhattan_from_expert")
        route_cost_delta = _first_number(comparison, "route_cost_delta")
        route_length_delta = _first_number(comparison, "route_length_delta")
        if target_distance is not None:
            trace["target_manhattan_from_expert"].append(target_distance)
        if route_cost_delta is not None:
            trace["route_cost_delta_from_expert"].append(route_cost_delta)
        if route_length_delta is not None:
            trace["route_length_delta_from_expert"].append(route_length_delta)

    planned_actions = _extract_route_actions(selected_plan)
    if planned_actions is not None:
        trace["planned_action_count"] += planned_actions
        trace["planned_action_observations"] += 1


def _record_step(trace: dict[str, Any], record: Mapping[str, Any]) -> None:
    trace["step_count"] += 1
    action = _first(record, "action", "action_name", "executed_action", "transition.action")
    forward = _is_forward_action(action)
    if forward is True:
        trace["forward_attempts"] += 1
        did_move = _first_bool(
            record,
            "did_move",
            "moved",
            "transition.did_move",
            "transition_info.did_move",
            "info.did_move",
        )
        if did_move is None:
            before_position = _coordinate(_first(record, "belief_before.position", "position_before", "state_before.position"))
            after_position = _coordinate(_first(record, "belief_after.position", "position_after", "state_after.position"))
            if before_position is not None and after_position is not None:
                did_move = before_position != after_position
        if did_move is None:
            trace["did_move_missing"] += 1
        else:
            trace["forward_move_observations"] += 1
            trace["forward_moves"] += int(did_move)
    elif forward is False:
        trace["turn_attempts"] += 1

    energy_before = _endpoint_number(record, "energy_before")
    energy_after = _endpoint_number(record, "energy_after")
    health_before = _endpoint_number(record, "health_before")
    health_after = _endpoint_number(record, "health_after")
    if trace["initial_energy"] is None:
        trace["initial_energy"] = energy_before if energy_before is not None else energy_after
    if energy_after is not None:
        trace["final_energy"] = energy_after
    if trace["initial_health"] is None:
        trace["initial_health"] = health_before if health_before is not None else health_after
    if health_after is not None:
        trace["final_health"] = health_after
    coverage = _endpoint_number(record, "coverage")
    seen = _endpoint_number(record, "seen_fraction")
    if coverage is not None:
        trace["last_coverage"] = coverage
    if seen is not None:
        trace["last_seen_fraction"] = seen

    explicit_fatigue = _first_number(
        record,
        "fatigue_health_loss",
        "fatigue_damage",
        "fatigue_loss",
        "info.fatigue_health_loss",
        "transition_info.fatigue_health_loss",
        "transition.fatigue_health_loss",
    )
    if explicit_fatigue is not None and explicit_fatigue > 0:
        trace["fatigue_health_loss"] += explicit_fatigue
        trace["fatigue_observed"] = True
    elif health_before is not None and health_after is not None and health_after < health_before:
        contact = (_contact_label(record) or "").lower()
        # A health loss not attributed to a direct known hazard is normally
        # fatigue in the provided environment.  Keep this as a diagnostic
        # estimate, never as a claim that hidden state was inspected.
        if contact not in _HAZARD_LABELS:
            trace["fatigue_health_loss"] += health_before - health_after
            trace["fatigue_observed"] = True

    contact_label = _contact_label(record)
    if contact_label is not None:
        trace["contacts"][contact_label] += 1
    direct_hazard = _first_bool(
        record,
        "direct_hazard_entry",
        "direct_hazard_at_position",
        "hazard_entry",
        "entered_hazard",
        "hit_hazard",
        "info.direct_hazard_entry",
        "transition_info.direct_hazard_entry",
        "transition.direct_hazard_entry",
    )
    contact_is_hazard = contact_label is not None and contact_label.lower() in _HAZARD_LABELS
    if direct_hazard is True or contact_is_hazard:
        trace["hazard_entries"] += 1
        if health_before is not None and health_after is not None and health_before > health_after:
            trace["direct_hazard_health_loss"] += health_before - health_after

    trace["wall_bumps"] += int(
        _first_bool(record, "wall_bump", "bumped_wall", "transition_info.wall_bump", "info.wall_bump") is True
    )
    trace["time_energy_cost_total"] += _first_number(
        record,
        "time_base_cost",
        "transition_info.time_base_cost",
        "info.time_base_cost",
    ) or 0.0
    trace["forward_energy_cost_total"] += _first_number(
        record,
        "forward_extra_cost",
        "transition_info.forward_extra_cost",
        "info.forward_extra_cost",
    ) or 0.0
    trace["turn_energy_cost_total"] += _first_number(
        record,
        "turn_extra_cost",
        "transition_info.turn_extra_cost",
        "info.turn_extra_cost",
    ) or 0.0
    trace["thermal_energy_cost_total"] += _first_number(
        record,
        "thermal_extra_this_tick",
        "transition_info.thermal_extra_this_tick",
        "info.thermal_extra_this_tick",
    ) or 0.0


def _record_endpoint(trace: dict[str, Any], record: Mapping[str, Any], *, is_start: bool) -> None:
    store = trace["start"] if is_start else trace["end"]
    store.update({str(key): value for key, value in record.items() if key not in {"event", "schema_version"}})
    energy = _endpoint_number(record, "energy_after")
    health = _endpoint_number(record, "health_after")
    coverage = _endpoint_number(record, "coverage")
    seen = _endpoint_number(record, "seen_fraction")
    if is_start:
        if energy is not None:
            trace["initial_energy"] = energy
        if health is not None:
            trace["initial_health"] = health
    else:
        if energy is not None:
            trace["final_energy"] = energy
        if health is not None:
            trace["final_health"] = health
        termination = _first_bool(record, "terminated", "termination", "info.terminated", "result.terminated")
        truncation = _first_bool(record, "truncated", "truncation", "info.truncated", "result.truncated")
        if termination is not None:
            trace["termination"] = termination
        if truncation is not None:
            trace["truncation"] = truncation
        end_reason = _reason(record)
        if end_reason is not None:
            trace["end_reason"] = end_reason
    if coverage is not None:
        trace["last_coverage"] = coverage
    if seen is not None:
        trace["last_seen_fraction"] = seen


def _read_trace(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    stem = path.name
    match = re.match(r"seed_(.+?)\.jsonl(?:\.gz)?$", stem)
    seed = _canonical_seed(match.group(1) if match else stem)
    if seed is None:
        return None, [f"Could not infer a seed from trace filename: {path}"]
    trace = _new_trace_summary(seed, path)
    warnings: list[str] = []
    event_index = 0
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    trace["invalid_lines"] += 1
                    continue
                if not isinstance(record, Mapping):
                    trace["invalid_lines"] += 1
                    continue
                event_index += 1
                event = (_normalise_text(record.get("event")) or "unknown").lower()
                trace["event_counts"][event] += 1
                _record_phase(trace, record, event_index)
                if event == "episode_start":
                    _record_endpoint(trace, record, is_start=True)
                elif event == "plan":
                    _record_plan(trace, record)
                elif event == "step":
                    _record_step(trace, record)
                elif event == "episode_end":
                    _record_endpoint(trace, record, is_start=False)
                else:
                    trace["unknown_events"] += 1
    except (OSError, UnicodeError) as exc:
        return None, [f"Could not read trace {path}: {exc}"]

    start_seed = _canonical_seed(_first(trace["start"], "seed"))
    if start_seed is not None and start_seed != seed:
        warnings.append(f"Trace filename seed {seed} differs from episode_start seed {start_seed} in {path.name}")
    if trace["invalid_lines"]:
        warnings.append(f"Ignored {trace['invalid_lines']} invalid JSONL line(s) in {path.name}")
    return trace, warnings


def _read_results_csv(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return rows, [f"Results CSV has no header: {path}"]
            if "seed" not in reader.fieldnames:
                warnings.append(f"Results CSV has no 'seed' column: {path}")
            for index, row in enumerate(reader, start=2):
                seed = _canonical_seed(row.get("seed"))
                if seed is None:
                    warnings.append(f"Skipped results row {index} without a seed in {path.name}")
                    continue
                if seed in rows:
                    warnings.append(f"Duplicate seed {seed} in {path.name}; retaining the last row")
                rows[seed] = {str(key): "" if value is None else str(value) for key, value in row.items()}
    except (OSError, UnicodeError, csv.Error) as exc:
        return rows, [f"Could not read results CSV {path}: {exc}"]
    return rows, warnings


def _row_number(row: Mapping[str, Any] | None, *aliases: str) -> float | None:
    if row is None:
        return None
    return _first_number(row, *aliases)


def _row_bool(row: Mapping[str, Any] | None, *aliases: str) -> bool | None:
    if row is None:
        return None
    return _first_bool(row, *aliases)


def _row_text(row: Mapping[str, Any] | None, *aliases: str) -> str | None:
    if row is None:
        return None
    return _normalise_text(_first(row, *aliases))


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _episode_status(terminated: bool | None, truncated: bool | None, end_reason: str | None) -> str:
    if terminated is True:
        return "terminated"
    if truncated is True:
        return "truncated"
    if end_reason:
        return "ended"
    return "unknown"


def _episode_case(seed: str, row: Mapping[str, Any] | None, trace: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge result and trace observations into one episode-level diagnosis."""
    endpoint = trace.get("end", {}) if trace else {}
    coverage = _first_not_none(
        _row_number(row, "coverage", "coverage_fraction", "physical_coverage"),
        _row_number(endpoint, "coverage", "coverage_fraction", "physical_coverage", "result.coverage"),
        trace.get("last_coverage") if trace else None,
    )
    seen = _first_not_none(
        _row_number(row, "seen_fraction", "seen", "seen_coverage", "observed_fraction"),
        _row_number(endpoint, "seen_fraction", "seen", "seen_coverage", "observed_fraction", "result.seen_fraction"),
        trace.get("last_seen_fraction") if trace else None,
    )
    terminated = _first_not_none(
        _row_bool(row, "terminated", "termination"),
        _row_bool(endpoint, "terminated", "termination", "result.terminated"),
        trace.get("termination") if trace else None,
    )
    truncated = _first_not_none(
        _row_bool(row, "truncated", "truncation"),
        _row_bool(endpoint, "truncated", "truncation", "result.truncated"),
        trace.get("truncation") if trace else None,
    )
    end_reason = _first_not_none(
        _row_text(row, "end_reason", "reason", "termination_reason", "status"),
        _row_text(endpoint, "end_reason", "reason", "termination_reason", "status", "result.end_reason"),
        trace.get("end_reason") if trace else None,
    )
    actions = _first_not_none(
        _row_number(row, "actions", "action_count", "steps"),
        _row_number(endpoint, "result.actions", "result.action_count"),
        trace.get("step_count") if trace else None,
    )
    forward_actions = _first_not_none(
        _row_number(row, "forward_actions", "forward_moves", "forward_count"),
        _row_number(endpoint, "result.forward_actions", "result.forward_moves"),
        trace.get("forward_attempts") if trace else None,
    )
    turns = _first_not_none(
        _row_number(row, "turns", "turn_actions", "turn_count"),
        _row_number(endpoint, "result.turns"),
        trace.get("turn_attempts") if trace else None,
    )
    replans = _first_not_none(_row_number(row, "replans", "plan_count"), _row_number(endpoint, "result.replans"))
    if replans is None and trace is not None:
        replans = trace["plan_count"]
    phase2_flag = _first_not_none(
        _row_bool(row, "reached_phase2", "phase2_reached"),
        _row_bool(endpoint, "result.reached_phase2", "result.phase2_reached"),
        (trace.get("last_phase") or 0) >= 2 if trace and trace.get("last_phase") is not None else None,
    )
    initial_energy = _first_not_none(
        _row_number(row, "initial_energy", "start_energy"),
        trace.get("initial_energy") if trace else None,
        _row_number(trace.get("start", {}) if trace else {}, "energy", "initial_energy", "state.energy", "initial_belief.energy"),
    )
    final_energy = _first_not_none(
        _row_number(row, "energy", "final_energy", "end_energy"),
        trace.get("final_energy") if trace else None,
        _row_number(endpoint, "energy", "final_energy", "state.energy", "result.energy"),
    )
    initial_health = _first_not_none(
        _row_number(row, "initial_health", "start_health"),
        trace.get("initial_health") if trace else None,
        _row_number(trace.get("start", {}) if trace else {}, "health", "initial_health", "state.health", "initial_belief.health"),
    )
    final_health = _first_not_none(
        _row_number(row, "health", "final_health", "end_health"),
        trace.get("final_health") if trace else None,
        _row_number(endpoint, "health", "final_health", "state.health", "result.health"),
    )
    fatigue = _first_not_none(
        _row_number(row, "fatigue_health_losses", "fatigue_health_loss", "fatigue_damage"),
        _row_number(endpoint, "result.fatigue_health_losses", "result.fatigue_health_loss"),
        trace.get("fatigue_health_loss") if trace else None,
    )
    hazards = _first_not_none(
        _row_number(row, "direct_hazard_entries", "hazard_entries", "hazard_contacts"),
        _row_number(endpoint, "result.direct_hazard_entries", "result.hazard_entries"),
        trace.get("hazard_entries") if trace else None,
    )
    baseline_coverage = _row_number(baseline, "coverage", "coverage_fraction", "physical_coverage")
    baseline_seen = _row_number(baseline, "seen_fraction", "seen", "seen_coverage", "observed_fraction")
    trace_steps = trace.get("step_count") if trace else None
    planned_actions = trace.get("planned_action_count") if trace else None
    planned_observations = trace.get("planned_action_observations") if trace else 0
    forward_attempts = trace.get("forward_attempts") if trace else None
    forward_move_observations = trace.get("forward_move_observations") if trace else 0
    forward_moves = trace.get("forward_moves") if trace else None
    selected_targets = trace.get("selected_targets", []) if trace else []
    target_switches = sum(
        int(previous != current)
        for previous, current in zip(selected_targets, selected_targets[1:])
    )
    phase2_start_step = _first_not_none(
        _row_number(row, "phase2_entry_step", "phase2_start_step"),
        _row_number(endpoint, "result.phase2_entry_step", "result.phase2_start_step"),
        trace.get("phase2_start_step") if trace else None,
    )
    if phase2_start_step is not None and phase2_start_step < 0:
        phase2_start_step = None
    wall_bumps = _first_not_none(
        _row_number(row, "wall_bumps", "wall_bump_count"),
        _row_number(endpoint, "result.wall_bumps", "result.wall_bump_count"),
        trace.get("wall_bumps") if trace else None,
    )
    time_energy_cost = _first_not_none(
        _row_number(row, "time_energy_cost_total"),
        _row_number(endpoint, "result.time_energy_cost_total"),
        trace.get("time_energy_cost_total") if trace else None,
    )
    forward_energy_cost = _first_not_none(
        _row_number(row, "forward_energy_cost_total"),
        _row_number(endpoint, "result.forward_energy_cost_total"),
        trace.get("forward_energy_cost_total") if trace else None,
    )
    turn_energy_cost = _first_not_none(
        _row_number(row, "turn_energy_cost_total"),
        _row_number(endpoint, "result.turn_energy_cost_total"),
        trace.get("turn_energy_cost_total") if trace else None,
    )
    thermal_energy_cost = _first_not_none(
        _row_number(row, "thermal_energy_cost_total"),
        _row_number(endpoint, "result.thermal_energy_cost_total"),
        trace.get("thermal_energy_cost_total") if trace else None,
    )
    meat_health_restored = _first_not_none(
        _row_number(row, "meat_health_restored"),
        _row_number(endpoint, "result.meat_health_restored"),
    )
    meat_health_wasted = _first_not_none(
        _row_number(row, "meat_health_wasted"),
        _row_number(endpoint, "result.meat_health_wasted"),
    )
    direct_hazard_health_losses = _first_not_none(
        _row_number(row, "direct_hazard_health_losses", "hazard_health_loss"),
        _row_number(endpoint, "result.direct_hazard_health_losses", "result.hazard_health_loss"),
        trace.get("direct_hazard_health_loss") if trace else None,
    )
    start = trace.get("start", {}) if trace else {}

    case: dict[str, Any] = {
        "seed": seed,
        "environment_set": _row_text(row, "environment_set")
        or _row_text(endpoint, "environment_set", "result.environment_set")
        or _row_text(start, "environment_set")
        or "unknown",
        "environment_variant": _row_text(row, "environment_variant", "environment_profile")
        or _row_text(endpoint, "environment_variant", "environment_profile", "result.environment_variant")
        or _row_text(start, "environment_variant", "environment_profile")
        or "unknown",
        "coverage": coverage,
        "seen_fraction": seen,
        "terminated": terminated,
        "truncated": truncated,
        "termination_status": _episode_status(terminated, truncated, end_reason),
        "end_reason": end_reason or "unknown",
        "actions": _to_int(actions),
        "forward_actions": _to_int(forward_actions),
        "turns": _to_int(turns),
        "replans": _to_int(replans),
        "phase2_reached": phase2_flag,
        "phase2_start_step": _to_int(phase2_start_step),
        "trace_step_count": _to_int(trace_steps),
        "plan_count": int(trace.get("plan_count", 0)) if trace else 0,
        "model_plan_count": int(trace.get("model_plan_count", 0)) if trace else 0,
        "expert_plan_count": int(trace.get("expert_plan_count", 0)) if trace else 0,
        "fallback_count": int(trace.get("fallback_count", 0)) if trace else 0,
        "fallback_rate": _ratio(int(trace.get("fallback_count", 0)) if trace else 0, int(trace.get("plan_count", 0)) if trace else 0),
        "fallback_reasons": _counter_dict(trace.get("fallback_reasons", Counter())) if trace else {},
        "plan_sources": _counter_dict(trace.get("plan_sources", Counter())) if trace else {},
        "divergence_comparisons": int(trace.get("divergence_comparisons", 0)) if trace else 0,
        "target_divergences": int(trace.get("target_divergences", 0)) if trace else 0,
        "target_divergence_rate": _ratio(
            int(trace.get("target_divergences", 0)) if trace else 0,
            int(trace.get("divergence_comparisons", 0)) if trace else 0,
        ),
        "planned_action_count": _to_int(planned_actions),
        "planned_action_observations": _to_int(planned_observations),
        "route_planned_to_executed_ratio": _ratio(planned_actions or 0, trace_steps or 0),
        "target_switches": int(target_switches),
        "forward_attempts": _to_int(forward_attempts),
        "forward_moves": _to_int(forward_moves),
        "forward_move_success_rate": _ratio(forward_moves or 0, forward_move_observations or 0),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_delta": initial_energy - final_energy if initial_energy is not None and final_energy is not None else None,
        "initial_health": initial_health,
        "final_health": final_health,
        "health_delta": initial_health - final_health if initial_health is not None and final_health is not None else None,
        "fatigue_health_loss": fatigue,
        "hazard_entries": _to_int(hazards) or 0,
        "direct_hazard_health_loss": direct_hazard_health_losses,
        "contacts": _counter_dict(trace.get("contacts", Counter())) if trace else {},
        "wall_bumps": _to_int(wall_bumps) or 0,
        "time_energy_cost_total": time_energy_cost,
        "forward_energy_cost_total": forward_energy_cost,
        "turn_energy_cost_total": turn_energy_cost,
        "thermal_energy_cost_total": thermal_energy_cost,
        "meat_health_restored": meat_health_restored,
        "meat_health_wasted": meat_health_wasted,
        "coverage_per_action": _ratio(coverage or 0.0, actions or 0),
        "coverage_per_forward_action": _ratio(coverage or 0.0, forward_actions or 0),
        "seen_per_forward_action": _ratio(seen or 0.0, forward_actions or 0),
        "plans_per_100_actions": 100.0 * (trace.get("plan_count", 0) if trace else 0) / actions if actions else None,
        "baseline_coverage": baseline_coverage,
        "coverage_delta": coverage - baseline_coverage if coverage is not None and baseline_coverage is not None else None,
        "baseline_seen_fraction": baseline_seen,
        "seen_delta": seen - baseline_seen if seen is not None and baseline_seen is not None else None,
        "has_trace": trace is not None,
        "trace_invalid_lines": int(trace.get("invalid_lines", 0)) if trace else 0,
    }
    return case


def _failure_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    coverage = [case["coverage"] for case in cases if case["coverage"] is not None]
    seen = [case["seen_fraction"] for case in cases if case["seen_fraction"] is not None]
    coverage_threshold = float(np.percentile(coverage, 20)) if len(coverage) >= 4 else None
    seen_threshold = float(np.percentile(seen, 20)) if len(seen) >= 4 else None
    failures: list[dict[str, Any]] = []
    for case in cases:
        reasons: list[str] = []
        score = 0
        if case["hazard_entries"] > 0:
            reasons.append("direct_hazard_entry")
            score += 10 + case["hazard_entries"]
        if case["termination_status"] == "terminated":
            reasons.append("terminated")
            score += 4
        elif case["termination_status"] == "truncated":
            reasons.append("truncated")
            score += 2
        if case["final_energy"] is not None and case["final_energy"] <= 1e-9:
            reasons.append("energy_depleted")
            score += 3
        if (case["fatigue_health_loss"] or 0.0) > 0.0:
            reasons.append("fatigue_health_loss")
            score += 2
        if case["phase2_reached"] is False:
            reasons.append("phase2_not_reached")
            score += 2
        if coverage_threshold is not None and case["coverage"] is not None and case["coverage"] <= coverage_threshold:
            reasons.append("bottom_coverage_quintile")
            score += 2
        if seen_threshold is not None and case["seen_fraction"] is not None and case["seen_fraction"] <= seen_threshold:
            reasons.append("bottom_seen_quintile")
            score += 1
        if case["fallback_rate"] is not None and case["fallback_rate"] >= 0.50 and case["plan_count"] >= 2:
            reasons.append("fallback_heavy")
            score += 1
        if case["coverage_delta"] is not None and case["coverage_delta"] < 0:
            reasons.append("coverage_below_baseline")
            score += 2
        if (
            case["target_divergences"] > 0
            and case["coverage_delta"] is not None
            and case["coverage_delta"] < 0
        ):
            reasons.append("divergent_targets_with_regression")
            score += 1
        case["failure_score"] = score
        case["failure_reasons"] = reasons
        if reasons:
            failures.append(case)
    failures.sort(
        key=lambda item: (
            -int(item["failure_score"]),
            float(item["coverage"]) if item["coverage"] is not None else float("inf"),
            _seed_sort_key(item["seed"]),
        )
    )
    return failures, {"bottom_coverage_quintile": coverage_threshold, "bottom_seen_quintile": seen_threshold}


def _aggregate(cases: Sequence[Mapping[str, Any]], traces: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter(str(case["termination_status"]) for case in cases)
    reason_counts: Counter[str] = Counter(str(case["end_reason"]) for case in cases)
    plan_sources: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    contacts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    all_plans = all_fallbacks = all_divergence_comparisons = all_divergences = 0
    all_model_plans = all_expert_plans = 0
    all_planned_actions = all_trace_steps = 0
    all_forward_move_observations = all_forward_moves = 0
    target_distances: list[Any] = []
    route_cost_deltas: list[Any] = []
    route_length_deltas: list[Any] = []
    for trace in traces.values():
        plan_sources.update(trace.get("plan_sources", {}))
        fallback_reasons.update(trace.get("fallback_reasons", {}))
        contacts.update(trace.get("contacts", {}))
        event_counts.update(trace.get("event_counts", {}))
        all_plans += int(trace.get("plan_count", 0))
        all_fallbacks += int(trace.get("fallback_count", 0))
        all_model_plans += int(trace.get("model_plan_count", 0))
        all_expert_plans += int(trace.get("expert_plan_count", 0))
        all_divergence_comparisons += int(trace.get("divergence_comparisons", 0))
        all_divergences += int(trace.get("target_divergences", 0))
        all_planned_actions += int(trace.get("planned_action_count", 0))
        all_trace_steps += int(trace.get("step_count", 0))
        all_forward_move_observations += int(trace.get("forward_move_observations", 0))
        all_forward_moves += int(trace.get("forward_moves", 0))
        target_distances.extend(trace.get("target_manhattan_from_expert", []))
        route_cost_deltas.extend(trace.get("route_cost_delta_from_expert", []))
        route_length_deltas.extend(trace.get("route_length_delta_from_expert", []))

    terminated = sum(case["termination_status"] == "terminated" for case in cases)
    truncated = sum(case["termination_status"] == "truncated" for case in cases)
    reached_phase2 = sum(case["phase2_reached"] is True for case in cases)
    fatigue_episodes = sum((case["fatigue_health_loss"] or 0) > 0 for case in cases)
    hazard_episodes = sum(case["hazard_entries"] > 0 for case in cases)
    contacts_episodes = sum(bool(case["contacts"]) for case in cases)
    action_values = [case["actions"] for case in cases if case["actions"] is not None]
    forward_values = [case["forward_actions"] for case in cases if case["forward_actions"] is not None]
    turn_values = [case["turns"] for case in cases if case["turns"] is not None]
    phase2_steps = [case["phase2_start_step"] for case in cases if case["phase2_start_step"] is not None and case["phase2_start_step"] >= 0]

    return {
        "episode_count": len(cases),
        "trace_episode_count": len(traces),
        "coverage": _numeric_summary(case["coverage"] for case in cases),
        "seen_fraction": _numeric_summary(case["seen_fraction"] for case in cases),
        "termination": {
            "status_counts": _counter_dict(status_counts),
            "end_reason_counts": _counter_dict(reason_counts),
            "terminated_rate": _ratio(terminated, len(cases)),
            "truncated_rate": _ratio(truncated, len(cases)),
        },
        "planner": {
            "plan_events": all_plans,
            "plans_per_episode": _ratio(all_plans, len(cases)),
            "plan_sources": _counter_dict(plan_sources),
            "model_plan_events": all_model_plans,
            "expert_plan_events": all_expert_plans,
            "fallback_events": all_fallbacks,
            "fallback_rate": _ratio(all_fallbacks, all_plans),
            "fallback_reasons": _counter_dict(fallback_reasons),
        },
        "model_expert_target_divergence": {
            "comparable_plan_events": all_divergence_comparisons,
            "divergent_plan_events": all_divergences,
            "divergence_rate": _ratio(all_divergences, all_divergence_comparisons),
            "episodes_with_comparison": sum(case["divergence_comparisons"] > 0 for case in cases),
            "episodes_with_divergence": sum(case["target_divergences"] > 0 for case in cases),
            "target_manhattan_from_expert": _numeric_summary(target_distances),
            "route_cost_delta_from_expert": _numeric_summary(route_cost_deltas),
            "route_length_delta_from_expert": _numeric_summary(route_length_deltas),
        },
        "phase_transition": {
            "episodes_reaching_phase2": reached_phase2,
            "phase2_reached_rate": _ratio(reached_phase2, len(cases)),
            "first_phase2_step": _numeric_summary(phase2_steps),
            "phase1_plan_events": sum(int(trace.get("plans_by_phase", {}).get("1", 0)) for trace in traces.values()),
            "phase2_plan_events": sum(int(trace.get("plans_by_phase", {}).get("2", 0)) for trace in traces.values()),
        },
        "resources_and_safety": {
            "initial_energy": _numeric_summary(case["initial_energy"] for case in cases),
            "final_energy": _numeric_summary(case["final_energy"] for case in cases),
            "energy_delta": _numeric_summary(case["energy_delta"] for case in cases),
            "initial_health": _numeric_summary(case["initial_health"] for case in cases),
            "final_health": _numeric_summary(case["final_health"] for case in cases),
            "health_delta": _numeric_summary(case["health_delta"] for case in cases),
            "fatigue_health_loss": _numeric_summary(case["fatigue_health_loss"] for case in cases),
            "fatigue_episode_rate": _ratio(fatigue_episodes, len(cases)),
            "hazard_entries": int(sum(case["hazard_entries"] for case in cases)),
            "hazard_episode_rate": _ratio(hazard_episodes, len(cases)),
            "direct_hazard_health_loss": _numeric_summary(case["direct_hazard_health_loss"] for case in cases),
            "time_energy_cost_total": _numeric_summary(case["time_energy_cost_total"] for case in cases),
            "forward_energy_cost_total": _numeric_summary(case["forward_energy_cost_total"] for case in cases),
            "turn_energy_cost_total": _numeric_summary(case["turn_energy_cost_total"] for case in cases),
            "thermal_energy_cost_total": _numeric_summary(case["thermal_energy_cost_total"] for case in cases),
            "meat_health_restored": _numeric_summary(case["meat_health_restored"] for case in cases),
            "meat_health_wasted": _numeric_summary(case["meat_health_wasted"] for case in cases),
            "contact_counts": _counter_dict(contacts),
            "contact_episode_rate": _ratio(contacts_episodes, len(cases)),
        },
        "route_efficiency": {
            "actions": _numeric_summary(action_values),
            "forward_actions": _numeric_summary(forward_values),
            "turns": _numeric_summary(turn_values),
            "turn_fraction": _ratio(sum(turn_values), sum(action_values)) if action_values else None,
            "coverage_per_action": _numeric_summary(case["coverage_per_action"] for case in cases),
            "coverage_per_forward_action": _numeric_summary(case["coverage_per_forward_action"] for case in cases),
            "seen_per_forward_action": _numeric_summary(case["seen_per_forward_action"] for case in cases),
            "plans_per_100_actions": _numeric_summary(case["plans_per_100_actions"] for case in cases),
            "target_switches": _numeric_summary(case["target_switches"] for case in cases),
            "wall_bumps": _numeric_summary(case["wall_bumps"] for case in cases),
            "planned_action_events": all_planned_actions,
            "executed_trace_steps": all_trace_steps,
            "planned_to_executed_action_ratio": _ratio(all_planned_actions, all_trace_steps),
            "forward_move_success_rate": _ratio(all_forward_moves, all_forward_move_observations),
            "forward_move_observations": all_forward_move_observations,
        },
        "trace_event_counts": _counter_dict(event_counts),
    }


def _baseline_comparison(cases: Sequence[Mapping[str, Any]], baseline_path: Path | None) -> dict[str, Any] | None:
    if baseline_path is None:
        return None
    paired = [
        {
            "seed": case["seed"],
            "coverage": case["coverage"],
            "baseline_coverage": case["baseline_coverage"],
            "coverage_delta": case["coverage_delta"],
            "seen_fraction": case["seen_fraction"],
            "baseline_seen_fraction": case["baseline_seen_fraction"],
            "seen_delta": case["seen_delta"],
        }
        for case in cases
        if case["baseline_coverage"] is not None or case["baseline_seen_fraction"] is not None
    ]
    paired.sort(key=lambda row: _seed_sort_key(row["seed"]))
    coverage_deltas = [row["coverage_delta"] for row in paired if row["coverage_delta"] is not None]
    seen_deltas = [row["seen_delta"] for row in paired if row["seen_delta"] is not None]
    return {
        "baseline_results": str(baseline_path),
        "paired_episode_count": len(paired),
        "coverage_delta": _numeric_summary(coverage_deltas),
        "seen_delta": _numeric_summary(seen_deltas),
        "coverage_improved_count": sum(delta > 0 for delta in coverage_deltas),
        "coverage_regressed_count": sum(delta < 0 for delta in coverage_deltas),
        "seen_improved_count": sum(delta > 0 for delta in seen_deltas),
        "seen_regressed_count": sum(delta < 0 for delta in seen_deltas),
        "paired_by_seed": paired,
    }


def _compact_number(value: Any, digits: int = 3) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _diagnostic_suggestions(aggregate: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> list[str]:
    suggestions: list[str] = []
    planner = aggregate["planner"]
    divergence = aggregate["model_expert_target_divergence"]
    resources = aggregate["resources_and_safety"]
    transition = aggregate["phase_transition"]
    route = aggregate["route_efficiency"]
    fallback_rate = planner.get("fallback_rate")
    if fallback_rate is not None and fallback_rate >= 0.25:
        suggestions.append(
            "The selector falls back frequently. Inspect the listed fallback reasons before increasing model capacity; "
            "the training targets or confidence threshold may be the bottleneck."
        )
    if divergence.get("divergent_plan_events", 0):
        suggestions.append(
            "Use divergent model/expert targets as a focused replay slice. Compare their route cost, visibility gain, "
            "and later coverage rather than treating all target disagreements as errors."
        )
    if resources.get("fatigue_episode_rate") is not None and resources["fatigue_episode_rate"] > 0:
        suggestions.append(
            "Fatigue is observed. Prioritize route-energy margin, meat timing, and turn reduction before pursuing more aggressive frontier targets."
        )
    if resources.get("hazard_entries", 0) > 0:
        suggestions.append(
            "Direct hazard entries occurred. Keep the deterministic safety mask as the hard constraint and inspect those individual traces first."
        )
    if transition.get("phase2_reached_rate") is not None and transition["phase2_reached_rate"] < 0.9:
        suggestions.append(
            "Some episodes do not reach physical-coverage phase. Inspect phase-1 target churn and the timing of the phase transition."
        )
    if route.get("turn_fraction") is not None and route["turn_fraction"] > 0.25:
        suggestions.append(
            "Turn share is high. Target scoring may need an explicit heading-aware efficiency feature or a penalty for route churn."
        )
    if baseline is not None and baseline["coverage_delta"]["mean"] is not None and baseline["coverage_delta"]["mean"] < 0:
        suggestions.append(
            "Mean paired coverage is below baseline. Do not replace the expert globally yet; train/evaluate on the regression seeds and retain uncertainty-gated fallback."
        )
    if not suggestions:
        suggestions.append(
            "No strong automatic failure signature was available. Use the lowest-coverage cases and any target divergences as the next qualitative trace-review set."
        )
    return suggestions


def _render_report(
    results_path: Path,
    trace_dir: Path | None,
    aggregate: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    warnings: Sequence[str],
    top_k: int,
) -> str:
    coverage = aggregate["coverage"]
    seen = aggregate["seen_fraction"]
    termination = aggregate["termination"]
    planner = aggregate["planner"]
    divergence = aggregate["model_expert_target_divergence"]
    transition = aggregate["phase_transition"]
    resources = aggregate["resources_and_safety"]
    route = aggregate["route_efficiency"]
    lines = [
        "# V12 exploration failure analysis",
        "",
        f"- Results: `{results_path}`",
        f"- Trace directory: `{trace_dir if trace_dir is not None else 'not supplied (endpoint-only analysis)'}`",
        f"- Episode records: {aggregate['episode_count']} (traces: {aggregate['trace_episode_count']})",
        "",
        "## Outcome",
        "",
        f"- Coverage: mean **{_compact_number(coverage['mean'])}**, median {_compact_number(coverage['median'])}, "
        f"p10–p90 {_compact_number(coverage['p10'])}–{_compact_number(coverage['p90'])}.",
        f"- Seen fraction: mean **{_compact_number(seen['mean'])}**, median {_compact_number(seen['median'])}.",
        f"- Terminated: {_compact_number(termination['terminated_rate'])}; truncated: {_compact_number(termination['truncated_rate'])}.",
        "",
        "### End reasons",
        "",
    ]
    end_reasons = termination["end_reason_counts"]
    if end_reasons:
        lines.extend(_markdown_table(["Reason", "Episodes"], [[key, value] for key, value in end_reasons.items()]))
    else:
        lines.append("No end reason was logged.")

    lines.extend(
        [
            "",
            "## Planner and model behavior",
            "",
            f"- Plan events: {planner['plan_events']} ({_compact_number(planner['plans_per_episode'])} per episode).",
            f"- Fallbacks: {planner['fallback_events']} ({_compact_number(planner['fallback_rate'])} of plan events).",
            f"- Model/expert comparable targets: {divergence['comparable_plan_events']}; divergent: "
            f"{divergence['divergent_plan_events']} ({_compact_number(divergence['divergence_rate'])}).",
            "",
            "### Plan sources",
            "",
        ]
    )
    if planner["plan_sources"]:
        lines.extend(_markdown_table(["Source", "Plans"], [[key, value] for key, value in planner["plan_sources"].items()]))
    else:
        lines.append("No `plan` records were available.")
    if planner["fallback_reasons"]:
        lines.extend(["", "### Fallback reasons", ""])
        lines.extend(_markdown_table(["Reason", "Plans"], [[key, value] for key, value in planner["fallback_reasons"].items()]))

    lines.extend(
        [
            "",
            "## Phase transition and efficiency",
            "",
            f"- Reached phase 2: {transition['episodes_reaching_phase2']}/{aggregate['episode_count']} "
            f"({_compact_number(transition['phase2_reached_rate'])}).",
            f"- First phase-2 step: mean {_compact_number(transition['first_phase2_step']['mean'])}; "
            f"median {_compact_number(transition['first_phase2_step']['median'])}.",
            f"- Coverage per action: {_compact_number(route['coverage_per_action']['mean'], 5)}; "
            f"coverage per forward action: {_compact_number(route['coverage_per_forward_action']['mean'], 5)}.",
            f"- Turn fraction: {_compact_number(route['turn_fraction'])}; plans per 100 actions: "
            f"{_compact_number(route['plans_per_100_actions']['mean'])}.",
            f"- Planned/executed trace action ratio: {_compact_number(route['planned_to_executed_action_ratio'])}; "
            f"forward move success: {_compact_number(route['forward_move_success_rate'])}.",
            "",
            "## Resources and safety",
            "",
            f"- Final energy: mean {_compact_number(resources['final_energy']['mean'])}; "
            f"mean energy spent {_compact_number(resources['energy_delta']['mean'])}.",
            f"- Fatigue-health loss: mean {_compact_number(resources['fatigue_health_loss']['mean'])}; "
            f"episodes affected {_compact_number(resources['fatigue_episode_rate'])}.",
            f"- Direct hazard entries: {resources['hazard_entries']} across "
            f"{_compact_number(resources['hazard_episode_rate'])} of episodes.",
            "",
        ]
    )
    if resources["contact_counts"]:
        lines.extend(["### Contacts", ""])
        lines.extend(_markdown_table(["Contact", "Count"], [[key, value] for key, value in resources["contact_counts"].items()]))
        lines.append("")

    if baseline is not None:
        lines.extend(
            [
                "## Paired baseline comparison",
                "",
                f"Paired seeds: {baseline['paired_episode_count']}. Mean coverage delta: "
                f"**{_compact_number(baseline['coverage_delta']['mean'])}**; mean seen delta: "
                f"**{_compact_number(baseline['seen_delta']['mean'])}**.",
                f"Coverage improved on {baseline['coverage_improved_count']} paired seeds and regressed on "
                f"{baseline['coverage_regressed_count']}.",
                "",
            ]
        )
        most_negative = sorted(
            (row for row in baseline["paired_by_seed"] if row["coverage_delta"] is not None),
            key=lambda row: (row["coverage_delta"], _seed_sort_key(row["seed"])),
        )[: min(top_k, 10)]
        if most_negative:
            lines.extend(["### Largest coverage regressions", ""])
            lines.extend(
                _markdown_table(
                    ["Seed", "Current", "Baseline", "Delta"],
                    [
                        [row["seed"], _compact_number(row["coverage"]), _compact_number(row["baseline_coverage"]), _compact_number(row["coverage_delta"])]
                        for row in most_negative
                    ],
                )
            )
            lines.append("")

    lines.extend(
        [
            "## Cases to inspect",
            "",
            "The CSV contains all flagged cases. Low-tail thresholds: coverage "
            f"{_compact_number(thresholds['bottom_coverage_quintile'])}, seen "
            f"{_compact_number(thresholds['bottom_seen_quintile'])}.",
            "",
        ]
    )
    if failures:
        rows = []
        for case in failures[:top_k]:
            rows.append(
                [
                    case["seed"],
                    _compact_number(case["coverage"]),
                    _compact_number(case["seen_fraction"]),
                    case["termination_status"],
                    _compact_number(case["coverage_delta"]),
                    ", ".join(case["failure_reasons"]),
                ]
            )
        lines.extend(_markdown_table(["Seed", "Coverage", "Seen", "End", "Δ coverage", "Why inspect"], rows))
    else:
        lines.append("No episodes met the configured failure/low-tail conditions.")

    lines.extend(["", "## Suggested next checks", ""])
    lines.extend(f"- {suggestion}" for suggestion in _diagnostic_suggestions(aggregate, baseline))
    if warnings:
        lines.extend(["", "## Input warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:50])
        if len(warnings) > 50:
            lines.append(f"- … {len(warnings) - 50} more warning(s) in `failure_analysis.json`.")
    lines.append("")
    return "\n".join(lines)


_FAILURE_CASE_COLUMNS = (
    "seed",
    "environment_set",
    "environment_variant",
    "coverage",
    "seen_fraction",
    "termination_status",
    "end_reason",
    "actions",
    "forward_actions",
    "turns",
    "replans",
    "phase2_reached",
    "phase2_start_step",
    "plan_count",
    "model_plan_count",
    "expert_plan_count",
    "fallback_count",
    "fallback_rate",
    "fallback_reasons",
    "divergence_comparisons",
    "target_divergences",
    "target_divergence_rate",
    "planned_action_count",
    "trace_step_count",
    "route_planned_to_executed_ratio",
    "target_switches",
    "forward_move_success_rate",
    "initial_energy",
    "final_energy",
    "energy_delta",
    "initial_health",
    "final_health",
    "health_delta",
    "fatigue_health_loss",
    "hazard_entries",
    "direct_hazard_health_loss",
    "contacts",
    "wall_bumps",
    "time_energy_cost_total",
    "forward_energy_cost_total",
    "turn_energy_cost_total",
    "thermal_energy_cost_total",
    "meat_health_restored",
    "meat_health_wasted",
    "coverage_per_action",
    "coverage_per_forward_action",
    "seen_per_forward_action",
    "plans_per_100_actions",
    "baseline_coverage",
    "coverage_delta",
    "baseline_seen_fraction",
    "seen_delta",
    "failure_score",
    "failure_reasons",
)


def _write_failure_csv(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FAILURE_CASE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            row = dict(case)
            for key in ("fallback_reasons", "contacts", "failure_reasons"):
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            writer.writerow(row)


def analyze(
    results_path: Path,
    trace_dir: Path | None,
    output_dir: Path,
    *,
    baseline_results: Path | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    """Analyze a completed run and write the three standard diagnostic outputs.

    This public function makes the module convenient to use from a notebook or
    a future experiment controller as well as from the CLI.
    """
    results_path = Path(results_path)
    trace_dir = Path(trace_dir) if trace_dir is not None else None
    output_dir = Path(output_dir)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    results, warnings = _read_results_csv(results_path)
    baseline_rows: dict[str, dict[str, str]] = {}
    if baseline_results is not None:
        baseline_results = Path(baseline_results)
        baseline_rows, baseline_warnings = _read_results_csv(baseline_results)
        warnings.extend(baseline_warnings)
    traces: dict[str, dict[str, Any]] = {}
    if trace_dir is None:
        warnings.append("No trace directory supplied; generated endpoint-only analysis.")
    elif not trace_dir.is_dir():
        warnings.append(f"Trace directory does not exist or is not a directory: {trace_dir}")
    else:
        trace_paths = sorted(
            set(trace_dir.glob("seed_*.jsonl.gz")) | set(trace_dir.glob("seed_*.jsonl")),
            key=lambda item: item.name,
        )
        if not trace_paths:
            warnings.append(f"No seed_*.jsonl.gz traces found in {trace_dir}")
        for path in trace_paths:
            trace, trace_warnings = _read_trace(path)
            warnings.extend(trace_warnings)
            if trace is None:
                continue
            seed = trace["seed"]
            if seed in traces:
                warnings.append(f"Duplicate trace seed {seed}; retaining {path.name}")
            traces[seed] = trace

    result_only = sorted(set(results) - set(traces), key=_seed_sort_key)
    trace_only = sorted(set(traces) - set(results), key=_seed_sort_key)
    if result_only:
        warnings.append(f"{len(result_only)} result seed(s) have no trace")
    if trace_only:
        warnings.append(f"{len(trace_only)} trace seed(s) have no results row")
    all_seeds = sorted(set(results) | set(traces), key=_seed_sort_key)
    cases = [_episode_case(seed, results.get(seed), traces.get(seed), baseline_rows.get(seed)) for seed in all_seeds]
    failures, thresholds = _failure_cases(cases)
    aggregate = _aggregate(cases, traces)
    baseline = _baseline_comparison(cases, baseline_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "inputs": {
            "results": str(results_path),
            "trace_dir": str(trace_dir) if trace_dir is not None else None,
            "baseline_results": str(baseline_results) if baseline_results is not None else None,
        },
        "input_coverage": {
            "result_rows": len(results),
            "trace_files": len(traces),
            "joined_episodes": len(set(results) & set(traces)),
            "result_only_seeds": result_only,
            "trace_only_seeds": trace_only,
        },
        "aggregate": aggregate,
        "failure_thresholds": thresholds,
        "failure_case_count": len(failures),
        "baseline_comparison": baseline,
        "warnings": warnings,
    }
    analysis_path = output_dir / "failure_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_failure_csv(output_dir / "failure_cases.csv", failures)
    (output_dir / "report.md").write_text(
        _render_report(results_path, trace_dir, aggregate, failures, thresholds, baseline, warnings, top_k),
        encoding="utf-8",
    )
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", "--episode-results", dest="results", type=Path, required=True, help="Episode-results CSV from run_adaptive_explorer_v12.py")
    parser.add_argument(
        "--trace-dir",
        "--traces",
        dest="trace_dir",
        type=Path,
        default=None,
        help="Optional directory containing seed_*.jsonl.gz traces; omit for endpoint-only analysis.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for failure_analysis.json, failure_cases.csv, and report.md")
    parser.add_argument("--baseline-results", type=Path, default=None, help="Optional same-seed baseline CSV for paired coverage/seen deltas")
    parser.add_argument("--top-k", type=int, default=25, help="Maximum number of cases shown in report.md (default: 25)")
    args = parser.parse_args(argv)
    try:
        analysis = analyze(
            args.results,
            args.trace_dir,
            args.output_dir,
            baseline_results=args.baseline_results,
            top_k=args.top_k,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Analyzed {} episode(s): {} flagged case(s). Outputs: {}".format(
            analysis["aggregate"]["episode_count"],
            analysis["failure_case_count"],
            args.output_dir.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Print a compact diagnosis for one JSON episode trace; no PyTorch required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--events", type=int, default=12, help="How many route events to show")
    args = parser.parse_args()
    data = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    route = data["route"]
    forwards = [event for event in route if event["action"] == "forward"]
    turns = [event for event in route if event["action"].startswith("turn_")]
    repeats = sum(not event["moved_to_new_cell"] for event in forwards)
    energy = sum(event["energy"]["spent"] for event in route)
    actual_thermal = sum(event["energy"]["cost_components"]["thermal"] for event in route)
    planned_energy = sum(float(event.get("planner_expected_action_energy", 0.0)) for event in route)
    planned_thermal = sum(float(event.get("planner_expected_thermal_energy", 0.0)) for event in route)
    contacts = [event for event in route if event["contacted"] != "Empty"]
    bridges = sum(bool(event.get("resource_bridge_override", False)) for event in route)
    escapes = sum(bool(event.get("turn_escape", False)) for event in route)
    conserved_meat = sum(bool(event.get("conserve_forward_meat", False)) for event in route)
    target_events = [event for event in route if event.get("selected_target_flat_index") is not None]
    target_teacher_matches = sum(bool(event.get("target_matched_teacher")) for event in target_events)

    print(f"episode={data['episode']} seed={data['seed']}")
    print(f"start={data['initial_agent']['coordinate']['text']} heading={data['initial_agent']['heading']}")
    print(f"special_cells={data['special_cells']}")
    print(f"summary={data['summary']}")
    print(f"forwards={len(forwards)} turns={len(turns)} repeated_forwards={repeats} total_energy_spent={energy:.3f}")
    print(
        f"planner_expected_energy={planned_energy:.3f} planner_expected_thermal={planned_thermal:.3f} "
        f"actual_thermal={actual_thermal:.3f}"
    )
    print(f"resource_bridges={bridges} turn_escapes={escapes} conserved_meat_moves={conserved_meat}")
    if target_events:
        print(f"target_teacher_matches={target_teacher_matches}/{len(target_events)}")
    print(f"special_contacts={[(event['step'], event['contacted'], event['position_after']['text']) for event in contacts]}")
    print("route_preview:")
    for event in route[:max(0, args.events)]:
        print(
            f"  step={event['step']:3d} {event['action']:10s} -> {event['position_after']['text']} "
            f"heading={event['heading_after']:5s} energy_spent={event['energy']['spent']:.3f}"
        )


if __name__ == "__main__":
    main()

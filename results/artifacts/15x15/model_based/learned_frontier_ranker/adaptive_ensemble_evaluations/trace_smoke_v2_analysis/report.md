# V12 exploration failure analysis

- Results: `experiments/adaptive_v12/trace_smoke_v2.csv`
- Trace directory: `experiments/adaptive_v12/trace_smoke_v2`
- Episode records: 1 (traces: 1)

## Outcome

- Coverage: mean **0.022**, median 0.022, p10–p90 0.022–0.022.
- Seen fraction: mean **0.200**, median 0.200.
- Terminated: 0.000; truncated: 1.000.

### End reasons

| Reason | Episodes |
| --- | --- |
| max_steps | 1 |

## Planner and model behavior

- Plan events: 5 (5.000 per episode).
- Fallbacks: 0 (0.000 of plan events).
- Model/expert comparable targets: 5; divergent: 0 (0.000).

### Plan sources

| Source | Plans |
| --- | --- |
| deterministic_expert | 5 |

## Phase transition and efficiency

- Reached phase 2: 0/1 (0.000).
- First phase-2 step: mean n/a; median n/a.
- Coverage per action: 0.00444; coverage per forward action: 0.00556.
- Turn fraction: 0.200; plans per 100 actions: 100.000.
- Planned/executed trace action ratio: 7.000; forward move success: 1.000.

## Resources and safety

- Final energy: mean 6.329; mean energy spent 3.671.
- Fatigue-health loss: mean 0.000; episodes affected 0.000.
- Direct hazard entries: 0 across 0.000 of episodes.

### Contacts

| Contact | Count |
| --- | --- |
| Empty | 5 |

## Cases to inspect

The CSV contains all flagged cases. Low-tail thresholds: coverage n/a, seen n/a.

| Seed | Coverage | Seen | End | Δ coverage | Why inspect |
| --- | --- | --- | --- | --- | --- |
| 16100 | 0.022 | 0.200 | truncated | n/a | truncated, phase2_not_reached |

## Suggested next checks

- Some episodes do not reach physical-coverage phase. Inspect phase-1 target churn and the timing of the phase transition.

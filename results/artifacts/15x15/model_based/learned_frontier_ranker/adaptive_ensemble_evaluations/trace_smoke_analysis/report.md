# V12 exploration failure analysis

- Results: `experiments/adaptive_v12/trace_smoke.csv`
- Trace directory: `experiments/adaptive_v12/trace_smoke`
- Episode records: 2 (traces: 2)

## Outcome

- Coverage: mean **0.076**, median 0.076, p10–p90 0.076–0.076.
- Seen fraction: mean **0.391**, median 0.391.
- Terminated: 0.000; truncated: 1.000.

### End reasons

| Reason | Episodes |
| --- | --- |
| max_steps | 2 |

## Planner and model behavior

- Plan events: 34 (17.000 per episode).
- Fallbacks: 0 (0.000 of plan events).
- Model/expert comparable targets: 34; divergent: 0 (0.000).

### Plan sources

| Source | Plans |
| --- | --- |
| deterministic_expert | 34 |

## Phase transition and efficiency

- Reached phase 2: 0/2 (0.000).
- First phase-2 step: mean n/a; median n/a.
- Coverage per action: 0.00378; coverage per forward action: 0.00472.
- Turn fraction: 0.200; plans per 100 actions: 85.000.
- Planned/executed trace action ratio: 5.725; forward move success: 1.000.

## Resources and safety

- Final energy: mean 5.759; mean energy spent 4.241.
- Fatigue-health loss: mean 1.000; episodes affected 1.000.
- Direct hazard entries: 0 across 0.000 of episodes.

### Contacts

| Contact | Count |
| --- | --- |
| Empty | 39 |
| Flower | 1 |

## Cases to inspect

The CSV contains all flagged cases. Low-tail thresholds: coverage n/a, seen n/a.

| Seed | Coverage | Seen | End | Δ coverage | Why inspect |
| --- | --- | --- | --- | --- | --- |
| 16000 | 0.076 | 0.373 | truncated | n/a | truncated, fatigue_health_loss, phase2_not_reached |
| 16001 | 0.076 | 0.409 | truncated | n/a | truncated, fatigue_health_loss, phase2_not_reached |

## Suggested next checks

- Use divergent model/expert targets as a focused replay slice. Compare their route cost, visibility gain, and later coverage rather than treating all target disagreements as errors.
- Fatigue is observed. Prioritize route-energy margin, meat timing, and turn reduction before pursuing more aggressive frontier targets.
- Some episodes do not reach physical-coverage phase. Inspect phase-1 target churn and the timing of the phase transition.

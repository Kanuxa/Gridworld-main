# V12 exploration failure analysis

- Results: `experiments/adaptive_v12/standard_expert.csv`
- Trace directory: `not supplied (endpoint-only analysis)`
- Episode records: 47 (traces: 0)

## Outcome

- Coverage: mean **0.605**, median 0.604, p10–p90 0.515–0.692.
- Seen fraction: mean **0.983**, median 0.996.
- Terminated: 1.000; truncated: 0.000.

### End reasons

| Reason | Episodes |
| --- | --- |
| unknown | 47 |

## Planner and model behavior

- Plan events: 0 (0.000 per episode).
- Fallbacks: 0 (n/a of plan events).
- Model/expert comparable targets: 0; divergent: 0 (n/a).

### Plan sources

No `plan` records were available.

## Phase transition and efficiency

- Reached phase 2: 47/47 (1.000).
- First phase-2 step: mean n/a; median n/a.
- Coverage per action: 0.00303; coverage per forward action: 0.00374.
- Turn fraction: 0.188; plans per 100 actions: 0.000.
- Planned/executed trace action ratio: n/a; forward move success: n/a.

## Resources and safety

- Final energy: mean 9.603; mean energy spent n/a.
- Fatigue-health loss: mean 14.957; episodes affected 1.000.
- Direct hazard entries: 0 across 0.000 of episodes.

## Cases to inspect

The CSV contains all flagged cases. Low-tail thresholds: coverage 0.556, seen 0.978.

| Seed | Coverage | Seen | End | Δ coverage | Why inspect |
| --- | --- | --- | --- | --- | --- |
| 10040 | 0.378 | 0.822 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10004 | 0.507 | 0.920 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10033 | 0.507 | 0.942 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10008 | 0.520 | 0.920 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10007 | 0.524 | 0.960 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10010 | 0.547 | 0.978 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10025 | 0.556 | 0.933 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 10042 | 0.489 | 1.000 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 10029 | 0.502 | 0.987 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 10036 | 0.538 | 0.987 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 10027 | 0.560 | 0.973 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 10030 | 0.587 | 0.964 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 10021 | 0.591 | 0.978 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 10022 | 0.636 | 0.978 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 10028 | 0.560 | 0.987 | terminated | n/a | terminated, fatigue_health_loss |
| 10017 | 0.564 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |
| 10002 | 0.569 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |
| 10032 | 0.573 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |
| 10031 | 0.578 | 0.996 | terminated | n/a | terminated, fatigue_health_loss |
| 10001 | 0.582 | 0.987 | terminated | n/a | terminated, fatigue_health_loss |
| 10044 | 0.591 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |
| 10016 | 0.596 | 0.987 | terminated | n/a | terminated, fatigue_health_loss |
| 10012 | 0.600 | 0.987 | terminated | n/a | terminated, fatigue_health_loss |
| 10018 | 0.600 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |
| 10020 | 0.604 | 1.000 | terminated | n/a | terminated, fatigue_health_loss |

## Suggested next checks

- Fatigue is observed. Prioritize route-energy margin, meat timing, and turn reduction before pursuing more aggressive frontier targets.

## Input warnings

- No trace directory supplied; generated endpoint-only analysis.
- 47 result seed(s) have no trace

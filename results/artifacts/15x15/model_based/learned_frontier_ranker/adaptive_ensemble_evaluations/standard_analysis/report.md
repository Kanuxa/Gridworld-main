# V12 exploration failure analysis

- Results: `experiments/adaptive_v12/standard_results.csv`
- Trace directory: `experiments/adaptive_v12/standard_trace`
- Episode records: 100 (traces: 100)

## Outcome

- Coverage: mean **0.600**, median 0.604, p10–p90 0.520–0.680.
- Seen fraction: mean **0.986**, median 0.991.
- Terminated: 1.000; truncated: 0.000.

### End reasons

| Reason | Episodes |
| --- | --- |
| health_zero_fatigue | 100 |

## Planner and model behavior

- Plan events: 8442 (84.420 per episode).
- Fallbacks: 0 (0.000 of plan events).
- Model/expert comparable targets: 5664; divergent: 0 (0.000).

### Plan sources

| Source | Plans |
| --- | --- |
| deterministic_expert | 5664 |
| phase2_expert | 2778 |

## Phase transition and efficiency

- Reached phase 2: 100/100 (1.000).
- First phase-2 step: mean 69.080; median 65.000.
- Coverage per action: 0.00301; coverage per forward action: 0.00373.
- Turn fraction: 0.193; plans per 100 actions: 42.814.
- Planned/executed trace action ratio: 3.642; forward move success: 1.000.

## Resources and safety

- Final energy: mean 9.578; mean energy spent 0.422.
- Fatigue-health loss: mean 14.900; episodes affected 1.000.
- Direct hazard entries: 0 across 0.000 of episodes.

### Contacts

| Contact | Count |
| --- | --- |
| Empty | 19551 |
| Meat | 281 |
| Flower | 109 |

## Cases to inspect

The CSV contains all flagged cases. Low-tail thresholds: coverage 0.555, seen 0.981.

| Seed | Coverage | Seen | End | Δ coverage | Why inspect |
| --- | --- | --- | --- | --- | --- |
| 20075 | 0.360 | 0.964 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20096 | 0.364 | 0.898 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20047 | 0.396 | 0.893 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20034 | 0.427 | 0.898 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20091 | 0.440 | 0.969 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20042 | 0.467 | 0.951 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20020 | 0.484 | 0.973 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20087 | 0.516 | 0.960 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20029 | 0.520 | 0.969 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20054 | 0.524 | 0.964 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20053 | 0.529 | 0.960 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20059 | 0.547 | 0.938 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile, bottom_seen_quintile |
| 20012 | 0.498 | 0.982 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20014 | 0.520 | 0.991 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20050 | 0.533 | 0.991 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20060 | 0.533 | 1.000 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20090 | 0.538 | 0.987 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20038 | 0.542 | 0.982 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20001 | 0.547 | 1.000 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20025 | 0.551 | 0.996 | terminated | n/a | terminated, fatigue_health_loss, bottom_coverage_quintile |
| 20074 | 0.556 | 0.947 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 20040 | 0.564 | 0.969 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 20051 | 0.569 | 0.969 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 20069 | 0.569 | 0.947 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |
| 20030 | 0.582 | 0.973 | terminated | n/a | terminated, fatigue_health_loss, bottom_seen_quintile |

## Suggested next checks

- Fatigue is observed. Prioritize route-energy margin, meat timing, and turn reduction before pursuing more aggressive frontier targets.

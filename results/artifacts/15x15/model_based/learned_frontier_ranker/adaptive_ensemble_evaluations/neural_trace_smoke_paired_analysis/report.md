# V12 exploration failure analysis

- Results: `experiments/adaptive_v12/neural_trace_smoke.csv`
- Trace directory: `experiments/adaptive_v12/neural_trace_smoke`
- Episode records: 1 (traces: 1)

## Outcome

- Coverage: mean **0.076**, median 0.076, p10–p90 0.076–0.076.
- Seen fraction: mean **0.324**, median 0.324.
- Terminated: 0.000; truncated: 1.000.

### End reasons

| Reason | Episodes |
| --- | --- |
| max_steps | 1 |

## Planner and model behavior

- Plan events: 17 (17.000 per episode).
- Fallbacks: 0 (0.000 of plan events).
- Model/expert comparable targets: 17; divergent: 17 (1.000).

### Plan sources

| Source | Plans |
| --- | --- |
| model | 17 |

## Phase transition and efficiency

- Reached phase 2: 0/1 (0.000).
- First phase-2 step: mean n/a; median n/a.
- Coverage per action: 0.00378; coverage per forward action: 0.00472.
- Turn fraction: 0.200; plans per 100 actions: 85.000.
- Planned/executed trace action ratio: 2.400; forward move success: 1.000.

## Resources and safety

- Final energy: mean 5.028; mean energy spent 4.972.
- Fatigue-health loss: mean 1.000; episodes affected 1.000.
- Direct hazard entries: 0 across 0.000 of episodes.

### Contacts

| Contact | Count |
| --- | --- |
| Empty | 19 |
| Meat | 1 |

## Paired baseline comparison

Paired seeds: 1. Mean coverage delta: **-0.004**; mean seen delta: **-0.111**.
Coverage improved on 0 paired seeds and regressed on 1.

### Largest coverage regressions

| Seed | Current | Baseline | Delta |
| --- | --- | --- | --- |
| 16200 | 0.076 | 0.080 | -0.004 |

## Cases to inspect

The CSV contains all flagged cases. Low-tail thresholds: coverage n/a, seen n/a.

| Seed | Coverage | Seen | End | Δ coverage | Why inspect |
| --- | --- | --- | --- | --- | --- |
| 16200 | 0.076 | 0.324 | truncated | -0.004 | truncated, fatigue_health_loss, phase2_not_reached, coverage_below_baseline, divergent_targets_with_regression |

## Suggested next checks

- Use divergent model/expert targets as a focused replay slice. Compare their route cost, visibility gain, and later coverage rather than treating all target disagreements as errors.
- Fatigue is observed. Prioritize route-energy margin, meat timing, and turn reduction before pursuing more aggressive frontier targets.
- Some episodes do not reach physical-coverage phase. Inspect phase-1 target churn and the timing of the phase transition.
- Mean paired coverage is below baseline. Do not replace the expert globally yet; train/evaluate on the regression seeds and retain uncertainty-gated fallback.

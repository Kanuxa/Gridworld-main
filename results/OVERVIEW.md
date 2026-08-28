# Results overview

## Fair 15x15 partial-observation comparison

The core comparison constrains the deployable controller to a 5x5 observation
and an agent-owned reconstructed map. The full-information oracle is reported
separately as an upper bound.

| Controller | Mean coverage | P10 coverage | Evaluation |
| --- | ---: | ---: | --- |
| Partial-observation frontier planner | 71.73% | 67.96% | 50 matched maps |
| Spatial-target PPO with planner executor | 69.22% | 65.51% | Best checkpoint evaluation |
| Planner-residual spatial-target PPO | 70.39% | 65.29% | Best checkpoint, episode 700 |
| Full-information oracle | 78.69% | 76.00% | 50 matched maps; privileged |

The planner is the strongest valid partial-observation controller in this
completed evaluation. The planner-residual PPO has the strongest learned mean,
but does not exceed the planner's tail reliability. The oracle receives the
entire map at reset and must not be included in the fair ranking.

## Core evidence

- [Matched planner versus oracle comparison](artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_by_episode.csv)
- [Matched comparison summary](artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_summary.json)
- [Frontier planner 50-map metrics](artifacts/15x15/non_model_based/partial_observation_frontier_planner/baseline_50/planner_metrics.csv)
- [Spatial-target PPO summary](artifacts/15x15/model_based/spatial_target_ppo_with_planner/summary.json)
- [Planner-residual PPO summary](artifacts/15x15/model_based/planner_residual_target_ppo/summary.json)
- [Full-information oracle summary](artifacts/15x15/benchmarks/full_information_oracle/summary.json)

## Historical results

The direct-action DQN, PPO/RND, and R2D2 training/evaluation logs are retained
in `artifacts/15x15/model_based/archived_direct_action_rl/`. Their historical
run names are preserved to match `MANIFEST.tsv` and `SHA256SUMS`. The approach
catalog maps each original experiment to its descriptive final implementation.

Learned frontier-ranker evaluations, paired failure analyses, and two-phase
planner verification outputs are also present under their matching 15x15
paths. Use `results_catalog.csv` to filter by approach or artifact type.

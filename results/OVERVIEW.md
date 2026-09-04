# Results overview

## Fair 15x15 partial-observation comparison

The core comparison constrains the deployable controller to a 5x5 observation
and an agent-owned reconstructed map. The full-information oracle is reported
separately as an upper bound.

| Controller | Mean coverage | P10 coverage | Evaluation |
| --- | ---: | ---: | --- |
| Partial-observation frontier planner | 71.41% | 67.96% | 50 held-out, matched maps |
| Spatial-target PPO with planner executor | 67.32% | 62.18% | Selected checkpoint, episode 600 |
| Planner-residual spatial-target PPO | 68.80% | 62.67% | Selected checkpoint, episode 700 |
| Full-information oracle | 78.69% | 76.00% | 50 matched maps; privileged |

The planner is the strongest valid partial-observation controller in this
held-out paired evaluation. The learned-minus-planner mean paired coverage
differences are -4.10 percentage points for spatial-target PPO (95% bootstrap
CI [-5.67, -2.67]) and -2.61 for residual PPO ([-3.98, -1.25]). They are
inferences across fresh maps for two saved checkpoints, not across independent
PPO training seeds. The oracle receives the entire map at reset and must not be
included in the fair ranking.

## Paired planner-component ablation

On 50 further matched 15x15 maps (seeds 65,000--65,049), removing resource
recovery from the released planner lowers mean coverage by 10.41 percentage
points (95% bootstrap CI [-12.89, -7.89]); removing route revisit costs lowers
it by 1.48 points ([-2.21, -0.79]). Removing the safe-frontier-forward rule
lowers it by 1.86 points ([-3.23, -0.44]). The thermal-route-cost removal is
inconclusive: -0.18 points ([-1.04, 0.75]). These are component stress tests
within this implementation, not a global parameter search.

## Fixed-condition sensitivity tests

The saved planner-residual PPO checkpoint and unchanged planner were each
evaluated on two further disjoint, 50-map paired 15x15 ranges. In the 15%
visual-object-dropout condition, the residual checkpoint records 64.99% mean
coverage / 55.42% P10 versus 62.69% / 53.33% for the planner: residual minus
planner is +2.29 percentage points (95% bootstrap CI [-0.49, 5.20]; exact sign
test p=0.0789). In the one-restorative-resource condition, the corresponding
values are 55.30% / 53.24% and 56.04% / 52.84%; residual minus planner is
-0.74 points ([-1.64, 0.13]; p=0.0789). Neither interval excludes zero.

These are deliberately limited simulation sensitivity checks: detection
dropout only alters the public visual patch, and resource scarcity only changes
the baseline meat count from three to one. They are not evidence of real-robot
robustness or replication across PPO training runs.

## Evaluation-only persistent-prior diagnostic

On a third, fresh 50-map nominal range, the same residual checkpoint records
68.79% mean coverage / 62.58% P10 with its released persistent planner prior
and 67.66% / 60.44% after its planner target-score weight and tie bonus are
zeroed only at inference. The paired full-prior minus stripped-prior mean is
1.13 percentage points (95% bootstrap CI [-0.41, 2.68]; exact sign test
p=0.392). The interval includes zero, so it does not establish a resolved
inference-time dependency. As the checkpoint was trained with the prior, this
does not estimate the effect of training a no-prior architecture.

## Core evidence

- [Held-out planner versus learned comparison](artifacts/15x15/benchmarks/frontier_planner_vs_learned_heldout/comparison_by_episode.csv)
- [Held-out comparison summary](artifacts/15x15/benchmarks/frontier_planner_vs_learned_heldout/comparison_summary.json)
- [Matched planner versus oracle comparison](artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_by_episode.csv)
- [Matched oracle comparison summary](artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_summary.json)
- [Held-out planner-component outcomes](artifacts/15x15/benchmarks/frontier_planner_component_ablation_heldout/comparison_by_episode.csv)
- [Held-out planner-component summary](artifacts/15x15/benchmarks/frontier_planner_component_ablation_heldout/comparison_summary.json)
- [Object-dropout sensitivity summary](artifacts/15x15/benchmarks/frontier_planner_vs_residual_object_dropout_heldout/comparison_summary.json)
- [Resource-scarcity sensitivity summary](artifacts/15x15/benchmarks/frontier_planner_vs_residual_resource_scarce_heldout/comparison_summary.json)
- [Persistent-prior diagnostic summary](artifacts/15x15/benchmarks/residual_persistent_prior_inference_ablation_heldout/summary.json)
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

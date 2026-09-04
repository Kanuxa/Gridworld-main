# Approach catalogue

This catalogue is the authoritative taxonomy for the final repository. It
separates an approach by the source of its decision rule, not by whether it
uses an internal map. An agent-owned map reconstructed from public observations
is valid for both groups; access to the hidden environment map is not.

## Non-model-based approaches

| Final location | Meaning | Historical source |
| --- | --- | --- |
| `models/non_model_based/deterministic_sweep_baseline/` | A simple deterministic coverage sweep with local hazard avoidance. This is the original deliberately simple baseline. | `dumb_run.py` |
| `models/non_model_based/two_phase_belief_map_planner/` | The earlier two-phase information-then-coverage planner. It uses an agent-owned belief map and heading-aware A* route execution. | `run_phase_planner.py` |
| `models/non_model_based/partial_observation_frontier_planner/` | The final deterministic reference: thermal-aware frontier selection plus heading-aware Dijkstra execution, revisit control, and resource-aware meat routing. | Current standalone planner |

The two planners are not full-information methods. Their map is built only
from the current and past 5x5 observations. The shared belief-map utilities
also support learned target rankers, which is why they live in
`models/shared/belief_map_tools/` rather than inside one experiment folder.

## Model-based approaches

### Direct-action DQN family

Each member learns Q-values for the atomic actions `forward`, `left`, and
`right`. The common architecture fuses local sensory patches with recurrent
state; later variants change the exploration credit, replay, or route prior.
The original initial-heuristics source is retained for lineage, while the later
compact recurrent patch-fusion implementation in `models/benchmarks/` is the
lower DQN reference used by the final comparison.

| Final location | Distinguishing mechanism | Historical source |
| --- | --- | --- |
| `dqn/recurrent_patch_fusion_initial_heuristics/` | Initial recurrent patch-fusion Double DQN with structured exploration heuristics. | `train_dqn_v5.py` |
| `dqn/windowed_coverage_dqn/` | Windowed coverage shaping and prioritised recurrent replay. | `train_dqn_v6.py` |
| `dqn/novelty_elite_replay_dqn/` | Episodic observation novelty, high-coverage elite replay, and self-imitation. | `train_dqn_v7.py` |
| `dqn/future_coverage_credit_dqn/` | Credits setup turns/actions for later coverage gain and retains a light novelty signal. | `train_dqn_v8.py` |
| `dqn/safe_route_memory_dqn/` | Two-stage seen/coverage objective with local route-score alignment and thermal exclusion rules. | `train_dqn_v9.py` |
| `dqn/goal_conditioned_memory_dqn/` | Goal-conditioned recurrent Double DQN with an explicit agent memory map and teacher guidance. | `train_dqn_v10.py` |

### PPO/RND family

| Final location | Distinguishing mechanism | Historical source |
| --- | --- | --- |
| `ppo/recurrent_ppo_rnd_frontier_prior/` | Recurrent PPO with random-network-distillation novelty and a heuristic frontier action prior. | `train_ppo_rnd_v1.py` |
| `ppo/symmetric_recurrent_ppo_rnd/` | A symmetry-oriented refinement of the recurrent PPO/RND frontier-prior policy. | `train_ppo_rnd_v3.py` |
| `ppo/spatial_target_ppo_with_planner/` | PPO scores map targets; the deterministic planner supplies every local route action. Teacher target guidance decays to zero. | Current Coverage V7 |
| `ppo/planner_residual_target_ppo/` | PPO learns a bounded residual over persistent planner target scores; the planner remains both route executor and strategic anchor. | Current Coverage V8 |

The final two rows are model-based despite using a deterministic planner. The
learned policy chooses *where to go*; the planner chooses *how to get there*.
This is the central hybrid condition in the comparison.

### R2D2 family

| Final location | Distinguishing mechanism | Historical source |
| --- | --- | --- |
| `r2d2/recurrent_distributional_dqn/` | LSTM, distributional C51, duelling Double DQN, and prioritised sequence replay. | `train_r2d2_v1.py` |
| `r2d2/future_coverage_distributional_dqn/` | Tuned R2D2 with future-coverage credit and adaptive exploration. | `train_r2d2_v2.py` |

### Learned target ranking over the belief-map planner

`models/model_based/learned_frontier_ranker/` contains the learned frontier
and exploration scorers plus the uncertainty-gated ensemble. It does not emit
atomic actions: it ranks safe targets supplied by the belief-map planner. When
ensemble disagreement is too high, it falls back to the deterministic expert.
The included static-equivalence test verifies that the no-checkpoint mode
matches the two-phase planner.

## Benchmarks

| Final location | Why it is a benchmark |
| --- | --- |
| `models/benchmarks/recurrent_patch_fusion_dqn_reference/` | The later compact recurrent patch-fusion DQN reference. It is a lower learned benchmark, not the best learned result. |
| `models/benchmarks/full_information_oracle/` | Full-map beam-search oracle. It is intentionally privileged and therefore an upper bound, not a valid partial-observation controller. |
| `models/benchmarks/planner_vs_oracle/` | Recreates the same seeded map for the frontier planner and oracle, records a map fingerprint, and reports per-seed gaps. |
| `models/benchmarks/planner_vs_learned/` | Evaluates the unchanged planner and selected saved PPO checkpoints on the same held-out maps, records fingerprints and checkpoint hashes, and reports paired coverage differences. |
| `models/benchmarks/frontier_planner_component_ablation/` | Removes one named component from the released frontier planner on the same held-out maps, records all outcomes, and reports paired effects. |
| `models/benchmarks/planner_vs_residual_ppo/` | Generalises matched residual-PPO evaluation across map presets, fixed observation/resource scenarios, and independently trained residual checkpoints; it records scenario configurations, checkpoint hashes, and map fingerprints. |

## Interpretation of the completed comparison

On the held-out paired 15x15 evaluation, the final planner remains the best
deployable controller among the two selected saved PPO checkpoints. Residual
PPO has the stronger learned mean coverage but is 2.61 percentage points below
the planner on mean paired coverage and does not exceed its P10 reliability.
This is evidence about the present task configuration and fixed checkpoints,
not a general claim that learned controllers cannot improve planners. In
particular, the planner already encodes much of the structure needed for safe
frontier progress, resource timing, thermal cost, and revisit avoidance; a
learned residual needs strong evidence before overriding it.

A 50-map subtractive ablation strengthens the deterministic interpretation:
removing resource recovery, route revisit costs, or the safe-frontier-forward
rule lowers coverage in the released implementation, while the isolated
thermal-cost removal is inconclusive. This is deliberately narrower than a
claim that those settings are universally optimal.

Two additional 50-map fixed-condition tests give a deliberately narrow
robustness screen for the selected residual checkpoint: 15% public
visual-object dropout has a directionally positive residual-minus-planner
difference of 2.29 points (95% CI [-0.49, 5.20]) and one restorative resource
has a directionally negative difference of -0.74 points ([-1.64, 0.13]). Both
intervals include zero, so neither establishes a method advantage under its
stated condition. They are not independent PPO training runs or real-sensor
experiments.

A matched evaluation-only diagnostic also zeros the residual checkpoint's
persistent planner scores and tie bonus on 50 fresh nominal maps. Full prior
is directionally +1.13 points over the stripped version (95% CI [-0.41, 2.68];
p=0.392), but the interval includes zero. This does not establish dependence,
and it is not a substitute for training the architecture without its prior.

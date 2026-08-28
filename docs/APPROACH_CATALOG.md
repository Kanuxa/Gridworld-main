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

## Interpretation of the completed comparison

The final planner remains the best deployable controller on the completed
matched 15x15 evaluation. The residual target PPO reached the strongest learned
mean coverage, but it did not exceed the planner's P10 reliability. This is
evidence about the present task configuration, not a general claim that learned
controllers cannot improve planners. In particular, the planner already
encodes most of the structure needed for safe frontier progress, resource
timing, thermal cost, and revisit avoidance; a learned residual needs strong
evidence before overriding it.

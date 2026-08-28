# Run archive layout

All historical and current run artifacts are included in this directory.

| Location | Contents |
| --- | --- |
| `15x15/non_model_based/` | Deterministic sweep, the two-phase belief-map planner, final frontier-planner outputs, and verification trajectories. |
| `15x15/model_based/archived_direct_action_rl/` | All historical DQN, PPO/RND, and R2D2 runs from the legacy archive. Original artifact names, `MANIFEST.tsv`, and checksums are preserved. |
| `15x15/model_based/dqn/` | The long goal-conditioned memory DQN run. |
| `15x15/model_based/learned_frontier_ranker/` | Learned frontier/exploration scorers, adaptive ensemble evaluations, and retained obsolete prototypes. |
| `15x15/model_based/spatial_target_ppo_with_planner/` | Spatial target PPO runs with planner execution. |
| `15x15/model_based/planner_residual_target_ppo/` | Planner-residual spatial target PPO runs. |
| `15x15/benchmarks/` | The recurrent DQN reference, full-information oracle, and matched frontier-planner versus oracle comparison. |
| `31x31/` | Current area-scaled frontier-planner and planner-residual PPO results. |
| `45x45/` | Current area-scaled frontier-planner results. |

`Gridworld-main old` used only 15x15 worlds, including its adaptive and
verification experiments. Consequently no legacy run appears under 31x31 or
45x45.

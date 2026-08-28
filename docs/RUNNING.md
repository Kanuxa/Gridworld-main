# Running the approaches

Run all commands from the `Gridworld-main final` directory after installing
`requirements.txt`. Every generated artifact should be written beneath `runs/`.

## Non-model-based controllers

```bash
python -m models.non_model_based.deterministic_sweep_baseline.run \
  --episodes 10 --seed 50000 --output runs/15x15/deterministic_sweep_baseline.csv

python -m models.non_model_based.two_phase_belief_map_planner.run \
  --episodes 50 --seed 50000 --output runs/15x15/two_phase_belief_map_planner.csv

python -m models.non_model_based.partial_observation_frontier_planner.run \
  --preset 15x15-baseline --episodes 50 --seed 50000 --trace-every 1 \
  --save-dir runs/15x15/partial_observation_frontier_planner
```

The final frontier planner also supports `31x31-area-scaled` and
`45x45-area-scaled` presets.

## Learned controller families

Run a direct-action DQN experiment, for example the novelty and elite-replay
variant:

```bash
python -m models.model_based.dqn.novelty_elite_replay_dqn.train \
  --episodes 1200 --save_dir runs/15x15/novelty_elite_replay_dqn
```

Run an R2D2 experiment:

```bash
python -m models.model_based.r2d2.future_coverage_distributional_dqn.train \
  --episodes 1200 --save_dir runs/15x15/future_coverage_distributional_dqn
```

Run the planner-assisted target PPO policies:

```bash
python -m models.model_based.ppo.spatial_target_ppo_with_planner.train \
  --episodes 3000 --save_dir runs/15x15/spatial_target_ppo_with_planner

python -m models.model_based.ppo.planner_residual_target_ppo.train \
  --preset 31x31-area-scaled --episodes 3000 \
  --save-dir runs/31x31/planner_residual_target_ppo
```

## Benchmarks

```bash
python -m models.benchmarks.recurrent_patch_fusion_dqn_reference.train \
  --episodes 1200 --save_dir runs/15x15/recurrent_patch_fusion_dqn_reference

python -m models.benchmarks.full_information_oracle.run \
  --episodes 50 --seed 50000 --trace-every 10 --search-beam-width 8192 \
  --save-dir runs/15x15/full_information_oracle

python -m models.benchmarks.planner_vs_oracle.run \
  --episodes 50 --seed 50000 --trace-every 1 --god-search-beam-width 8192 \
  --save-dir runs/15x15/frontier_planner_vs_oracle
```

The oracle reads the complete map at reset. Report it only as an upper bound,
not in the fair partial-observation ranking.

## GUI

```bash
streamlit run gui/current_environment/checkpoint_runner.py
```

The current checkpoint runner supports the reference recurrent DQN, spatial
target PPO, and planner-residual target PPO through the three files in
`trains/`.

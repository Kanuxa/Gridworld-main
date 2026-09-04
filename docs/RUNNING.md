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

python -m models.benchmarks.planner_vs_learned.run \
  --episodes 50 --seed 60000 --bootstrap-resamples 20000 \
  --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_vs_learned_heldout

python -m models.benchmarks.frontier_planner_component_ablation.run \
  --episodes 50 --seed 65000 --bootstrap-resamples 20000 \
  --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_component_ablation_heldout

python -m models.benchmarks.planner_vs_residual_ppo.run \
  --episodes 50 --seed 82000 --scenario object-dropout-15 \
  --bootstrap-resamples 20000 --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_vs_residual_object_dropout_heldout

python -m models.benchmarks.planner_vs_residual_ppo.run \
  --episodes 50 --seed 83000 --scenario resource-scarce \
  --bootstrap-resamples 20000 --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_vs_residual_resource_scarce_heldout
```

The oracle reads the complete map at reset. Report it only as an upper bound,
not in the fair partial-observation ranking. The planner-versus-learned command
evaluates the two saved PPO checkpoints and the unchanged planner on the same
held-out maps; its inference is across maps for these fixed checkpoints, not
across independent PPO training seeds.

The component-ablation command removes one released planner mechanism at a
time on the same fresh maps. It is a subtractive, matched comparison rather
than a parameter sweep; its outputs include all per-map coverage differences,
variant configurations, exact sign tests, and bootstrap intervals.

`planner_vs_residual_ppo` evaluates the saved residual-PPO checkpoint and the
unchanged planner under one fixed scenario at a time. `object-dropout-15`
independently omits 15% of non-empty public visual-patch objects; hidden world
state is not changed. `resource-scarce` changes only the 15x15 baseline meat
count from three to one. These commands compare matched map instances and
archive scenario configurations, but they assess neither real-sensor
robustness nor independent PPO training seeds.

For the evaluation-only persistent-prior diagnostic, first evaluate the
released and stripped versions on the same new range, then analyse their two
CSVs directly:

```bash
python -m models.benchmarks.planner_vs_residual_ppo.run \
  --episodes 50 --seed 84000 --bootstrap-resamples 20000 \
  --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_vs_residual_prior_full_heldout

python -m models.benchmarks.planner_vs_residual_ppo.run \
  --episodes 50 --seed 84000 --remove-persistent-prior \
  --bootstrap-resamples 20000 --bootstrap-seed 25218029 \
  --save-dir runs/15x15/benchmarks/frontier_planner_vs_residual_without_persistent_prior_heldout

python tools/analyse_residual_prior_inference_ablation.py
```

The final command verifies that map fingerprints match before comparing the
two outcomes. It zeros planner target scores and the tie bonus only during
evaluation; it is therefore a dependency diagnostic, not a no-prior training
ablation.

### Independent training-seed replication protocol

The current report makes inference across fresh maps for fixed saved learned
checkpoints; it does not make a training-procedure claim across PPO seeds. The
following pre-specified protocol is provided for that missing experiment. It
trains residual-PPO seeds 13, 29, and 53 with the released settings, freezes a
new 50-map range beginning at seed 86,000, and evaluates all selected
`best_coverage.pt` files against the same unchanged planner. The protocol JSON
is written before training, and an incomplete set of seeds is not evaluated as
a completed replication.

```bash
python tools/run_residual_ppo_replications.py --dry-run
python tools/run_residual_ppo_replications.py
```

This is experimental infrastructure rather than a reported result. It should
be run only when the full three-seed budget is available; changing seeds,
training length, or the held-out range requires a new output root so that the
original protocol remains auditable.

## GUI

```bash
streamlit run gui/current_environment/checkpoint_runner.py
```

The current checkpoint runner supports the reference recurrent DQN, spatial
target PPO, and planner-residual target PPO through the three files in
`trains/`.

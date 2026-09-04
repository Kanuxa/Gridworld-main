# Towards Affordance in Robotics: Model-Based vs Non-Model-Based Approaches

This repository is the final, self-contained source layout for the Gridworld
research project. It reframes the work as a controlled comparison between
learned controllers and deterministic, non-model-based controllers under the
same partially observed hazardous Gridworld.

The valid task gives every deployable controller only a 5x5 egocentric
observation. A controller may maintain an agent-owned map from its observations
and action history, but it must never read the hidden world grid. The only
exception is the full-information oracle, which is explicitly a privileged
upper bound rather than a fair competitor.

## Research framing

The project answers two related questions:

1. Can a learned policy discover an exploration strategy that improves on an
   explicit planner under partial observation?
2. If a planner already supplies safe navigation and frontier selection, does a
   learned target policy add reliable value or mainly introduce variance?

The final reference is the **partial-observation frontier planner**. It is a
deterministic controller that reconstructs its own map, scores safe frontiers,
accounts for thermal cost, revisit cost, and resources, and uses heading-aware
Dijkstra search for local execution. It is the best valid partial-observation
controller in the completed comparison.

## Layout

| Location | Purpose |
| --- | --- |
| `gui/current_environment/` | Current environment, checkpoint runner, and GUI support. |
| `models/non_model_based/` | Deterministic sweep baseline and the two belief-map planner stages. |
| `models/model_based/` | DQN, PPO/RND, R2D2, learned frontier rankers, and planner-assisted PPO. |
| `models/benchmarks/` | Lower learned reference, full-information oracle, and matched comparison. |
| `models/shared/` | Environment-compatible model interfaces and belief-map planning utilities. |
| `trains/` | GUI-compatible adapters for the three checkpoint formats supported by the current runner. |
| `docs/` | Complete approach catalogue, experiment commands, and artifact policy. |
| `runs/` | All preserved experiment artifacts, grouped first by map size and then by research role. |
| `results/` | Curated evaluations, metrics, comparisons, analyses, and searchable result indexes. |
| `reports/` | The prior technical report, retained as a historical source document. |

All public directory names describe an algorithm or experimental role rather
than a serial version number. The original version identifiers remain only in
some legacy checkpoint metadata so a researcher can trace an artifact back to
its source implementation.

## Key results on the held-out paired 15x15 benchmark

| Controller | Observation regime | Mean coverage | P10 coverage | Role |
| --- | --- | ---: | ---: | --- |
| Partial-observation frontier planner | 5x5 observation + agent map | 71.41% | 67.96% | Held-out reference |
| Spatial-target PPO with planner executor | 5x5 observation + agent map | 67.32% | 62.18% | Selected checkpoint, episode 600 |
| Planner-residual target PPO | 5x5 observation + agent map | 68.80% | 62.67% | Selected checkpoint, episode 700 |
| Full-information oracle | Full map at reset | 78.69% | 76.00% | Privileged upper bound |

The first three results are paired on the same 50 held-out maps (seeds
60,000--60,049) and use the same partial-observation constraint. Relative to
the planner, the mean paired coverage differences are -4.10 percentage points
for spatial-target PPO and -2.61 points for residual PPO. These tests concern
the two saved checkpoints across fresh maps, not training-seed variance. The
oracle is not a fair competitor: it quantifies the remaining value of full map
knowledge. See [the approach catalogue](docs/APPROACH_CATALOG.md) for the
direct-action families, historical lineage, and caveats on sample size.

The released planner also has a separate 50-map paired component ablation on
seeds 65,000--65,049. Resource recovery, route revisit costs, and the
safe-frontier-forward rule each have a negative removal effect; the isolated
thermal-cost removal is inconclusive. The exact per-map outcomes and intervals
are linked from [the results overview](results/OVERVIEW.md).

Two further matched 50-map sensitivity tests assess the fixed residual PPO
checkpoint under 15% visual-object detection dropout and a one-restorative-
resource condition. Their residual-minus-planner mean differences are +2.29
points (95% bootstrap CI [-0.49, 5.20]) and -0.74 points ([-1.64, 0.13]),
respectively; neither interval excludes zero. They are limited simulated
condition shifts, not real-robot validation or PPO training-seed replication.

An additional matched, evaluation-only diagnostic removes the residual
checkpoint's persistent planner target scores and tie bonus after training.
The released full-prior version is directionally +1.13 points higher (95%
bootstrap CI [-0.41, 2.68]; p=0.392), so the result is inconclusive. It is a
prior-dependency check, not a no-prior training ablation.

## Quick start

Install the project requirements from this directory:

```bash
python -m pip install -r requirements.txt
```

Run the non-model-based reference controller:

```bash
python -m models.non_model_based.partial_observation_frontier_planner.run \
  --preset 15x15-baseline --episodes 50 --seed 50000 \
  --save-dir runs/15x15/partial_observation_frontier_planner
```

Train the planner-assisted spatial target PPO:

```bash
python -m models.model_based.ppo.spatial_target_ppo_with_planner.train \
  --episodes 3000 --save_dir runs/15x15/spatial_target_ppo_with_planner
```

Run the privileged upper bound separately:

```bash
python -m models.benchmarks.full_information_oracle.run \
  --episodes 50 --seed 50000 --search-beam-width 8192 \
  --save-dir runs/15x15/full_information_oracle
```

More commands, including the matched oracle comparison and the historical
families, are in [docs/RUNNING.md](docs/RUNNING.md).

## Included runs

All available generated runs are included. Every artifact from
`Gridworld-main old` is stored under `runs/15x15/`, because that project used
only the 15x15 environment. Current-project runs remain separated by their
actual map size: `runs/15x15/`, `runs/31x31/`, and `runs/45x45/`.

The direct-action RL archive retains its original run names and manifests for
auditability, while its parent directory records its meaningful research role.
See [runs/README.md](runs/README.md) for the exact layout.

# Dissertation figures

`tools/generate_report_figures.py` generates these Matplotlib PNGs from the
curated results archive. The matching copies in `report/images/` are embedded
by `report/main.tex`.

| Figure | Evidence used | Role in the report |
| --- | --- | --- |
| `final_benchmark_coverage.png` | Held-out planner-versus-learned summary plus archived 31x31 summaries | Main: 15x15 paired and 31x31 descriptive coverage / P10 comparison. |
| `paired_learned_delta_distribution.png` | `artifacts/15x15/benchmarks/frontier_planner_vs_learned_heldout/comparison_by_episode.csv` | Appendix: paired learned-minus-planner differences on held-out maps. |
| `frontier_planner_component_ablation.png` | `artifacts/15x15/benchmarks/frontier_planner_component_ablation_heldout/comparison_summary.json` | Main: paired effect of removing one deterministic planner component. |
| `paired_oracle_map_comparison.png` | `artifacts/15x15/benchmarks/frontier_planner_vs_oracle/comparison_by_episode.csv` | Appendix: matched planner-versus-oracle control. |
| `ppo_checkpoint_trajectories.png` | The two final PPO `episode_metrics.csv` files | Appendix: single-run checkpoint diagnostics. |
| `paired_oracle_delta_distribution.png` | The paired planner-versus-oracle comparison CSV | Appendix: distribution behind the paired-control estimate. |

The oracle figures show a privileged full-information upper bound and are not
part of the fair partial-observation ranking. The held-out learned figure
supports inference across maps for two fixed checkpoints, not training-seed
variation. The PPO trajectories are descriptive: they are not a multi-seed
uncertainty estimate.

The component-ablation figure tests one subtraction at a time on 50 further
matched maps. It supports resource recovery, route revisit costs, and the
safe-frontier-forward rule in this configuration; its thermal-cost interval
includes zero and is deliberately reported as inconclusive.

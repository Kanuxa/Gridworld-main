# Artifact policy

This final repository includes all available generated runs, checkpoints,
traces, configuration files, metrics, and historical archives from both source
projects. The originals were copied, never moved or modified.

All artifacts from `Gridworld-main old` are classified as 15x15 runs. Current
project outputs are grouped by their actual setting: 15x15, 31x31, or 45x45.
The 15x15 historical direct-action archive preserves its original directory
names, `MANIFEST.tsv`, and `SHA256SUMS` so the published audit trail remains
valid. Its parent directories provide the meaningful research taxonomy.

New outputs should be written under the matching map-size and approach
directory in `runs/`. Avoid overwriting preserved results; use a new
descriptive subdirectory for a rerun.

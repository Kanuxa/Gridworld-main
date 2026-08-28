# Results archive

This directory is a curated, non-binary view of the complete `runs/` archive.
It contains every available evaluation, metric, training log, configuration,
comparison, and analysis report. It intentionally excludes checkpoints and raw
trajectory payloads, which remain in `runs/`.

| File or directory | Contents |
| --- | --- |
| `artifacts/` | Results mirrored from `runs/` while preserving map-size and approach paths. |
| `results_catalog.csv` | One searchable row per copied result artifact. |
| `summary_metrics.csv` | Normalised aggregate metrics extracted from every `summary.json`. |
| `OVERVIEW.md` | Completed-comparison interpretation and direct links to core evidence. |

All legacy artifacts are under `artifacts/15x15/`. The current project also
contains true 31x31 and 45x45 results. Consult each run's configuration before
comparing different map sizes or environment settings.

Rebuild this folder after adding runs:

```bash
python tools/assemble_results.py
```

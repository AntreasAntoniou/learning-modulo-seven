# Dated validation draft: 15 October 2026

Snapshot verified at **16:48:57 UTC**, with **560/576** complete fits. All 288 binary fits are present; scalar Transformer-8 lacks two fits and scalar Transformer-12 lacks fourteen. This is a local paper draft, not the final selected/tested release. v0.1.0 and the original twelve-fit files are preserved.

- `snapshot.json`: all 560 per-fit within-selected and final observations; exact cases, exposure, parameter counts and content hashes. Outside metrics always refer to the within-selected checkpoint.
- `summary.json`: descriptive counts and ranges. The 99% threshold was chosen after observing the data and has no role in selection.
- `recipe-statistics.csv`: all 192 planned architecture/encoding/LR/decay cells. Means and sample SDs require all three seeds; six incomplete cells have blank statistics. No winner is designated.
- `illustrative-curves.jsonl.gz`: every 5,000-presentation observation for twelve trajectories (four binary recipes, three seeds each). Recipes were chosen post hoc to illustrate contrasting behaviours, retaining unsuccessful seeds. Training loss is a sample-weighted window mean.

The exporter checked hashes, exact cadence, finite values and within-checkpoint choice using all 1,120,000 validation records from the verified private histories. Those complete histories also contain 44,800,000 individual training batches. The large histories are not all included here; full final asset packaging remains pending.

Run `python analysis/verify_results.py` from the repository root to check the exported metrics, statistics, early-snapshot agreement and illustrative curves. `python analysis/make_figures.py` rebuilds the report figures. The script `analysis/export_million_snapshot.py` accepts explicit paths to the archived result snapshot, history index and histories; it has no cloud-writing or test-evaluation operation.

All 576 fits and their histories must be complete before the original validation-only selection is frozen and the 96 retained checkpoints enter one final test phase. This draft does not change that procedure.

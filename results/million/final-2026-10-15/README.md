# Completed million-example experiment — 15 October 2026

All 576 fits completed before validation-only selection was frozen. All 96 retained checkpoints were evaluated once on 65,536 held-out integers per domain. The final test results do not alter the frozen choices.

- `snapshot.json`: all fit identities, hyperparameters, selected and final validation points, source and artifact hashes.
- `selection.json`: 32 frozen architecture/encoding configurations and their 96 case IDs. `original_selection_sha256` identifies the immutable original selection before infrastructure fields were removed.
- `test-results.json`, `selected-per-seed.csv`, `selected-statistics.csv`: all selected test results and three-seed means/sample SDs. Accuracy is a fraction in data files; loss is cross-entropy. Seeds change training, not the dataset.
- `recipe-statistics.csv`: all 192 three-seed validation cells, including failures to learn.
- `selected-curves.jsonl.gz`: all 2,000 validation observations for each retained fit, with sample-weighted training-window loss and accuracy.
- `release-assets.json`: 35 scientific archives and all 1,443 member hashes. Download these from [v1.0.0](https://github.com/AntreasAntoniou/learning-modulo-seven/releases/tag/v1.0.0).
- `history-audit.json`, `prediction-audit.json`: verification scope and numerical differences. Saved accuracies match exactly; every saved logit row was rescored. Sampled CPU predicted classes agree with GPU classes, but raw-logit and probability tolerance failures are retained.

The 32 history archives hold every training batch and validation observation for all 18 fits of each architecture/encoding. Their `elapsed_seconds` values are attempt-local counters and are not a global time axis; plot `samples` for training exposure. Histories contain 46,080,000 batches and 1,152,000 validation observations. The predictions archive contains raw integer IDs and seven logits for each model/domain. The data archive contains the exact five mutually disjoint splits.

The selected-checkpoints archive contains safe numeric tensor arrays (`.npz`, load with `allow_pickle=False`) and scientific metadata. Reconstruct a model with `experiments/million/models.py:model_for`, using `case.variant`, `meta` and `width`; load tensors by their stored state-dictionary names. These are the within-selected weights, not necessarily the final update's weights. Optimizer/RNG state and private cloud lineage are not exported.

After downloading all 35 archives into one directory, verify hashes, every training window and every saved test metric without training or reselecting:

```bash
python analysis/verify_million_final.py --assets /path/to/downloaded/assets
```

The original twelve-fit snapshot and intermediate `draft-2026-10-15` directory remain unchanged. All curves are full resolution; illustration choices do not change the released population.

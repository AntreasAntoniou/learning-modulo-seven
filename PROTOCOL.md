# Experiment protocol

## Completed primary screen

Six task/encoding conditions: sphere volume, scalar modulo 7, binary modulo 7, product of two scalars, sin(6πx²), and mean pairwise squared distance of four scalars. Sixteen architecture variants; three nested training pools (256, 4,096, 65,536); three optimisation seeds; two learning rates. Total: 1,728 fits.

Modulo within domain: integers [0, 131072). Outside: [131072, 262144). Binary inputs have 18 bits; the highest bit is constant in training. The scalar is n / 65535.5 − 1. Each domain has 4,096 validation values and 8,192 test values. Splits are disjoint and evaluation values stay fixed across training amounts. Data seed 1729 is combined with a task-specific hash; scalar and binary modulo share raw integers.

Training: 1,200 steps, batch 128, sampling with replacement; AdamW with learning rate 0.0003 or 0.003, β=(0.9,0.999), ε=1e−8, decay 0; dropout 0; gradient clipping 1. Checkpoint validation at step 1 and every 100 steps. The best checkpoint minimises within-validation loss; exact checkpoint ties retain the earlier checkpoint. For each architecture, amount and seed, the learning rate is then selected by within-validation loss (lower LR breaks an exact tie). Test results are recorded for every fit's selected checkpoint but do not select checkpoint or LR. Outside validation is diagnostic.

The data-size comparison holds *steps* fixed. Nominal passes are 600, 37.5 and 2.34375, not equal epochs. Optimisation seeds do not resample the underlying dataset. Reported standard deviations use ddof=1.

## Separate observed-bit diagnostic

After the first primary seed was inspected, a 96-fit diagnostic was declared: all sixteen variants, three seeds and two learning rates, 65,536 training examples. It uses 17 bits, within [0,98304), outside [98304,131072), and the same split sizes and training recipe as the short screen. All bits vary during training; the joint highest-two-bit pattern 11 is held out. This changes several factors together and is not an isolated intervention on one bit. It remains separate in results and interpretation.

## Completed million-example sweep

The full machine-readable design is [config.json](experiments/million/config.json).

| Quantity | Value |
|---|---|
| Within domain | [0, 3145728) |
| Outside domain | [3145728, 4194304) |
| Unique training integers | 1,000,000 |
| Validation values per domain | 8,192 |
| Test values per domain | 65,536 |
| Data seed | 1729 |
| Encodings | scalar n / ((3145728−1)/2) − 1; 22 binary bits, each ±1 |
| Epochs | 10 full independently shuffled passes |
| Batch sizes | 128; 8 at each exact 5,000-example boundary |
| Total presentations / updates | 10,000,000 / 80,000 per fit |
| Learning rates | 0.0003, 0.003, constant |
| Weight decays | 0, 0.01, 0.1, applied to all learned parameters |
| Optimisation seeds | 101, 202, 303 |
| Optimiser | AdamW; β=(0.9,0.999), ε=1e−8 |
| Dropout / norm clipping | 0 / 1 |
| Precision | float32; TF32 disabled |
| Validation cadence | Both domains every 5,000 presentations |
| Logging / persistence | Every batch; checkpoint and metric segment every 100,000 presentations |
| Fits | 16 variants × 2 encodings × 2 learning rates × 3 decays × 3 seeds = 576 |

All five splits are mutually disjoint on raw integers; scalar float32 inputs were checked for aliasing. Every binary feature varies during training, with only the joint highest-two-bit pattern 11 absent. The fixed split is shared by all models. Epoch shuffles are paired across architecture and hyperparameters for a given seed.

**Selection, fixed before the run:**

1. Per fit: minimise within-validation cross-entropy, then maximise accuracy for exact loss ties; keep earlier checkpoint for a complete tie.
2. Wait for exactly one complete result for every preregistered fit. Reject source, dataset or configuration mismatches.
3. Per architecture/encoding: choose LR/decay by mean best within-validation loss across the three seeds. Ties use higher mean accuracy, then smaller decay and LR. Retain all three seeds, never the best seed alone.
4. Record an overall validation winner per encoding. Freeze the choices once.
5. Evaluate the 96 retained models once on both test domains. A create-once fence precedes any test predictions. Training has no test-file input.

The dated v0.1.0 public snapshot contains twelve binary g2/f3 fits and remains unchanged. The preserved intermediate 15 October draft adds a fixed 560-fit snapshot (288 binary and 272 scalar), all 192 planned three-seed recipe cells and twelve illustrative trajectories. Incomplete cells receive no average. Neither historical snapshot is the final phase or establishes a sweep winner. All 576 fits subsequently completed and the frozen choices were evaluated in one final test phase on 15 October 2026. The final records are in [final-2026-10-15](results/million/final-2026-10-15/); all 96 retained models and both test domains are included. The overall within-validation winners are binary g2/f3 (LR 0.0003, decay 0.01) and scalar relational2 (LR 0.0003, decay 0.1). No test-driven reselection was performed. The draft uses a post-hoc 99% threshold for descriptive counts and post-hoc illustrative curve choices; neither changes the preregistered selection rule. The follow-up changes range and training budget as well as data size, so it is not a fourth point on the primary data-size curve.

## Architecture definitions

MLP depths 3/20/50 count linear layers including the output; GELU hidden activations. Residual controls at depths 20/50 add each interior update divided by the square root of the number of interior layers. Transformers use 2/8/12 pre-normalised blocks, four heads, feed-forward width twice the model width, feature-position embeddings and a classification token. Relational modules use 2/8/12 untied rounds of all ordered pair messages, including self-pairs, with mean aggregation and scaled residual updates.

Conditional hypernetworks use f(x,g(x;θ)). Only θ is independently learned; g emits every weight and bias of f. The five variants are g2/f3, g8/f3, g12/f3, g8/f1, g8/f5; f's hidden width is eight. All families approach a 32,768 learned-parameter target by changing width. Exact counts are in the fit records. This does not match FLOPs, generated coefficient counts or optimisation difficulty.

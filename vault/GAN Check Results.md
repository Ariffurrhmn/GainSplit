---
tags: [results]
created: 2026-11-11
updated: 2026-11-11
---

# GAN Check Results

Hub: [[00 - Project Context]]. Answers [[Risks]] item 13 (GAN under-tuned next to the tuned FGM arm). Rule and design: [[Decisions]] (2026-11-10, GAN fairness check). Main results: [[Bootstrap - All Passes]], [[Pass 3 Results]].

Run by the owner on Kaggle (GPU T4 x2, commit 3e431d2), plan `gan_check`, 12/12 runs, **dev only** (`summary.csv` has no test columns, so the test set was not touched). Raw files: `results/gan_check/`. Settings checked in each `result.json`: generator lr 1e-5 / 2e-4 for the `glr` runs (BERT lr stays 5e-5); `step_mult` 3 for the `steps3x` runs (600 steps instead of 200).

Baseline: the existing Pass 1 dev scores for the same seeds 0-2.

## Dev macro-F1 (VERIFIED, computed from `gan_check/summary.csv` and `pass1/summary.csv`)
| Setting | New mean (sd) | Current mean (sd) | Diff | Per-seed diff | Rule passed |
|---|---|---|---|---|---|
| A4 10%, generator lr 1e-5 | 0.8677 (0.0020) | 0.8661 (0.0070) | +0.0016 | +0.0030, +0.0063, -0.0046 | no |
| A4 10%, generator lr 2e-4 | 0.8681 (0.0024) | 0.8661 (0.0070) | +0.0019 | +0.0006, +0.0075, -0.0023 | no |
| A4 1%, 3x steps | 0.7729 (0.0232) | 0.7962 (0.0108) | -0.0233 | -0.0555, +0.0121, -0.0266 | no |
| A1 1%, 3x steps (control) | 0.7885 (0.0042) | 0.7952 (0.0129) | -0.0067 | +0.0057, +0.0001, -0.0259 | n/a |

Rule (fixed before the runs): beat the current A4 on mean dev macro-F1 by more than one sd **and** on all 3 seeds. No setting passes: the generator-lr changes gain about 0.2 points (a quarter of one sd) and lose on seed 2; 3x training lowers A4 by 2.3 points and A1 by 0.7.

## Outcome
- **No robustness run.** Paper wording: "A generator learning-rate sweep (1e-5, 2e-4) and 3x longer training did not improve the GAN on the development set."
- Reading (INFERRED): the GAN's null result is not an artefact of its generator learning rate or of too few steps at 1%. Longer training at 1% hurts both BERT and the GAN (overfitting on 228 labeled headlines), more so for the GAN, with high seed variance.
- Limit: this is a small sweep (two values of one setting), not full tuning; the limitations section still names unequal tuning ([[Risks]] item 13).

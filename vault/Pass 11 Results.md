---
tags: [results]
created: 2026-11-19
updated: 2026-11-19
---

# Pass 11 Results

Hub: [[00 - Project Context]]. Predictions (frozen before the run): [[Pass 11 Predictions]]. Plan: `PLAN.md` (repo root). Related: [[Pass 10 Results]], [[Pass 2a Results]], [[Review - Draft v2]].

Run by the owner on Kaggle: notebook 22 (`arifurrhmn/notebook-22`, code commit `c1c2a18`, predictions status FROZEN in that commit, 289 tests passed on Kaggle, 70/70 runs, 16,355 s of notebook time). Raw files in `results/pass11/` (zip, runs, logs, pretraining receipts, analysis tables, the 70% budget file, notebook log). The four dose encoders stay in `results/pass11/ckpt/` (not committed; weights are git-ignored). Numbers below are VERIFIED from `results/pass11/analysis/*.csv`. Method: paired bootstrap over test items, 10,000 resamples, same resample for every seed, statistic = mean over seeds 0-4, Holm within each table. Runs of A1, A4, A5, A7, A20, A36 at budgets already run come from Passes 1, 2a, 2b, 8a, 10.

## What was run
| Block | Runs | Tag |
|---|---|---|
| A5, A7, A1, A20 at 70% labels (new budget file, nested between the 30% and 100% sets) | 20 | |
| A36 at 30% and 5% | 10 | |
| A4 and A1 at 1% with 35x the steps (7,000 steps, about 9.9 passes over U) | 10 | `long35` |
| A20 at 1% from MLM encoders pretrained for 100 / 350 / 1,000 / 3,000 steps | 20 | `dose*` |
| A37 (GAN + helper heads on the MLM encoder) at 10% and 1% | 10 | |

## Results
**Our recipe (A36) vs the seed recipe (A5), macro-F1 points** (`ours_vs_seed`; all robust):
| Labels | A36 | A5 | A36 - A5 [95% CI] |
|---|---|---|---|
| 1% | 88.81 | 78.12 | +10.68 [9.58, 11.84] |
| 5% | 90.61 | 86.62 | +3.99 [3.14, 4.82] |
| 10% | 91.60 | 88.44 | +3.15 [2.37, 3.93] |
| 30% | 92.41 | 90.77 | +1.64 [1.01, 2.26] |

**The seed recipe vs BERT** (`seed_parts`, Holm over 9 rows): A5 - A1 is -2.30 at 1% (robust), then +0.00, +0.23, -0.08, -0.25 at 5, 10, 30, 70% (all n.s.). A4 (GAN alone) - A1: +0.10, -0.11, -0.13 at 1, 5, 10% (n.s.), -0.55 at 30% (p 0.0066, Holm 0.053).

**The seed paper's regime, 70% labels** (`high_f1`, `high_accuracy`, `high_summary`; Holm over 6 rows). Test macro-F1 / accuracy: A1 92.74 / 92.76; A5 92.49 / 92.51; A20 93.15 / 93.16; A7 (FGM) 93.38 / 93.40. A7 - A1 +0.64 (Holm 0.021); A5 - A7 -0.89 (Holm 0.002); A5 - A20 -0.66 (Holm 0.054); A5 - A1 -0.25 (n.s.); A20 - A1 +0.41 (n.s.). With all labels (Pass 1, 7, 10): A1 93.35, A7 93.90, A20 93.43 accuracy. **No arm reaches the seed paper's 93.88% at 70% on our split** (best 93.40); FGM reaches it only with all labels. The seed paper's numbers are not comparable (see below).

**GAN-BERT with 35x the steps at 1%** (`gan_long`, Holm over 3 rows): A4 73.45 (s.d. 5.74) vs A1 81.37 at the same steps: -7.92 [-8.92, -6.94]. A4 at 35x vs A4 at normal steps: -7.07; A1 at 35x vs normal: +0.95 [0.31, 1.59]. More passes over U make the GAN worse.

**MLM dose curve at 1%** (`dose`, `dose_summary`; gain over A1 80.42): 100 steps -0.36 [-1.12, 0.40] (n.s.); 350 steps +1.66; 1,000 steps +3.08; 3,000 steps +5.24; 10,000 steps +7.42 (Pass 8a). Shares of the full gain: -5%, 22%, 42%, 71%, 100%. The 100-step encoder reads 6,400 rows, the GAN's exposure to U at 1%.

**The seed recipe on the MLM encoder** (`a37`): A37 86.64 vs A20 87.84 at 1% (-1.20 [-1.70, -0.69], robust); 90.61 vs 91.02 at 10% (-0.40, Holm 0.059).

## Verdicts against the frozen predictions
- **H11.18, H11.19 SUPPORTED:** A36 > A5 at 1, 5, 10 and 30%.
- **H11.21 SUPPORTED:** the seed recipe does not beat BERT at any budget (1% is significantly worse). The verdict code labels a significantly negative result "SUPPORTED (not better)" because the prediction said "not better"; read the sign.
- **H11.22 SUPPORTED:** the seed recipe adds nothing on the MLM encoder (significantly worse at 1%).
- **H11.20 partly:** "the seed recipe is not better than A1, A7, A20 at 70%" holds (A5 is below all three). The two numeric checks written with it fail: the spread of the four arms is 0.89 macro-F1 points (predicted within 0.7), and A1 (92.76) and A5 (92.51) accuracies are below the predicted 93.0-94.2 band. The prediction was too high for 70% labels.
- **H11.16 NOT SUPPORTED:** predicted no difference between A4 and A1 at 35x steps; A4 is 7.9 points worse (opposite direction, significant).
- **H11.5 holds** (gain rises with steps). **H11.6 fails** (100-step gain -0.36, predicted at least +1.5). **H11.7 fails** (3,000-step share 71%, predicted at least 85%).
- Not part of this run: H11.1-4 (CPU analyses), H11.8-15 (iSarcasmEval in-domain, RoBERTa), H11.17 (JEPA seeds), H11.23 (seed-aware interval of A36 - A5). They stay open.

## Reading (INFERRED)
- **The comparison with the seed recipe is clean and large at low labels** and shrinks with labels (+10.7 at 1%, +1.6 at 30%). Neither the GAN nor the helper-head design adds anything over BERT at any tested budget, and the full recipe is worse at 1%.
- **The exposure objection does not rescue the GAN.** With 35x the steps the GAN gets worse (training became unstable, large seed spread), while MLM's gain grows steadily with exposure and is zero at the GAN's exposure. So exposure matters for MLM, and the GAN's null is not a short-training artefact. Caveat: one long schedule, one learning rate; the instability is not diagnosed.
- **The "we match the seed paper's 93.88" line does not hold at 70%.** The earlier plan expected all arms near 93.4-93.9 there; they are at 92.5-93.4. The seed paper's reported 93.88 is not comparable: its Table II support is 28,619 for every model (the whole dataset), its BERT baseline scores 92.04, and it ran about 25 epochs for GAN-BERT vs about 10 for BERT (VERIFIED from the PDF; see `PLAN.md` section F). The supportable statement is that the recipe adds nothing over plain fine-tuning on a held-out test set.

## Gaps
- The byte-identical re-run check planned in `PLAN.md` D.3 (A1 and A20 at 1%, seed 0, reproducing Passes 1 and 8a after the data-loading change) was not part of notebook 22. The only re-run in the results is Pass 10's A20 rerun. The change in `data.py` (budget lookup) is covered by tests; an exact-match check is still open.
- Single long-GAN configuration; no diagnosis of the instability (generator loss, p_fake curves are in `runs/A4_p1_s*_long35/result.json` train_log).
- 70% regime: A36 was not run at 70% (self-training needs U; cost).

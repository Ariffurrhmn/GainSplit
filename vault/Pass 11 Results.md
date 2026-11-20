---
tags: [results]
created: 2026-11-19
updated: 2026-11-20
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

## Notebook 23: whole-dataset scoring (2026-11-19)
Run by the owner on Kaggle (`arifurrhmn/notebook-23`, code commit `03429aa`, predictions FROZEN with the H11.24-28 addendum, 293 tests passed, 16/16 runs, 103 min). Raw files in `results/pass11_full/`. Training is exactly that of the untagged runs (tag `full`); only the scoring differs: each run is also scored on every training row (L and U), dev, test, and on all rows together (28,501 rows in our dataset). Accuracy at the 0.5 threshold. Numbers VERIFIED from `results/pass11_full/analysis/`.

| Arm | Labels | L | U | all training rows | dev | test | **all rows** | all rows - test |
|---|---|---|---|---|---|---|---|---|
| A1 | 70% | 100.00 | 92.39 | 97.72 | 92.22 | 92.76 | 96.67 | +3.91 |
| A5 | 70% | 100.00 | 91.76 | 97.53 | 92.04 | 92.51 | 96.48 | +3.97 |
| A1 | 100% | 100.00 | - | 100.00 | 92.84 | 93.35 | 98.62 | +5.27 |

- **Exact reproduction (H11.27 SUPPORTED):** all 16 runs reproduce the originals' test probabilities exactly (max absolute difference at most 1e-6), including the A1 re-run at 1%, seed 0. This closes the re-run gap listed under notebook 22: Pass 1, Pass 11 and the current code give identical predictions.
- **H11.24 SUPPORTED:** accuracy on all rows is 3.97 points above test accuracy for A5 at 70%.
- **H11.25 SUPPORTED, with a reading rule:** A5's all-rows accuracy is 96.48, above 93.88. So 93.88 is not what an all-rows evaluation of a well-fit model of this kind gives either; it lies between our held-out (92.5) and all-rows (96.5) figures for the same recipe. The seed paper's figure therefore cannot be placed on a held-out scale (INFERRED); it is not "explained" by the all-rows protocol. The paper must not say that it is.
- **H11.26 (descriptive):** a fully trained BERT scores 98.62 on all rows and 93.35 on test; the seed paper's BERT scores 92.04. That is below our held-out figure and far below our all-rows figure, so its BERT baseline was less fit (their Fig. 4 shows training accuracy of about 0.94-0.95) or scored differently (INFERRED).
- **H11.28:** A1 at 70% SUPPORTED (U 92.39 vs test 92.76). A5 at 70% not supported by a margin of 0.05: U 91.76 vs test 92.51 is a 0.75 gap against the 0.7 predicted. The unlabeled rows score level with, or slightly below, held-out rows for both arms: the GAN reading U as unlabeled text does not give A5 an advantage on U.
- **What it means for the paper:** evaluating over all rows raises accuracy by about 4 points at 70% labels and 5 at full labels because the training rows are memorized (L accuracy 100%). A number computed over all headlines cannot be compared with held-out accuracy, and "beating 93.88" is not a meaningful target. Our models would score 96.5-98.6 under that protocol; this is a statement about the protocol, not a result to report as an improvement.

## Notebook 24: CPU analyses on existing predictions (2026-11-19)
Pushed to Kaggle as a private CPU notebook (`arifurrhmn/notebook-24`, code commit `e277a55`, predictions FROZEN, 3 tests passed, 0.8 min). Raw files in `results/pass11_cpu/` (zip, analysis tables, notebook log). Sanity check inside the notebook: A20 - A1 at 1% reproduces +7.42. Numbers VERIFIED from `results/pass11_cpu/analysis/`; 10,000 resamples, Holm within each table.

**Polarity-flip difference-in-differences** (error rate in points, positive-surface sarcastic headlines [n = 107] minus other sarcastic [n = 1,248]; helper arm minus base arm; Holm over 4 rows):
| Contrast | Budget | DiD [95% CI] | p / p_holm | error on positive group: base -> helper | error on other sarcastic: base -> helper |
|---|---|---|---|---|---|
| A2 vs A1 | 1% | +2.59 [0.03, 5.12] | 0.047 / 0.141 | 23.2 -> 20.0 | 26.7 -> 20.9 |
| A2 vs A1 | 10% | +2.76 [0.58, 5.14] | 0.016 / 0.064 | 14.2 -> 15.1 | 15.4 -> 13.5 |
| A31 vs A20 | 1% | +0.98 [-1.09, 2.84] | 0.329 / 0.658 | 17.6 -> 15.7 | 15.0 -> 12.1 |
| A31 vs A20 | 10% | -0.65 [-2.62, 1.15] | 0.503 / 0.658 | 11.4 -> 10.1 | 9.5 -> 8.9 |
- **H11.1 not supported after Holm** (direction positive at both budgets; raw intervals just exclude 0, Holm p 0.141 and 0.064). Reading (INFERRED, exploratory): helper heads help the positive-surface sarcastic headlines less than the others; at 1% their error still falls (23.2 to 20.0), at 10% it rises slightly (14.2 to 15.1) while the others fall. No evidence that the helpers push the encoder the wrong way in absolute terms at 1%.
- **H11.2 SUPPORTED:** no difference-in-differences on the MLM encoder.

**Seed-aware (two-level) bootstrap** (`cluster_bootstrap`; seeds resampled, then test headlines; "ratio" = cluster width / test-only width):
| Contrast | Budget | Diff | Test-only 95% | Cluster 95% | Ratio | Seeds ahead |
|---|---|---|---|---|---|---|
| A20 - A1 | 1% | +7.42 | [6.53, 8.30] | [6.18, 8.68] | 1.41 | 5/5 |
| A20 - A1 | 10% | +2.80 | [2.09, 3.53] | [1.96, 3.64] | 1.16 | 5/5 |
| A2 - A1 | 1% | +1.63 | [1.12, 2.16] | [0.63, 2.58] | 1.89 | 5/5 |
| A6 - A1 | 1% | +1.62 | [1.10, 2.15] | [0.80, 2.45] | 1.57 | 5/5 |
| A7 - A1 | 1% | +1.58 | [1.03, 2.15] | [0.46, 2.82] | 2.12 | 4/5 |
| A30 - A1 | 1% | +4.17 | [3.44, 4.90] | [2.99, 5.35] | 1.63 | 5/5 |
| A22 - A20 | 1% / 10% | -1.75 / -1.17 | [-2.60, -0.92] / [-1.83, -0.52] | [-2.88, -0.65] / [-2.01, -0.34] | 1.34 / 1.27 | 0/5 both |
| A36 - A5 | 1% | +10.68 | [9.54, 11.82] | [8.57, 13.36] | 2.10 | 5/5 |
| A36 - A5 | 5% | +3.99 | [3.14, 4.85] | [2.91, 5.09] | 1.28 | 5/5 |
| A36 - A5 | 10% | +3.15 | [2.38, 3.94] | [2.21, 4.09] | 1.21 | 5/5 |
| A36 - A5 | 30% | +1.64 | [1.01, 2.27] | [0.84, 2.45] | 1.27 | 5/5 |
| A20@1% - A1@10% | | -0.38 | [-1.14, 0.37] | [-1.43, 0.65] | 1.38 | 2/5 |
- **H11.3 SUPPORTED:** the MLM gain over BERT keeps its interval above 0 at 1% and 10% (+7.42 [6.18, 8.68], +2.80 [1.96, 3.64]).
- **H11.23 SUPPORTED:** A36 - A5 excludes 0 at every budget, 1% to 30% (+10.68 [8.57, 13.36] at 1%; one seed gives +15.2).
- **H11.4 not supported:** the cluster intervals of the three single routes are 1.6-2.1 times as wide as the test-only ones (predicted at least 2 for all three: 1.89, 1.57, 2.12), and none of them includes 0 (predicted at least one would). The +1.6 gains stay positive when training splits are resampled; FGM (A7) is ahead on 4 of 5 seeds (seed 2: -0.01).
- JEPA stays below MLM on every seed at both budgets; MLM at 1% vs BERT at 10% stays level (-0.38 [-1.43, 0.65]).

## Notebook 25: E1, in-domain iSarcasmEval (2026-11-20)
Run by the owner on Kaggle (`arifurrhmn/notebook639a3cfda1`, code commit `8110f91` = HEAD at the time, 302 tests passed, 120/120 runs plus three short pretrainings, 87.5 min). Receipts in `results/pass11_isarc/` (zip, notebook log `notebook_25_pass11_isarc.log`, runs, pretraining logs and info, budgets, analysis). The three new encoders (`mlm_isc28`, `mlm_isc100`, `mlm_hsub`) were also downloaded into `results/pass11/e1_receipt/ckpt/` (git-ignored, not committed). Numbers VERIFIED from `results/pass11_isarc/analysis/`; paired bootstrap over test items, 10,000 resamples, Holm within each table.

Mean iSarcasmEval test ROC-AUC (SD over 10 seeds):
| N labeled | A1 (BERT) | A20 isc28 | A20 isc100 | A20 hs (Headlines encoder) |
|---|---|---|---|---|
| 228 | 52.61 (3.93) | 51.27 (5.91) | 51.65 (5.59) | 51.19 (6.37) |
| 780 | 55.48 (4.12) | 57.93 (4.56) | 57.83 (4.74) | 57.73 (3.04) |
| 3,120 | 65.99 (1.60) | 67.94 (2.00) | - | 64.81 (1.95) |
Macro-F1 contrasts (A20 isc28 - A1): -0.21 / +1.01 / -0.66 at 228 / 780 / 3,120, all n.s. (p_holm 0.77-1.00).

- **H11.8 not supported** at N = 228 (-1.33 [-3.71, 1.07], p_holm 1.00) and N = 780 (+2.45 [0.06, 4.87], raw p 0.045, p_holm 0.45: in the predicted range but not robust); the N = 3,120 half is SUPPORTED (+1.95 [-0.23, 4.18]).
- **H11.9 not supported:** macro-F1 gain at N = 228 is -0.21 [-1.37, 0.97], below the predicted 0 to +4.
- **H11.10 mixed:** SUPPORTED at 228 and 3,120, not at 780 (+2.25 [-0.16, 4.75], p_holm 0.61, but the point estimate is outside +-2).
- **H11.11 not supported:** the 3,120-headline encoder gains +3.00 [2.16, 3.87] over BERT at Headlines 1% (predicted at least +4 and at least half of the full-pool +7.4; it keeps about 40%) and +1.18 [0.52, 1.85] at 10%. It is below the full-pool encoder by -4.41 and -1.62 (all p_holm < 0.001). The pool size explains a large part, though not all, of the smaller gain on a small pool (INFERRED).
- **H11.12 SUPPORTED:** 100 passes vs 28.7 passes: +0.38 and -0.10 AUC (both intervals inside +-1.6); no overfitting collapse.
- **Reading rule:** the in-domain gain neither replicates (N = 228: -1.33) nor collapses cleanly (N = 780: +2.45 with an interval just above 0, not robust). "In between, smaller and noisier." Domain-matched pretraining (isc28) is level with the Headlines encoder at 228 and 780 (+0.08, +0.20), and ahead only at N = 3,120 (+3.12, raw p 0.018, p_holm 0.19), so no robust evidence that in-domain text helps more than out-of-domain text.
- **Caveat:** with 228 labels BERT is near chance (AUC 52.6, SD 3.9); a comparison at the floor can neither show nor exclude a gain. The 14.3% base rate and noisy author labels make iSarcasmEval a weak test bed here; what can be said (INFERRED): MLM pretraining does not give a reliable gain on iSarcasmEval at these budgets, so the claim in the paper must stay scoped to Headlines.

## Gaps
- The byte-identical re-run check is closed by notebook 23 (see above).
- Single long-GAN configuration; no diagnosis of the instability (generator loss, p_fake curves are in `runs/A4_p1_s*_long35/result.json` train_log).
- 70% regime: A36 was not run at 70% (self-training needs U; cost).

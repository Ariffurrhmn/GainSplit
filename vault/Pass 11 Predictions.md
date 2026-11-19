---
tags: [predictions, plan]
created: 2026-11-19
updated: 2026-11-19
status: FROZEN
---

# Pass 11 Predictions

Hub: [[00 - Project Context]]. Plan: `PLAN.md` (repo root, sections B and D). Related: [[Experiment Plan]], [[Review - Draft v2]], [[Pass 10 Results]].

**Status: FROZEN on 2026-11-19** (set by the owner; code of the run notebook: commit cd14168). To freeze: edit what you disagree with, change `status` above to `FROZEN`, fill the line below, commit and push **before** the commit that a run notebook clones. Every `result.json` records its commit, so the order can be audited. After freezing, a change needs a new dated entry here and a re-freeze.

Frozen at: the commit that contains `status: FROZEN` in this file (`git log -- "vault/Pass 11 Predictions.md"`), 2026-11-19. Run-notebook code: `cd14168`. Any later change needs a new dated entry here and a re-freeze.

Rules (same as Passes 4-10): labels are IDEA until a result exists. A directional prediction is *supported* only if the difference goes the predicted way and p_holm < 0.05 (paired bootstrap over test items, 10,000 resamples, statistic = mean over seeds, Holm within each named table). A "no difference" prediction is *supported* if p_holm >= 0.05, and the interval is reported (not proof of equality). Any outcome is reportable. Metrics: test macro-F1 in points unless stated; iSarcasmEval ROC-AUC in points (x100).

## E0a. Polarity-flip difference-in-differences (CPU, existing predictions)
Statistic: [error(helper arm, positive-surface sarcastic) - error(base arm, same)] - [same on the other sarcastic headlines]; n = 107 positive-surface sarcastic test headlines (1,355 sarcastic in all).
- **H11.1** A2 vs A1: DiD > 0 (helpers hurt positive-surface sarcastic headlines more). Expected small; the interval may include 0 because n = 107.
- **H11.2** A31 vs A20: DiD not different from 0 (pretraining removes the helper effect, as in Pass 10).
Exploratory; reported whatever the outcome.

## E0b. Seed-aware bootstrap (CPU, existing predictions)
Two-level bootstrap: resample the five training seeds, then test items.
- **H11.3** A20 - A1 at 1% and 10% keeps an interval above 0 (the +7.4 and +2.8 survive).
- **H11.4** The single-route gains over A1 at 1% (A2, A6, A7, about +1.6 each) get intervals at least twice as wide as the test-only intervals; at least one of the three includes 0.

## E2. MLM dose curve at 1% labels
Encoders: MLM, seed 0, steps in {100, 350, 1,000, 3,000}; 10,000 steps is Pass 8a. A20 at 1%, seeds 0-4, gain over A1 (80.42).
- **H11.5** The gain rises with steps (each step count at least as high as the previous one, within 0.5 points).
- **H11.6** The 100-step encoder (6,400 rows read, the same U exposure as GAN-BERT at 1%) already gains at least +1.5 over A1, but at most 60% of the 10,000-step gain (+7.42).
- **H11.7** The 3,000-step encoder reaches at least 85% of the 10,000-step gain.
Reading rule written in advance: gain at 100 steps <= +1 -> exposure is a plausible partial explanation of the GAN null, E3 becomes must-run; gain at 100 steps >= +4 -> the objective, not exposure, separates MLM from the GAN-type routes.

## E1. In-domain comparison on iSarcasmEval (author labels)
L = N tweets from the non-dev training rows, N in {228, 780, 3,120}, 10 seeds; scored on the iSarcasmEval test set (1,400 tweets, 14.3% sarcastic). **Primary metric: ROC-AUC.** A20-iS = MLM on the iSarcasmEval training texts minus dev, 1,400 steps at batch 64 (about 28.7 passes), no labels, no test. A20-iS100 = 4,900 steps. A20-HS = the Headlines MLM encoder (Pass 8a). Size-matched control = MLM on a random 3,120-headline subset, 1,400 steps, A20 on Headlines at 1% and 10%.
- **H11.8** A20-iS > A1 in AUC at N = 228 and N = 780, by +1 to +5 points; at N = 3,120 by at most +2 (gain shrinks with labels, as on Headlines).
- **H11.9** The in-domain gain is smaller than on Headlines: in macro-F1, A20-iS - A1 at N = 228 lies between 0 and +4 points (Headlines at 1%: +7.4).
- **H11.10** A20-HS is not better than A1 (AUC difference within +-2 and p_holm >= 0.05): pretraining on headlines does not help tweets, consistent with the zero-shot result.
- **H11.11** Size-matched control: the 3,120-headline MLM encoder still gains at least +4 over A1 at Headlines 1% and at least half of the full-pool gain, so a small pool alone does not explain a small iSarcasmEval gain.
- **H11.12** The 100-pass encoder is within 1 AUC point of the 28.7-pass encoder (no overfitting collapse).
Reading rules written in advance: "replicates" = AUC gain >= +3, robust at N = 228 and 780; "collapses" = gain < +1 or interval includes 0 at both N = 228 and 780. A null counts as "no detectable gain", not "no gain" (the test set has 200 positives).

## E4. RoBERTa-base (A1 vs A20)
Learning rate chosen on dev only: A1 at 10%, lr in {2e-5, 3e-5, 5e-5}, seeds 0-2, best mean dev macro-F1, ties within one s.d. go to the smaller lr. MLM pretraining 10,000 steps, seed 0, other settings as Pass 8a. Test at 1% and 10%, seeds 0-4. No masked tests.
- **H11.13** A20-RoBERTa > A1-RoBERTa at 1% and 10% (p_holm < 0.05).
- **H11.14** The gain at 1% is positive but not larger than BERT's +7.42 (RoBERTa's pretraining already includes news text).
- **H11.15** A1-RoBERTa at 1% is above A1-BERT (80.42).

## E3 (optional). GAN-BERT with many passes over U
A4 and A1 at 1% with `step_mult = 35` (7,000 steps, about 9.9 passes over U), seeds 0-4. Contrast: A4 - A1 at matched steps.
- **H11.16** |A4 - A1| < 1 point and p_holm >= 0.05 (the GAN null survives long training).
Note: `step_mult` also multiplies the passes over L; the matched A1 is the control.

## E5 (optional). Two more JEPA pretraining seeds
JEPA seeds 1 and 2, other settings as Pass 8b; A22 at 1% and 10%, seeds 0-4.
- **H11.17** Every JEPA encoder is below the seed-0 MLM encoder at 1% and 10% (robust), and the range of the three JEPA encoders' 1% means is below 1.5 points.

## Not predicted (descriptive only)
Frozen linear-probe scores of the new encoders; training curves; A6-type exposure on the self-training student.

## Added 2026-11-19 (agreed lane; frozen together with the rest)

### E6. Seed recipe vs our recipe across budgets
Arms: A5 (GAN + helpers, the seed recipe re-implemented), A1, A7, A20, A36. Budgets 1, 5, 10, 30% (A36 new at 5 and 30%) and 70% (new; A1, A5, A7, A20 only). Seeds 0-4, test macro-F1 and accuracy. Contrasts in one Holm family each per budget set.
- **H11.18** A36 > A5 at 1% and 10% (robust); expected about +10 and +3 (existing runs).
- **H11.19** A36 > A5 at 5% and 30% (robust), with the gap shrinking as labels grow.
- **H11.20** At 70%, A1, A5, A7 and A20 are within 0.7 macro-F1 points of each other and all between 93.0 and 94.2 accuracy; no contrast between A5 and any of the other three is significant (p_holm >= 0.05) in the direction A5 > others.
- **H11.21** A5 is not better than A1 at any of 1, 5, 10, 30, 70% (no significant positive A5 - A1).
Reading rule written in advance: if H11.20 fails because A5 is clearly above the others at 70%, the sentence "the seed accuracy needs no GAN or helpers" is dropped.

### E3 (now must-run) and A5-on-MLM (optional)
- **H11.16** stands (A4 - A1 at 35x steps within 1 point, not significant).
- **H11.22** (optional arm) A5 on the MLM encoder is not better than A20 at 1% or 10% (no significant positive difference).

### E-CPU headline contrast
- **H11.23** A36 - A5 on the existing predictions, paired bootstrap over test items with the seed-aware interval of E0b: the interval excludes 0 at 1% and 10%.

## Addendum 2026-11-19, before notebook 23 (whole-dataset scoring; dated entry, frozen with this commit)
Context: notebook 22 results ([[Pass 11 Results]]) showed no arm reaching the seed paper's 93.88% at 70% labels on a held-out test set. The seed paper's Table II has support 28,619 for every model (the whole dataset). Notebook 23 trains A5 and A1 at 70% labels and A1 with all labels (5 seeds, tag `full`, exactly the training of the untagged runs) and scores them on L, U, dev, test and all rows together (our dataset: 28,501 rows). Plus an A1 re-run at 1%, seed 0 (tag `rerun11`). Accuracy at the 0.5 threshold. These are descriptive checks (no bootstrap); "all rows" = train + dev + test.
- **H11.24** A5 at 70%: accuracy on all rows is at least 2.0 points above its test accuracy (the training rows are fit much better than held-out rows).
- **H11.25** A5 at 70%: accuracy on all rows is at least 93.88 (the seed paper's figure), although its held-out test accuracy is 92.5. If this fails, an all-rows evaluation does not explain the seed paper's number and the paper must not suggest it does.
- **H11.26** A1 with all labels (the seed paper's BERT setting; their reported BERT accuracy: 92.04): descriptive only, no direction. Reading rule: if our all-rows accuracy for a fully trained BERT is far above 92.04, the seed paper's BERT was either less fit or not scored this way, and the paper says only that its numbers cannot be compared with held-out accuracy.
- **H11.27** Every `full` and `rerun11` run reproduces the original run's test probabilities exactly (max absolute difference at most 1e-6). If not, stop: training is no longer identical to Passes 1 and 11 and no earlier comparison with these runs is valid.
- **H11.28** For A5 and A1 at 70%: accuracy on U (the 30% that never had labels) is within 0.7 points of test accuracy, and accuracy on L is at least 2.0 points above test accuracy.

## Addendum 2026-11-19, before notebook 25 (E1 implementation notes; no hypothesis above is changed)
How H11.8-H11.12 are implemented, fixed before the run: iSarcasmEval training tweets are split as in Pass 3 (fixed 10% stratified dev split, never trained on; 3,120 non-dev tweets, 25.0% sarcastic). Label sets of N = 228, 780 and 3,120 tweets are drawn per seed (0-9) as prefixes of one seeded permutation per class, keeping the pool's class share, so they are nested; N = 3,120 is the whole pool. Fine-tuning is the A1 recipe unchanged (lr 5e-5, batch 32, max length 48, `max(200, 10 passes over L)` steps: 200 / 250 / 980). Arms and tags: A1 `isc`; A20 `isc28` (MLM on the non-dev training tweets, no labels, 1,400 steps at batch 64); A20 `isc100` (4,900 steps, N = 228 and 780 only); A20 `hs` (the Pass 8a Headlines encoder); size-matched control A20 `hsub` on Headlines at 1% and 10% labels, seeds 0-4, from an MLM encoder on a random 3,120-headline subset (fixed subset seed 20261119) for 1,400 steps. Primary metric ROC-AUC on the iSarcasmEval test set (1,400 tweets, 200 sarcastic); macro-F1 at the 0.5 threshold secondary. Paired bootstrap over test items, 10,000 resamples, mean over seeds, Holm within each table (`isarc_auc` 11 rows, `isarc_f1` 6 rows, `size_control` 4 rows). Where H11.8 says "robust" it means p_holm < 0.05 in `isarc_auc`.

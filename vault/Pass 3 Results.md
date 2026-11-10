---
tags: [results]
created: 2026-11-11
updated: 2026-11-11
---

# Pass 3 Results

Hub: [[00 - Project Context]]. Plan: [[Experiment Plan]] ("Pass 3 plan"). Earlier results: [[Bootstrap - All Passes]]. GAN fairness check (same night): [[GAN Check Results]]. Decisions: [[Decisions]].

Run by the owner on Kaggle (GPU T4 x2, commit 3e431d2, dataset `gainsplit-processed` with the 14-file version), 105/105 runs, test scored. Raw files: `results/pass3/`. The owner ran the plan twice by mistake (two notebooks); both runs give identical numbers in all 1,575 metric cells of `summary.csv`, so training is deterministic across Kaggle sessions. The second copy was not kept.

Analysis: owner re-ran `experiments/03_analysis.ipynb` locally on 2026-11-11 on pass1 + pass2a + pass2b + pass3 (255 runs, 105 tagged). The Pass 1-2 contrast table reproduces unchanged. Paired bootstrap, 10,000 resamples, RNG seed 0, as before. Output: `results/analysis/`.

Status: every number below is VERIFIED (read from the files named). Readings marked INFERRED are interpretation. Holm columns were computed from the same files (p reported as 0 treated as 1/10,001); they are robustness checks, not the pre-set test.

## Checks (VERIFIED)
- **Mask re-runs:** all 25 (A1, A3, A4, A6, A7 at 10%, seeds 0-4) reproduce their untagged Pass 1-2 runs exactly: test macro-F1 difference 0 and prediction files identical (`mask_rerun_check.md`; also checked by hand on dev and iSarcasmEval macro-F1).
- **Helper-label counts** (`result.json`, seed 0): flip25 / flip50 move the class counts toward uniform; vader and teacher2 replace the sentiment labels only (emotion counts unchanged); soft, sentonly and emoonly keep the original label file (they change the loss, not the labels). The flipped labels at seed 0 differ slightly between A3 10% and A2 1%: the flip is seeded per run, not once per seed (state in Methods).

## RQ2: helper-label quality (`label_quality_contrasts.md`)
Each row against its own base (A3 at 10%, A2 at 1%), points of test macro-F1.

| Contrast | Budget | Diff | 95% CI | p | Holm (16) |
|---|---|---|---|---|---|
| A3 flip50 - A3 | 10% | **-0.85** | [-1.27, -0.44] | 0.0006 | 0.009 |
| A3 vader - A3 | 10% | **-0.63** | [-1.01, -0.24] | 0.002 | 0.031 |
| A3 soft / flip25 / sentonly / emoonly - A3 | 10% | -0.16 to -0.21 | all include 0 | 0.19-0.35 | 1.0 |
| A3 teacher2 - A3 | 10% | -0.00 | [-0.31, 0.32] | 0.99 | 1.0 |
| A2 sentonly - A2 | 1% | -0.47 | [-0.86, -0.07] | 0.022 | 0.29 |
| A2 other rows - A2 | 1% | -0.37 to +0.17 | all include 0 | 0.07-0.97 | >= 0.74 |
| A9 (A2 shuffled) - A2 | 1% | -0.46 | [-0.88, -0.03] | 0.034 | 0.41 |
| A9 (A2 shuffled) - A1 | 1% | **+1.17** | [0.63, 1.71] | < 0.0001 | 0.002 |

Readings (INFERRED):
1. **At 1%, most of the helper gain is not label content.** A2 beats A1 by +1.63 ([[Bootstrap - All Passes]]); with shuffled labels (A9) +1.17 of it remains (robust). About 70% of the gain comes from having extra heads to train on the labeled rows, not from what the labels say. The rest (-0.46) is not robust to Holm.
2. **At 10%, label content matters only against heavy damage.** Half the labels flipped costs 0.85 points and VADER labels cost 0.63 (both robust); soft labels, 25% flips, a different teacher and dropping either head change nothing measurable. Together with A3 - A8 +0.57 (not robust, [[Bootstrap - All Passes]]), the 10% gain is fragile.
3. **Agreement with human labels does not order the sources.** Gold kappa ([[Gold Audit Results]]): teacher 1 0.24, VADER 0.26, teacher 2 0.14. Yet teacher 2 (worst kappa) costs nothing and VADER (best kappa) costs 0.63. So "label quality" as measured against surface sentiment does not predict usefulness. Possible cause: VADER's class balance is very different (negative / neutral / positive 7.6k / 7.8k / 7.4k vs teacher 1 7.5k / 12.8k / 2.5k), so its drop may come from the distribution, not accuracy (IDEA, untested).
4. Sentiment-only vs emotion-only: neither head alone differs from both heads at 10%; at 1% removing the emotion head costs 0.47 (not robust). The mostly-one-class emotion labels ([[Risks]] item 11) do not make the emotion head useless at 1%, but the evidence is weak.

## RQ3a: outlet-style shortcuts (`masked_test_macro_f1.md`)
Masking the 200 most outlet-indicative words (62% of test headlines touched) vs masking the same number of random words. Gap = random-mask F1 - cue-mask F1.

| Arm (10% unless noted) | Test | Cue-mask drop | Random-mask drop | Gap [95% CI] |
|---|---|---|---|---|
| A1 BERT | 88.21 | 5.57 | 2.41 | 3.16 [2.42, 3.90] |
| A3 helpers L+U | 88.64 | 5.64 | 2.35 | 3.29 [2.55, 4.06] |
| A4 GAN | 88.09 | 5.81 | 2.33 | 3.48 [2.71, 4.28] |
| A6 self-training | 88.81 | 4.74 | 1.70 | 3.04 [2.32, 3.76] |
| A7 FGM | 89.76 | 4.72 | 1.61 | 3.11 [2.39, 3.85] |
| A2 / A3 / A9 label-variant rows (15) | 81.6-88.6 | 5.0-5.8 | 2.0-2.8 | 2.85-3.42, every CI excludes 0 |
| A1 trained on iSarcasmEval (control) | 42.61 | 0.20 | -0.32 | 0.53 [-0.31, 1.38], p 0.22 |

Readings (INFERRED):
1. **Every Headlines-trained model relies on outlet words.** Cue masking costs about 3 points more than random masking in all 20 Headlines-trained rows (p < 0.0001 each). The control model trained on iSarcasmEval shows no gap, so the cue list itself is not what breaks the models.
2. **No method reduces that reliance.** The gap is about 3 points for BERT, the GAN, helpers, self-training and FGM alike (intervals overlap heavily; no between-arm test of the gap was run). FGM and self-training lose less in absolute terms, but also less under random masking: they are more robust to word removal in general, not less reliant on outlet cues.
3. Helper labels do not fix this either: the label-variant rows have the same gap.

## RQ3b: transfer to iSarcasmEval (`transfer_isarc_auc.md`)
ROC-AUC (points, 50 = chance) with bootstrap 95% intervals over test items, seeds averaged.
- **In-domain reference** (A1 trained on iSarcasmEval train): **65.8 [61.6, 69.8]**. The test set is learnable well above chance, so the Headlines models' near-chance scores are a transfer failure, not an impossible test.
- **Headlines-trained BERT arms: 47.2-57.2.** Intervals above 50 only for A5 30% (57.2 [53.6, 60.7]), A3 30% (55.0 [51.2, 58.8]), A3 vader 10% (55.1 [51.5, 58.8]), A3 emoonly 10% (54.3 [50.6, 58.0]): 4 of 41 distinct Headlines-trained BERT models (45 rows; the 4 `mask` rows duplicate their untagged runs), all helper arms trained on L + U. TF-IDF A0 is below chance at every budget (43.9-45.7, intervals exclude 50).
- Reading (INFERRED): helper heads trained on the unlabeled pool may carry a trace of transferable signal at higher budgets, but it is a few AUC points, 10 below the in-domain reference, and 4 of 41 could partly be chance (no correction applied). The planned A3 - A1 and A5 - A1 transfer contrasts at 30% were not computed.
- **AUC source** (VERIFIED, 2026-11-11): prediction files store probabilities rounded to 6 decimals, which creates ties for confident models (236-533 distinct scores out of 1,400 for A5 30%). AUC from the files differs from the in-training AUC in `summary.csv` by up to 1.3 points per seed (A5 30%: 57.19 vs 57.63 averaged). The paper uses one source; see [[Decisions]].

## RQ3c: do helper labels encode sarcasm? (`helper_vs_sarcasm.md`, `gold_vs_teacher_sarcasm.md`)
- Training set (22,800): teacher sentiment vs sarcasm NMI 0.004, Cramer's V 0.08; emotion NMI 0.002, V 0.06. A logistic probe on the 8 teacher probabilities predicts sarcasm at **ROC-AUC 0.546** (5-fold CV).
- Gold items (200): the owner's surface sentiment predicts sarcasm at AUC 0.458, the teacher's at 0.496; difference -0.038, permutation p 0.67.
- Reading (INFERRED): the helper labels carry almost no sarcasm signal of their own, and the teacher does not encode outlet style more than the human labels do. The helper heads are not a hidden shortcut to the sarcasm label. That fits RQ2 reading 1: at 1% they act mostly as extra training signal on the shared encoder (regularisation), not as a source of sarcasm information.

## Gold table (`gold_sentiment_agreement.md`)
Matches [[Gold Audit Results]]: teacher 1 accuracy 53.0 / kappa 0.237; VADER 50.5 / 0.256 (0.374 on the 104 sure items); teacher 2 48.0 / 0.137.

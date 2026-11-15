---
tags: [results]
created: 2026-11-15
updated: 2026-11-15
---

# Pass 10 Results

Hub: [[00 - Project Context]]. Plan and predictions (fixed before running): [[Experiment Plan]] "Pass 10 plan". Novelty: [[Novelty Check - MLM Pretraining]]. All four questions are in.

## Q1: helpers, FGM and self-training on the MLM encoder (notebook 19)
Run by the owner on Kaggle (slug `notebookd630829dea`, commit 166e5c2, 280 tests passed, 62/62 runs, 92.4 min). Raw files in `results/pass10_stack/` (zip, runs, logs, analysis, log renamed `notebook_19.log`); unpacked folder matches the zip (`diff -r`). Encoder check PASS; **re-run check PASS** (A20 and A21 at 1% seed 0 identical to Pass 8a; A2, A6, A7, A1 rows from earlier passes also identical). Numbers VERIFIED from the tables.

**Each cell on plain BERT and on the MLM encoder** (test macro-F1, mean over 5 seeds)
| Cell | 1% BERT | 1% MLM | 10% BERT | 10% MLM |
|---|---|---|---|---|
| none (A1 / A20) | 80.42 | 87.84 | 88.21 | 91.02 |
| helpers (A2 / A31) | 82.05 | 88.32 | 88.43 | 91.09 |
| FGM (A7 / A21) | 82.00 | 87.75 | 89.76 | 91.15 |
| self-training (A6 / A32) | 82.04 | 88.19 | 88.81 | 91.05 |
| helpers+FGM (A27 / A33) | 82.97 | 88.27 | 89.48 | 91.26 |
| helpers+self-training (A28 / A34) | 83.06 | **88.89** | 89.23 | 91.13 |
| FGM+self-training (A29 / A35) | 83.86 | 88.50 | 90.28 | 91.48 |
| all three (A30 / A36) | 84.60 | 88.81 | 89.88 | **91.60** |

**Gains on top of MLM** (`stack_f1`, Holm over 16 rows): at 1%, helpers +0.48 (p 0.007, Holm 0.089), FGM -0.09, self-training +0.35 (n.s.), helpers+FGM +0.43 (Holm 0.40), **helpers+self-training +1.05 [0.55, 1.56] (Holm < 0.001)**, FGM+self-training +0.67 (Holm 0.32), **all three +0.97 [0.39, 1.54] (Holm 0.023)**. At 10% nothing survives Holm (all three +0.58, p 0.018, Holm 0.20). MLM under the full stack: **A36 - A30 +4.21 at 1%, +1.71 at 10%** (robust).

**Interactions** (Holm over 14 rows): on the MLM encoder no pair of ingredients interacts (all within -0.4 to +0.4, n.s.); the combinations keep 110-280% of the summed single gains, i.e. they add up, but every gain is small. Pretraining shrinks each ingredient's own gain at 1%: helpers -1.15, FGM -1.67, self-training -1.26 (all robust); at 10% FGM -1.41 robust, the others n.s.

**Self-training health:** the MLM teacher is much stronger (dev 86.7-87.2 at 1% vs 79.3-81.6 on BERT); it keeps 17.6k-19.7k pseudo-labels, and **92-96% of them are correct** against the hidden training labels (first time measured).

**Verdicts:** H10.1 not supported (self-training alone +0.35 at 1%, n.s.; 0 at 10%). H10.2 holds (helpers and FGM add nothing significant on top of MLM; helpers +0.48 at 1% is p 0.007 before Holm). H10.3 SUPPORTED at 1% (+0.97), n.s. at 10%. H10.4 SUPPORTED (pretraining shrinks helpers' and FGM's gains); pretraining x self-training also negative (-1.26). H10.5 SUPPORTED (MLM adds +4.2 / +1.7 under the full stack).

## Q2 + Q3: pretraining seeds and the label curve (notebook 20)
Run by the owner on Kaggle (Kaggle slug `notebook6d64711193`, commit 166e5c2, 280 tests passed, 35/35 runs, 92.8 min; `RUN_FULL100 = True`, `RUN_LONG = False`). Raw files in `results/pass10_seeds/` (zip, unpacked runs and logs, pretraining receipts for seeds 1 and 2, analysis, log renamed `notebook_20.log`); unpacked folder matches the zip (`diff -r`). The seed-0 encoder was found in notebook 15's output and its weights matched the Pass 8a receipt. Numbers VERIFIED from those tables. Method: paired bootstrap over test items, 10,000 resamples, mean over seeds 0-4, Holm within each table.

**Pretraining receipts.** Seeds 1 and 2: about 33 min each, dev MLM loss 5.85 -> 3.66 / 3.69 (seed 0: 3.68). Frozen probe at 1%: 86.1 and 84.2 (seed 0: 86.4).

**Q2: three encoders** (test macro-F1 of A20, mean over 5 fine-tuning seeds)
| Budget | seed 0 | seed 1 | seed 2 | mean | range |
|---|---|---|---|---|---|
| 1% | 87.84 | 88.14 | 87.31 | 87.76 | 0.83 |
| 10% | 91.02 | 90.90 | 90.47 | 90.80 | 0.55 |

Gain over A1: seed 1 **+7.72** [6.82, 8.63] at 1%, **+2.68** at 10%; seed 2 **+6.89** [5.97, 7.82], **+2.26**; mean of the three **+7.34** [6.50, 8.22] and **+2.58** [1.92, 3.25]. All p_holm < 0.001; every encoder ahead of A1 on 5/5 fine-tuning seeds.

**Q3: label curve** (test macro-F1, mean over 5 seeds)
| Budget | A1 (BERT) | A20 (MLM) | Gain | A1 budget with the same F1 | Labels saved |
|---|---|---|---|---|---|
| 1% | 80.42 | 87.84 | +7.42 | 8.5% | 8.5x |
| 5% | 86.61 | 90.10 | **+3.49** [2.76, 4.22] | 22% | 4.4x |
| 10% | 88.21 | 91.02 | +2.80 | 33% | 3.3x |
| 30% | 90.85 | 91.89 | **+1.04** [0.50, 1.61] | 50% | 1.7x |
| 100% | 93.33 | 93.41 | +0.09 [-0.42, 0.60], n.s. | | |

"Same F1" is a log-linear interpolation of A1's curve (reading aid, no test). Cross-budget contrasts: MLM at 1% vs BERT at 10% -0.38 [-1.15, 0.41], n.s.; MLM at 10% vs BERT at 30% +0.17, n.s.; MLM at 5% vs BERT at 30% -0.75 (p 0.027, Holm 0.11); MLM at 30% vs BERT at 100% -1.43, robust. Accuracy at 100%: +0.08, n.s.

**Verdicts:** H10.6 SUPPORTED (both new encoders help at 1% and 10%; range across the three encoders 0.83 at 1%, under the 1.5-point bound). H10.7 SUPPORTED (5%, 30%; no difference at 100%). H10.8 holds as predicted (no significant difference for MLM-1% vs BERT-10% and MLM-10% vs BERT-30%; not proof of equality, intervals about +-0.8).

## Q4: outlet shortcut and transfer (notebook 21, CPU)
Run by the owner on Kaggle (commit 4342e20, about 1 min of analysis; every results folder found). Files in `results/analysis_pass10/` (zip, unpacked tables incl. `all_verdicts.csv`, log renamed `notebook_21.log`); unpacked tables match the zip. `download (8).txt` in Downloads was an exact copy of the log and stays there. Numbers VERIFIED from those tables; paired bootstrap, 10,000 resamples, Holm within each table.

**Cue-mask shortcut gap** (random-mask minus cue-mask macro-F1, arm minus A1, or minus A7 for FGM arms): every row between -0.6 and +0.8, none significant (smallest p_holm 0.89). A20 -0.07 at 1%, -0.17 at 10%; the other two encoders +0.04 / +0.69 at 1%; the full stack A36 +0.83 at 1% (p 0.063 before Holm). **H10.9 not supported:** the pretrained models lean on outlet cue words exactly as much as plain BERT. This replaces the Pass 8 reading ([[Pass 8 Results]] caveat, now confirmed).

**iSarcasmEval AUC minus word count alone (68.7):** every pretrained arm is 11-21 points below (A20 49.8 / 54.1 at 1% / 10%; JEPA A22 52.6 / 57.3 best; full stack A36 47.8 / 54.6); all robust. **H10.10 SUPPORTED:** Headlines-only pretraining gives no transfer to iSarcasmEval; models sit near chance (AUC 50 = chance).

## Reading (INFERRED)
- **MLM pretraining does almost all the work.** Each old ingredient keeps 0-0.5 points of its +1.6 once the encoder is pretrained; pretraining absorbs them (robust negative interactions). Self-training stays healthy (92-96% correct pseudo-labels) but has little left to teach, because the MLM model already gets those headlines right.
- **The best recipe at 1% is MLM + helpers + self-training (88.9) or all three (88.8), about +1 point over MLM alone (robust)**; at 10% the stack adds +0.6 but not robustly. So the full recipe is "pretrain, then stack": +8.4 over plain BERT at 1% (80.4 -> 88.8), of which about 7.4 is pretraining.
- The "golden" scenario from the plan (self-training adds clearly on its own) did not happen; the stack does add a small, robust extra at 1%.
- **The MLM gain is not a lucky pretraining run.** Three independent pretrainings give +6.9 to +7.7 at 1% and +2.3 to +2.8 at 10%, every fine-tuning seed ahead. The single-seed limitation of Pass 8 is closed.
- **Label efficiency, stated carefully:** with MLM pretraining, 1% of the labels (228 headlines) reaches the score plain BERT needs about 8-10% for; the saving shrinks as labels grow (4.4x at 5%, 3.3x at 10%, 1.7x at 30%) and vanishes with all labels. A smooth, monotone curve, as Gururangan et al. 2020 and Shi et al. 2023 report for other tasks.
- **What MLM learns is in-domain, not general sarcasm:** same cue-word reliance as BERT and no transfer to iSarcasmEval. The paper must say the gain is a label-efficiency gain on Headlines, not better sarcasm understanding.
- Encoder seed 2 has a weaker frozen probe (84.2 vs 86.1-86.4) yet fine-tunes almost as well (87.3 vs 87.8-88.1): the probe is a rough guide, not a predictor of the fine-tuned score.

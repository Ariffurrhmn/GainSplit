---
tags: [results]
created: 2026-11-15
updated: 2026-11-15
---

# Pass 9 Results

Hub: [[00 - Project Context]]. Plan and predictions (fixed before running): [[Experiment Plan]] "Pass 9 plan". Question: do helper heads on L (H), FGM (F) and self-training (S) add up when combined, or overlap?

Run by the owner on Kaggle: notebook 18 (`pass9`, 40 runs + 3 re-runs, about 68 min, commit 9aa0147, 264 tests passed). `pass9_100` (A27 at 100%) was not run, so H9.5 is untested. Raw files: `results/pass9/` (zip, unpacked runs and logs, analysis tables, notebook log renamed `notebook_18.log`); unpacked folder matches the zip (checked with `diff -r`). Numbers below are VERIFIED from those tables unless marked. Method: paired bootstrap over test items, 10,000 resamples, mean over seeds 0-4, Holm within each table. All arms use the off-the-shelf `bert-base-uncased` encoder, no task-adaptive pretraining.

## Checks: PASS
A2, A6 and A7 at 1% seed 0 re-run with the changed code: test F1 and predictions identical to Pass 2 (also A1 at 100%, read from Pass 7).

## Test macro-F1 (means over 5 seeds)
| Arm | Ingredients | 1% | 10% |
|---|---|---|---|
| A1 | BERT | 80.42 | 88.21 |
| A2 | H | 82.05 | 88.43 |
| A7 | F | 82.00 | 89.76 |
| A6 | S | 82.04 | 88.81 |
| A27 | H+F | 82.97 | 89.48 |
| A28 | H+S | 83.06 | 89.23 |
| A29 | F+S | 83.86 | **90.28** |
| A30 | H+F+S | **84.60** | 89.88 |

## Planned contrasts (`contrasts_f1`, one Holm family)
| Contrast | 1% | 10% |
|---|---|---|
| A30 - A1 (whole stack vs BERT) | **+4.17** [3.44, 4.89], 5/5 | **+1.67** [1.07, 2.25], 5/5 |
| A30 - A7 (stack vs FGM alone) | **+2.59** [2.07, 3.11], 5/5 | +0.13 n.s., 3/5 |
| A27 - A7 (H on top of F) | **+0.97**, 5/5 | -0.28 n.s., 1/5 |
| A27 - A2 (F on top of H) | **+0.92**, 4/5 | **+1.05**, 5/5 |
| A28 - A2 (S on top of H) | **+1.01**, 5/5 | **+0.80**, 5/5 |
| A29 - A7 (S on top of F) | **+1.86**, 5/5 | +0.53, p_holm 0.039, 5/5 |
| A30 - A27 (S on top of H+F) | **+1.63**, 5/5 | +0.40, p 0.026, p_holm 0.077, 5/5 |

Bold: p_holm < 0.001.

## Do the gains add up? (`interactions`, `additivity`)
Interaction = how far a combination falls short of (negative) or exceeds (positive) the sum of its parts' gains.

| Interaction | 1% | 10% |
|---|---|---|
| H x F | -0.66 (p 0.036, p_holm 0.27) | -0.49 (p 0.049, p_holm 0.30) |
| H x S | -0.61 (p 0.034, p_holm 0.27) | +0.21 n.s. |
| F x S | +0.24 n.s. | -0.07 n.s. |
| all three vs sum of single gains | -0.65 n.s. | -0.68 (p 0.060) n.s. |

Share of the summed single gains each combination keeps: at 1%, H+F 79%, H+S 81%, F+S 108%, all three 86%; at 10%, 72%, 125%, 97%, 71%. No interaction survives Holm.

## Self-training health (`pseudo_health`)
Teacher dev F1 rises with the teacher recipe (1%: A1 79.3, A2 80.6, A7 81.2, A2+A7 81.6; 10%: 87.0-88.3). Pseudo-labels kept after the 0.9 threshold and class balancing: 16.8k-19.0k of 22,572 unlabeled headlines at 1%, 17.1k-17.8k of about 20.5k at 10%; no run fell back to L only (spot-checked A30_p1_s0). Accuracy of the pseudo-labels against the hidden true labels is not logged.

## Verdicts (from `verdicts.md`)
- H9.1 SUPPORTED: the stack beats BERT at 1% (+4.2) and 10% (+1.7).
- H9.2 SUPPORTED at both budgets as predicted: beats FGM alone at 1% (+2.6, every seed); no difference at 10%.
- H9.3 not supported: H x F and the three-way shortfall are negative, as predicted, but not significant after Holm (raw p 0.036 for H x F).
- H9.4 (no direction): no significant interaction; F x S adds up almost exactly.
- H9.5 untested (`pass9_100` not run).

## Reading (INFERRED)
- **At 1% the three gains mostly add up**: +4.2 for the stack vs +4.8 for the sum, about 86%. The small shortfall sits in the pairs that include helpers, in the predicted direction but not significant. FGM and self-training add up fully (they act on different things: robustness of fine-tuning vs more training data).
- **At 10% helpers stop helping once FGM is there** (A27 - A7 -0.28, 1/5 seeds ahead), so the best 10% recipe is F+S (A29, 90.28), not the full stack. Consistent with helpers and FGM both being regularisers, which matter less with more labels.
- **Pretraining beats the whole stack**: MLM-TAPT alone (A20, [[Pass 8 Results]]) reaches 87.84 at 1% and 91.02 at 10%, above the best stack here (84.60 / 90.28). The two notebooks share A1 and the test set, but this comparison is across notebooks and not bootstrapped.
- Pass 9 cannot say whether any ingredient adds **on top of** pretraining. A hint from Pass 8: FGM adds nothing on top of MLM-TAPT (A21 vs A20, -0.09 / +0.13 by means), so the regularisers may be absorbed; self-training, which adds data rather than regularising, is the open candidate.
- iSarcasmEval AUC stays at chance for every Pass 9 arm (46-53).

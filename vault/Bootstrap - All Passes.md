---
tags: [results]
created: 2026-11-10
updated: 2026-11-10
---

# Bootstrap - All Passes

Hub: [[00 - Project Context]]. Per-pass notes: [[Pass 1 Results]], [[Pass 2a Results]], [[Pass 2b Results]]. Test choice: [[Decisions]] (analysis notebook scope).

Owner ran `experiments/03_analysis.ipynb` locally on 2026-11-10 with `RESULTS_DIRS` = pass1 + pass2a + pass2b: 150 test-scored runs (45 + 95 + 10; the 9 `fgm_tuning` runs are dev only and left out). Paired bootstrap over test headlines, 10,000 resamples, RNG seed 0, difference in macro-F1 averaged over the 5 seeds, two-sided p. Output: `results/analysis/`. The Pass 1 contrasts reproduce the earlier run to the last digit.

Status: every number below is VERIFIED (read from `results/analysis/contrasts_test_macro_f1.csv`). The Holm column was computed from the same file over the 38 research contrasts (the 5 A1-A0 floor checks left out; p reported as 0 treated as 1/10,001); it is a robustness check, not the pre-set test.

## Contrasts with p < 0.06 (points; 95% interval; p; Holm-adjusted p)
| Contrast | Budget | Diff | 95% CI | p | Holm |
|---|---|---|---|---|---|
| A7 - A1 FGM | 10% | **+1.55** | [1.06, 2.03] | < 0.0001 | 0.004 |
| A7 - A1 FGM | 1% | **+1.58** | [1.02, 2.14] | < 0.0001 | 0.004 |
| A4 - A7 GAN vs FGM | 10% | -1.67 | [-2.19, -1.16] | < 0.0001 | 0.004 |
| A4 - A7 GAN vs FGM | 1% | -1.48 | [-2.07, -0.88] | < 0.0001 | 0.004 |
| A6 - A1 self-training | 1% | +1.62 | [1.10, 2.13] | < 0.0001 | 0.004 |
| A6 - A1 self-training | 10% | +0.60 | [0.19, 1.00] | 0.004 | 0.105 |
| A4 - A6 GAN vs self-training | 1% | -1.52 | [-2.13, -0.91] | < 0.0001 | 0.004 |
| A4 - A6 GAN vs self-training | 10% | -0.72 | [-1.19, -0.27] | 0.002 | 0.047 |
| A2 - A1 helpers on L only | 1% | +1.63 | [1.10, 2.16] | < 0.0001 | 0.004 |
| A3 - A2 adding U to helpers | 1% | -1.76 | [-2.30, -1.23] | < 0.0001 | 0.004 |
| A5 - A1 GAN + helpers | 1% | -2.30 | [-2.99, -1.63] | < 0.0001 | 0.004 |
| A5 - A3 | 1% | -2.17 | [-2.79, -1.55] | < 0.0001 | 0.004 |
| A5 - A4 | 1% | -2.40 | [-3.10, -1.72] | < 0.0001 | 0.004 |
| Interaction A5 - A4 - A3 + A1 | 1% | -2.27 | [-3.09, -1.45] | < 0.0001 | 0.004 |
| A4 - A1 GAN | 30% | -0.55 | [-0.94, -0.16] | 0.007 | 0.158 |
| A3 - A8 helper-label content | 10% | +0.57 | [0.13, 1.01] | 0.011 | 0.253 |
| A5 - A4 helpers on top of GAN | 30% | +0.47 | [0.07, 0.87] | 0.024 | 0.537 |
| A3 - A1 helpers | 10% | +0.42 | [-0.00, 0.85] | 0.051 | 1.000 |

Not significant anywhere: A3 - A1 at 1/5/30% (p 0.22-0.65), A4 - A1 at 1/5/10% (p 0.57-0.75), A5 - A1 at 5/10/30%, A2 - A1 and A3 - A2 at 10%, A3 - A8 at 1%, the interaction at 5/10/30%. A1 beats the TF-IDF floor A0 by 8.7-11.3 points at every budget (p < 0.0001).

## What is now VERIFIED
1. **FGM helps at both budgets** (+1.6 points), robust to Holm. It is the largest effect at 10%.
2. **The GAN never helps.** A4 - A1 is null at 1/5/10% and negative at 30% (-0.55, p 0.007, not robust to Holm). FGM and self-training both beat the GAN at 1% and 10%.
3. **Self-training helps at 1%** (+1.6, robust); at 10% +0.6 (p 0.004, not robust to Holm).
4. **At 1%, helper heads help only on the labeled rows:** A2 +1.6 over A1, and adding the unlabeled batch takes it away (A3 - A2 = -1.8). Both robust.
5. **GAN + helpers at 1% is harmful** (-2.3, negative interaction), robust.
6. **The helper heads on L + U (the seed paper's design) give no significant gain at any budget.** Direction is positive at 5/10/30% (+0.2 to +0.4). The label content matters against shuffled labels at 10% (A3 - A8 +0.57, p 0.011) but that does not survive Holm.

## Wording for the paper (INFERRED)
- Main claim: in our controlled setup, the gain attributed to GAN-BERT + multi-task learning does not come from the GAN; simple supervised adversarial training (FGM) and self-training give the gains the GAN does not, and the helper heads give at most a small, non-significant gain when trained on the unlabeled pool.
- Report the pre-set test (all contrasts, no correction) as the main analysis and the Holm column as a robustness check; say which results survive it.

## RQ3 (iSarcasmEval, `results/analysis/isarc_test.md`, VERIFIED numbers)
All arms near chance (ROC-AUC 47-58). TF-IDF A0 is below chance at every budget (43.9-45.7). The highest is A5 at 30% (57.6 ± 2.7) and A3 at 30% (55.0 ± 0.6). No bootstrap on iSarcasmEval yet (the notebook tests Headlines contrasts only); a transfer contrast is a Pass 3 item (IDEA).

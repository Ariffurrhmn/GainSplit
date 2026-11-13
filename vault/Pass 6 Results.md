---
tags: [results]
created: 2026-11-14
updated: 2026-11-14
---

# Pass 6 Results

Hub: [[00 - Project Context]]. Plan and predictions: [[Experiment Plan]] ("Pass 6 plan"). Previous passes: [[Pass 4 Results]], [[Pass 5 Results]].

Run by the owner on Kaggle: notebook 12 (GPU T4 x2, Save & Commit, code commit a57421e, 230 tests passed first, data matched `manifest.json`, stage-1 steps 720), 20/20 runs in 31.8 min. Raw files: `results/pass6/` (zip, unpacked runs and logs, `notebook_12.log`); all 141 unpacked files match the zip.

**Stage-1 check: PASS.** All 20 runs' stage-1 dev macro-F1 equals Pass 4's A15_p10 for the same seed exactly (VERIFIED from each `result.json` and the notebook log), so stage 1 is the A15 model, as designed.

Numbers are VERIFIED from `results/pass6/summary.csv` and the committed baselines. Readings are INFERRED: seed-paired means and sign counts only. No paired bootstrap has been run yet.

## Means over 5 seeds (points)
| Budget | Arm | Test macro-F1 | Cue-masked F1 | iSarcasmEval AUC |
|---|---|---|---|---|
| 10% | A1 | 88.21 | 82.64 (mask re-run) | 50.96 |
| 10% | A7 | 89.76 | 85.04 (Pass 3) | 51.47 |
| 10% | A15 (pairs only; = stage 1) | 55.47 | 54.75 | 76.22 |
| 10% | **A18** (pairs, then L) | **88.51** | 83.53 | 58.79 |
| 10% | **A19** (pairs, then L + FGM) | **89.64** | 85.05 | 57.87 |
| 1% | A1 | 80.42 | 74.30 (mask re-run) | 50.20 |
| 1% | A7 | 82.00 | 76.94 (mask re-run) | 47.76 |
| 1% | **A18** | 76.05 | 71.25 | 65.80 |
| 1% | **A19** | 79.18 | 74.43 | 63.23 |

Length-only baseline on the iSarcasmEval test: AUC 68.7 [64.8, 72.5] ([[Pass 4 Results]]).

## Seed-paired differences
| Contrast | Test F1 10% | Test F1 1% | iSarc AUC 10% | iSarc AUC 1% |
|---|---|---|---|---|
| A18 - A1 | +0.29 (4/5) | **-4.37** (0/5) | +7.83 (5/5) | +15.60 (5/5) |
| A19 - A7 | -0.12 (2/5) | **-2.82** (0/5) | +6.40 (5/5) | +15.47 (5/5) |
| A19 - A18 | +1.13 (5/5) | +3.13 (5/5) | -0.92 (1/5) | -2.58 (0/5) |

Cue-mask gap (test F1 minus cue-masked F1): A18 4.98 vs A1 5.57 at 10%, 4.80 vs 6.12 at 1%; A19 4.59 vs A7 4.72 at 10%, 4.75 vs 5.06 at 1%. Small, not tested.

## Against the predictions
| | Prediction | Outcome (INFERRED) |
|---|---|---|
| H6.1 | A18 within 1 point of A1 in-domain | **Half.** Holds at 10% (+0.3); fails at 1% (-4.4, every seed) |
| H6.2 | A18's AUC falls well below A15's 76.2; a gain only if clearly above 68.7 | **Falls, and no gain.** 58.8 (10%) and 65.8 (1%), both below word count alone |
| H6.3 | A19 within 1 point of A7 in-domain | **Half.** Holds at 10% (-0.1); fails at 1% (-2.8) |

## Reading (INFERRED)
- Two-stage training fixes the in-domain loss from Passes 4-5 at 10% labels: A18 and A19 match A1 and A7. It is the first pair arm that does not hurt Headlines at 10%.
- The price is the transfer: stage 2 overwrites most of what the pairs taught. iSarcasmEval AUC drops from 76 (stage 1) to 59 at 10% and 66 at 1%, below the length-only baseline at both budgets. The AUC gain over A1 is real on every seed but smaller than what word count alone gets.
- At 1% labels stage 2 has only 200 steps on 228 headlines; the model keeps more of the pairs (AUC 66) and pays for it in-domain (-4.4 / -2.8).
- So no setting tried in Passes 4-6 gives both: in-domain at A1's level and transfer above the length baseline. The trade-off moves with how much the final model is shaped by the pairs, but it does not go away.
- FGM keeps helping on top (+1.1 at 10%, +3.1 at 1%, 5/5 seeds), as in every pass.

## Open
- Paired bootstrap with Holm correction for Pass 5 and Pass 6 contrasts (CPU, not written).
- Paper framing: the pair results (Passes 4-6) as a negative finding with a clear mechanism (length shortcut, AI writing tics, transfer-vs-in-domain trade-off), next to the Passes 1-3 ablation. Instructor agreement needed ([[Open Questions]]).

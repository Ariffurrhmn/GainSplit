---
tags: [results]
created: 2026-11-14
updated: 2026-11-14
---

# Pass 7 Results

Hub: [[00 - Project Context]]. Plan and predictions: [[Experiment Plan]] ("Pass 7 plan"). Baselines: [[Pass 1 Results]] (A1 at 100%), [[Pass 6 Results]] (A19 at 1% and 10%).

Run by the owner on Kaggle: notebook 13 (GPU T4 x2, Save & Commit, code commit e21d7c4, 231 tests passed first, data matched `manifest.json`, stage-1 steps 720), 11/11 runs in 127.8 min. Raw files: `results/pass7/` (zip, unpacked runs and logs, `notebook_13.log`); all 78 unpacked files match the zip. `download (6).txt` in Downloads was an exact copy of the log and stays there.

**Re-run check: PASS.** `A1_p100_s0_rerun` test F1 93.8754 equals Pass 1's A1_p100_s0 (notebook log), and its `preds_test.csv`, `preds_dev.csv` and `preds_isarc_test.csv` are byte-identical to Pass 1's (VERIFIED, `cmp`). Pass 1 has no masked-test files, so the re-run's masked predictions are new, not a check.

**Stage-1 check: PASS.** All 5 A19 runs' stage-1 dev macro-F1 equals Pass 4's A15_p10 for the same seed exactly, after 720 steps (VERIFIED from each `result.json` and the notebook log).

Numbers are VERIFIED from `results/pass7/summary.csv` and `results/pass1/summary.csv`. Readings are INFERRED: seed-paired means and sign counts only. No paired bootstrap has been run yet.

## Means over 5 seeds at 100% labels (points; sd in brackets)
| Arm | Test accuracy | Test macro-F1 | Best seed acc. | Dev macro-F1 | Cue-masked F1 | iSarcasmEval AUC |
|---|---|---|---|---|---|---|
| A1 (Pass 1) | 93.35 (0.38) | 93.33 | 93.90 | 92.81 | 89.39 (seed 0 re-run only) | 51.66 |
| **A7** (FGM) | **93.90** (0.25) | **93.89** | **94.32** (seed 0) | 93.45 | 90.37 | 53.77 |
| **A19** (pairs, then all labels + FGM) | 93.67 (0.08) | 93.66 | 93.76 | 93.38 | 89.90 | 57.48 |

Seed paper reference: 93.88% accuracy on its own 80/20 split ([[Seed Paper - Irfan 2025]]); not like-for-like ([[Decisions]] 2026-11-11). Length-only baseline on the iSarcasmEval test: AUC 68.7 ([[Pass 4 Results]]).

## Seed-paired differences (test, points)
| Contrast | Accuracy | Macro-F1 | iSarc AUC |
|---|---|---|---|
| A7 - A1 | **+0.55** (5/5; +0.32 to +0.84) | +0.56 (5/5) | +2.11 |
| A19 - A1 | +0.33 (4/5) | +0.33 (4/5) | +5.82 |
| A19 - A7 | -0.23 (1/5 higher; -0.63 to +0.07) | -0.23 | +3.71 (5/5) |

Dev ranks the arms the same way (A7 > A19 > A1), so the order does not depend on test.

## Against the predictions
| | Prediction | Outcome (INFERRED) |
|---|---|---|
| H7.1 | A7 beats A1 by +0.3 to +0.8; about 93.7-94.2% | **Holds.** +0.55, every seed; 93.90% |
| H7.2 | A19 within 0.3 of A7 | **Holds, narrowly.** -0.23, lower on 4/5 seeds |
| H7.3 | iSarcasmEval AUC of A7 and A19 near chance | **Mostly.** A7 53.8; A19 57.5, a small pair imprint on every seed, far below the length baseline 68.7 |

## Reading (INFERRED)
- **94% is not reached as a 5-seed mean.** A7 gets 93.90% (sd 0.25); one seed (s0) reaches 94.32%. The paper reports the mean.
- FGM is the only ingredient that helps at every budget: +1.6 at 1% and 10% ([[Bootstrap - All Passes]]) and +0.55 at 100% here, every seed. The gain shrinks with more labels, as expected.
- A7's mean equals the seed paper's reported 93.88% (different split, single number), with plain BERT plus FGM and no GAN, helper heads or unlabeled data. That supports the ablation's message that the seed paper's extra parts are not what carries the score (careful wording needed: not like-for-like).
- The pair stage costs a little in-domain at full labels (-0.23 vs A7) and keeps a small transfer imprint (+3.7 AUC) that stays well below word count alone. Same trade-off as Passes 4-6, now small on both sides.
- A19 has the smallest spread (sd 0.08); one reading is that the stage-1 start makes fine-tuning more stable. Five seeds are too few to claim this.

## Open
- Paired bootstrap with Holm correction for the Pass 5, 6 and 7 contrasts (CPU notebook, being written).
- A1 at 100% has masked-test predictions for seed 0 only (from this re-run), so the cue-mask gap at 100% is not compared across arms.

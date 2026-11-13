---
tags: [results]
created: 2026-11-14
updated: 2026-11-14
---

# Pass 5 Results

Hub: [[00 - Project Context]]. Plan, predictions and tuning: [[Experiment Plan]] ("Pass 5 plan"). Previous pass: [[Pass 4 Results]]. Decisions: [[Decisions]].

Run by the owner on Kaggle: notebook 11 (GPU T4 x2, Save & Commit, code commit af2764d, 219 tests passed first, data matched `manifest.json`, frozen settings `pair_weight = 0.1`, `cleaned_drop_pct = 20`), 30/30 runs in 33.6 min. Raw files: `results/pass5/` (zip, unpacked runs and logs, `notebook_11.log`); all 211 unpacked files match the zip.

Numbers below are VERIFIED from `results/pass5/summary.csv` and the committed baselines (`pass1`, `pass2b`, `pass4`). Readings are INFERRED: seed-paired means and sign counts only. No paired bootstrap has been run yet.

## Means over 5 seeds (points)
| Budget | Arm | Test macro-F1 | Cue-masked F1 | iSarcasmEval AUC |
|---|---|---|---|---|
| 10% | A1 (labels only) | 88.21 | n/a | 50.96 |
| 10% | A7 (FGM) | 89.76 | n/a | 51.47 |
| 10% | A10 (Pass 4, weight 1) | 73.38 | 69.37 | 72.43 |
| 10% | A11 (Pass 4, FGM, weight 1) | 74.43 | 70.27 | 73.74 |
| 10% | **A10-weighted** (weight 0.1) | 77.91 | 73.39 | 69.12 |
| 10% | **A16** (weight 0.1 + cleaned) | 77.69 | 73.34 | 68.95 |
| 10% | **A17** (A16 + FGM) | 79.44 | 75.13 | 69.53 |
| 1% | A1 | 80.42 | n/a | 50.20 |
| 1% | A7 | 82.00 | n/a | 47.76 |
| 1% | A10 (Pass 4) | 63.95 | 61.60 | 71.75 |
| 1% | A11 (Pass 4) | 64.40 | 61.86 | 73.13 |
| 1% | **A10-weighted** | 66.56 | 63.84 | 69.52 |
| 1% | **A16** | 66.18 | 63.51 | 68.93 |
| 1% | **A17** | 66.89 | 64.16 | 70.63 |

Length-only baseline on the iSarcasmEval test (word count alone, [[Pass 4 Results]] "iSarcasmEval length-shortcut check"): AUC 68.7 [64.8, 72.5].

## Seed-paired differences (test macro-F1)
| Contrast | 10% | 1% |
|---|---|---|
| Weight alone: A10-weighted - A10 | +4.53 (5/5) | +2.61 (5/5) |
| Still behind labels only: A10-weighted - A1 | -10.30 (0/5) | -13.86 (0/5) |
| Cleaning on top of the weight: A16 - A10-weighted | -0.22 (3/5) | -0.38 (2/5) |
| FGM on top: A17 - A16 | +1.74 (5/5) | +0.70 (5/5) |
| A17 - A7 | -10.32 (0/5) | -15.12 (0/5) |

iSarcasmEval AUC, seed-paired: the weight costs 3.3 (10%) and 2.2 (1%) points against Pass 4's A10 (0/5 seeds up). Cleaning changes AUC by -0.2 / -0.6. Every Pass 5 arm stays 18-23 points above A1/A7.

## Against the predictions
| | Prediction | Outcome (INFERRED) |
|---|---|---|
| H5.1 | some weight < 1 recovers most of the in-domain loss | **Rejected.** Weight 0.1 recovers 4.5 of 14.8 points at 10% (about 30%) and 2.6 of 16.5 at 1%; still 10-14 points below A1 |
| H5.2 | cleaning adds a further recovery | **Rejected.** A16 - A10-weighted is -0.2 / -0.4, mixed signs |
| H5.3 | transfer shrinks as the weight drops | **Supported.** AUC falls 2-4 points, and every Pass 5 arm (68.9-70.6) now sits at the length-only baseline (68.7, inside its interval) |

## Reading (INFERRED)
- Down-weighting the pairs trades transfer for in-domain accuracy and does not escape the trade-off. At weight 0.1 the model is still far below A1 on Headlines, and its iSarcasmEval ranking is no better than counting words.
- Removing the 20% of pairs that lean most on AI writing habits does nothing measurable, on either test. Whatever hurts the in-domain score is spread through the pairs, not concentrated in the worst ones.
- FGM is the only addition that helps consistently (+1.7 at 10%, 5/5 seeds), matching its effect in Passes 2-4.
- Taken with [[Pass 4 Results]]: mixing these generated pairs into training, at any weight tried, does not beat labels-only training in-domain. The transfer gain is mostly length, and what is left beyond length disappears once the pairs are down-weighted.

## Open
- Paired bootstrap with Holm correction for the contrasts above (CPU; a Pass 5 extension of notebook 09 is not written yet).
- Which of the remaining ideas in [[Research Ideas]] "After Pass 4" is still worth GPU time: two-stage training (pairs first, then labels only), pair-ranking loss, or length-matched pairs / regeneration with an "avoid these words" prompt.

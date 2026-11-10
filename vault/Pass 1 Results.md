---
tags: [results]
created: 2026-11-10
updated: 2026-11-10
---

# Pass 1 Results

Hub: [[00 - Project Context]]. Plan: [[Experiment Plan]]. Choices: [[Decisions]].

Run by the owner on Kaggle (GPU T4 x2, commit c0a44a5), 45/45 runs, 114 GPU-minutes in total. Raw files: `results/pass1/` (summary.csv, per-run result.json, predictions for dev, test and iSarcasmEval test). Setup: lr 5e-5, w = 0.3, steps = max(200, 10 passes over L), batch 32, seeds 0-4. First use of the test set; nothing was tuned after it.

Status of the numbers below: VERIFIED as computed from `results/pass1/summary.csv` (mean and sd over 5 seeds). Paired bootstrap over test items run by the owner on 2026-11-10 (`experiments/03_analysis.ipynb`, locally, 10,000 resamples, RNG seed 0; output in `results/analysis/`). Significance statements below are VERIFIED against that output where marked.

## Headlines test, macro-F1 (mean, sd over 5 seeds)
| Budget | A1 BERT | A3 + helpers | A4 GAN-BERT | A5 GAN + helpers |
|---|---|---|---|---|
| 1% (228) | 0.804 (0.012) | 0.803 (0.018) | 0.805 (0.016) | 0.781 (0.027) |
| 10% (2,280) | 0.882 (0.003) | **0.886** (0.005) | 0.881 (0.004) | 0.884 (0.003) |
| 100% (22,800) | 0.933 (0.004) | - | - | - |

ROC-AUC follows the same order at 10% (A3 0.950, A1 0.949, A5 0.947, A4 0.940). At 1% all three U-arms have a slightly higher AUC than A1 (0.892-0.894 vs 0.887) while macro-F1 does not move: they rank better but the 0.5 threshold is off. INFERRED.

## Paired differences in test macro-F1 (same seed, same labeled set), points
| Contrast | 1%: mean (seeds better) | 10%: mean (seeds better) |
|---|---|---|
| A3 - A1 (helpers) | -0.13 (3/5) | **+0.42 (4/5)** |
| A4 - A1 (GAN) | +0.10 (2/5) | -0.13 (1/5) |
| A5 - A1 (both) | -2.30 (1/5) | +0.23 (5/5) |
| A5 - A3 (GAN on top of helpers) | -2.17 (1/5) | -0.19 (1/5) |
| A5 - A4 (helpers on top of GAN) | -2.40 (0/5) | +0.36 (5/5) |
| Interaction A5 - A4 - A3 + A1 | -2.27 | -0.06 |

## Paired bootstrap, test macro-F1 (points; 95% interval; two-sided p)
Source: `results/analysis/contrasts_test_macro_f1.csv`. VERIFIED (read from the file). Same test resample across the 5 seeds, difference averaged over seeds. p = 0.0000 means no resample out of 10,000 crossed zero (p < 0.0001).

| Contrast | 1% | 10% |
|---|---|---|
| A3 - A1 (helpers) | -0.13 [-0.70, 0.45], p 0.65 | +0.42 [-0.00, 0.85], **p 0.051** |
| A4 - A1 (GAN) | +0.10 [-0.51, 0.70], p 0.75 | -0.13 [-0.58, 0.32], p 0.57 |
| A5 - A1 (both) | **-2.30 [-2.99, -1.63], p < 0.0001** | +0.23 [-0.19, 0.65], p 0.28 |
| A5 - A3 | **-2.17 [-2.79, -1.55], p < 0.0001** | -0.19 [-0.58, 0.20], p 0.36 |
| A5 - A4 | **-2.40 [-3.10, -1.72], p < 0.0001** | +0.36 [-0.08, 0.81], p 0.11 |
| Interaction A5 - A4 - A3 + A1 | **-2.27 [-3.09, -1.45], p < 0.0001** | -0.06 [-0.64, 0.52], p 0.85 |

No correction for multiple contrasts (12 tests); all are reported ([[Decisions]]).

## Reading
- **RQ1, GAN: no detectable effect.** VERIFIED: A4 - A1 is not significant at 1% or 10%, and the 95% intervals rule out a GAN gain larger than about 0.7 points (1%) and 0.3 points (10%). Matches H1 at 10%; at 1% it is also null, which is stronger than H1 predicted.
- **RQ1, helpers: a small, borderline gain at 10% only.** VERIFIED: A3 - A1 at 10% is +0.42 points with a 95% interval from -0.00 to +0.85 and p = 0.051, so it does **not** clear the 5% level; at 1% there is no effect. INFERRED: the direction is consistent (4/5 seeds, and A5 - A4 = +0.36 on 5/5 seeds, p = 0.11), so the paper should call it "a small gain of at most about 0.9 points, not significant at the 5% level", not a clear win. The Pass 2 budgets 5% and 30% for A3 test whether the same direction holds elsewhere.
- **Combination at 1% is harmful.** VERIFIED: A5 loses 2.3 points against A1 (p < 0.0001) and against both single-component arms; the interaction term is -2.3 points (p < 0.0001). So the two components that are each harmless at 1% hurt together. The GAN itself trained normally in those runs (p_fake on real ~0.01, on fake ~0.42 at the last step, training sarcasm loss ~0.02), so it is not a GAN collapse. Likely cause (INFERRED, not tested): with 228 labels the sarcasm term is a small share of a loss that also holds the GAN and two helper terms on 32 U rows per step.
- **GAN at 1% may not have settled.** INFERRED from `results/analysis/fig_gan_curves.png`: at 1% runs have 200 steps and the mean fake-probability on generated vectors is still rising (0.33 at step 100 to 0.43 at step 200), while at 10% it levels off near 0.6-0.65 by step 400. A reviewer could say the 1% GAN arms were under-trained. Answer in the paper: equal steps for every arm is a design choice ([[Decisions]]); a longer-run check at 1% is an optional extra (IDEA).
- **Effect size vs data:** going from 10% to 100% labels gives +5.1 points; the best method effect at 10% is +0.4. Worth stating plainly in the paper.
- **RQ3, transfer:** every arm is at chance on iSarcasmEval test (ROC-AUC 0.49-0.54, sarcastic-class F1 0.11-0.18; 200 of 1,400 items are sarcastic; table in `results/analysis/isarc_test.md`, VERIFIED). Even A1 at 100% (AUC 0.517) does not transfer. So no arm transfers. This supports the outlet-style shortcut reading ([[Risks]]) but does not prove it; the shortcut checks in Pass 3 test it directly.
- Seed noise: sd 0.003-0.005 at 10%, 0.012-0.027 at 1%. The bootstrap intervals are about +-0.4 points at 10% and +-0.6 at 1%: the smallest effects this design can detect.

## Figures
`results/analysis/fig_f1_vs_budget.(png|pdf)` (Pass 1 budgets only; Pass 2 adds 5% and 30%) and `fig_gan_curves.(png|pdf)`.

## Cost (measured)
A1: 0.5 / 1.4 / 12.7 min at 1 / 10 / 100%. U-arms: about 0.7 min at 1% and 2.1 min at 10%.

---
tags: [results]
created: 2026-11-10
updated: 2026-11-10
---

# Pass 1 Results

Hub: [[00 - Project Context]]. Plan: [[Experiment Plan]]. Choices: [[Decisions]].

Run by the owner on Kaggle (GPU T4 x2, commit c0a44a5), 45/45 runs, 114 GPU-minutes in total. Raw files: `results/pass1/` (summary.csv, per-run result.json, predictions for dev, test and iSarcasmEval test). Setup: lr 5e-5, w = 0.3, steps = max(200, 10 passes over L), batch 32, seeds 0-4. First use of the test set; nothing was tuned after it.

Status of the numbers below: VERIFIED as computed from `results/pass1/summary.csv` (mean and sd over 5 seeds). Statements about significance are INFERRED from seed-level paired differences only; the paired bootstrap over test items is still to run (analysis notebook, [[Experiment Plan]] roadmap step 4).

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

## Reading (INFERRED until the bootstrap runs)
- **RQ1, GAN:** no gain at 1% or 10%. A4 is within 0.1-0.2 points of A1 at both budgets and wins on 1-2 of 5 seeds. Matches prediction H1 for 10%; at 1% it is also null, which is stronger than H1 predicted.
- **RQ1, helpers:** a small gain at 10% (+0.4 points, 4/5 seeds) and none at 1%. All of A5's gain over A1 at 10% comes from the helpers (A5 - A4 = +0.36 on every seed, A5 - A3 = -0.19).
- **Combination at 1% is harmful:** A5 loses 2.3 points against A1 with the largest spread (sd 0.027; seed 4: -6.6 points). The GAN itself trained normally in those runs (p_fake on real ~0.01, on fake ~0.42 at the last step, training sarcasm loss ~0.02), so it is not a GAN collapse. Likely cause: with 228 labels the sarcasm term is a small share of a loss that also holds the GAN and two helper terms on 32 U rows per step. Not tested.
- **Effect size vs data:** going from 10% to 100% labels gives +5.1 points; the best method effect at 10% is +0.4. Worth stating plainly in the paper.
- **RQ3, transfer:** every arm is at chance on iSarcasmEval test (ROC-AUC 0.49-0.54, sarcastic-class F1 0.11-0.18; 200 of 1,400 items are sarcastic). Even A1 at 100% (AUC 0.517) does not transfer. So the RQ3 question "does any arm drop less?" has the answer: all drop to chance; no arm transfers. This supports the outlet-style shortcut reading ([[Risks]]) but does not prove it; the shortcut checks in Part 3 test it directly.
- Seed noise: sd 0.003-0.005 at 10%, 0.012-0.027 at 1%. At 1% a 5-seed design can only detect differences of about 2 points; the paired bootstrap pools test items and seeds and should be tighter.

## Cost (measured)
A1: 0.5 / 1.4 / 12.7 min at 1 / 10 / 100%. U-arms: about 0.7 min at 1% and 2.1 min at 10%.

---
tags: [results]
created: 2026-11-10
updated: 2026-11-10
---

# Pass 2b Results

Hub: [[00 - Project Context]]. Plan: [[Experiment Plan]]. Earlier passes: [[Pass 1 Results]], [[Pass 2a Results]]. Epsilon choice: [[Decisions]].

Run by the owner on Kaggle (GPU T4 x2, commit 62d0020), 10/10 runs: A7 (BERT + FGM on the word embeddings, epsilon 0.5) at 1% and 10%, seeds 0-4, test scored. Raw files: `results/pass2b/`. Same frozen settings as the other passes.

**Bootstrap done 2026-11-10: significance, intervals and which readings are now VERIFIED are in [[Bootstrap - All Passes]].** Status: means, sds and per-seed differences VERIFIED as computed from the three `summary.csv` files.

## Checks (VERIFIED)
- 10 runs, 4 files each, epsilon 0.5 in every run, steps 200 (1%) and 720 (10%) as for every arm.
- The 10% runs for seeds 0-2 reproduce the `fgm_tuning` dev scores for epsilon 0.5 exactly (0.8816, 0.8784, 0.8855): deterministic.

## Headlines test, macro-F1 in points (mean, sd over 5 seeds)
| Arm | 1% | 10% |
|---|---|---|
| A1 BERT | 80.42 (1.24) | 88.21 (0.31) |
| A2 helpers, L only | 82.05 (0.64) | 88.43 (0.41) |
| A3 helpers, L + U | 80.29 (1.84) | 88.64 (0.51) |
| A4 GAN-BERT | 80.52 (1.59) | 88.09 (0.40) |
| A6 self-training | 82.04 (0.75) | 88.81 (0.30) |
| **A7 FGM** | 82.00 (0.61) | **89.76 (0.15)** |

A7 test ROC-AUC: 90.1 (1%), 96.2 (10%).

## Seed-paired differences, points: mean (seeds better)
| Contrast | 1% | 10% |
|---|---|---|
| A7 - A1 FGM | +1.58 (4/5) | **+1.55 (5/5)** |
| A4 - A7 GAN vs FGM | -1.48 (1/5) | **-1.67 (0/5)** |
| A7 - A6 FGM vs self-training | -0.04 (2/5) | +0.95 (5/5) |
| A7 - A3 FGM vs helpers | +1.71 (5/5) | +1.12 (5/5) |
| A7 - A2 FGM vs helpers on L | -0.05 (1/5) | +1.33 (5/5) |

## Reading (seed-level; see [[Bootstrap - All Passes]] for what the bootstrap confirms)
- **FGM is the largest method effect in the study.** At 10% it gains +1.55 points over A1 on every seed (smallest seed gain +1.33) with the lowest spread of any arm (sd 0.15); about four times the helper gain. At 1% it gains +1.6, level with self-training (A6) and helpers on L only (A2).
- **Adversarial training helps; the GAN does not.** A7 uses no unlabeled data at all, only small worst-case pushes on the labeled examples, and beats the GAN-BERT arm by 1.5-1.7 points. So the GAN's lack of gain is not because "adversarial ideas do not help here". The useful part is perturbation-based regularisation of the encoder; the generator / discriminator machinery and the unlabeled pool add nothing. This is the cleanest answer to RQ1 so far.
- **At 1% three different routes reach about 82 points** (A2, A6, A7), all +1.6 over A1, while the arms that put unlabeled-side losses on a tiny L (A3, A5) do not. The ceiling at 1% may be the 228 labels themselves.
- **Fairness caveat for reviewers:** A7 had its epsilon picked from 3 values on dev, A3/A5 had w picked from 3 values, A6 and the GAN arms used fixed literature settings with no tuning. A reviewer may say the GAN was under-tuned. Answer: the GAN pilot was healthy ([[Decisions]]), and the GAN is never above A1 at any budget, so tuning would have to find a gain the GAN never shows. Still worth one sentence in the limitations ([[Risks]]).
- **RQ3:** A7 does not transfer either (iSarcasmEval ROC-AUC 47.8 at 1%, 51.5 at 10%).

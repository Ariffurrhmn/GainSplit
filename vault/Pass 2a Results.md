---
tags: [results]
created: 2026-11-10
updated: 2026-11-10
---

# Pass 2a Results

Hub: [[00 - Project Context]]. Plan: [[Experiment Plan]]. Choices: [[Decisions]]. Earlier pass: [[Pass 1 Results]].

Run by the owner on Kaggle (GPU T4 x2, commit bfe964d), 104/104 runs. Raw files: `results/pass2a/` (summary.csv, per-run result.json, predictions). Same frozen settings as Pass 1 (lr 5e-5, w = 0.3, steps = max(200, 10 passes over L), batch 32, seeds 0-4).

**Bootstrap done 2026-11-10: significance, intervals and which readings are now VERIFIED are in [[Bootstrap - All Passes]].** Status: means, sds and per-seed differences are VERIFIED as computed from `results/pass1/summary.csv` + `results/pass2a/summary.csv`.

## Sanity checks (VERIFIED)
- 104 runs; every planned (arm, budget) cell has 5 seeds; `fgm_tuning` has 9 dev-only runs.
- A2 runs use no U batch (`uses_u` false, 22,572 unlabeled rows unused at 1%); A8 runs use U.
- A6's retrained teacher at 10% reproduces Pass 1's A1 dev scores exactly (seeds 0-2: 0.8645, 0.8639, 0.8734), so training is deterministic across sessions.

## FGM epsilon (A7, p = 10%, dev only, seeds 0-2)
| epsilon | mean dev macro-F1 (sd) | seeds |
|---|---|---|
| 0.5 | 0.8818 (0.0036) | 0.8816, 0.8784, 0.8855 |
| 1.0 | 0.8840 (0.0057) | 0.8794, 0.8822, 0.8904 |
| 2.0 | 0.8796 (0.0055) | 0.8754, 0.8775, 0.8858 |

Chosen: **0.5** (DELEGATED, [[Decisions]]). Every epsilon scores about 1.5 points above A1 on dev (A1 seeds 0-2: 0.8673). If that holds on test in `pass2b`, FGM is the largest method effect at 10% (INFERRED; dev is tuning evidence only).

## Headlines test, macro-F1 in points (mean, sd over 5 seeds); Pass 1 and 2a pooled
| Arm | 1% | 5% | 10% | 30% | 100% |
|---|---|---|---|---|---|
| A0 TF-IDF + LR | 69.71 (0.68) | 75.34 (0.63) | 78.14 (0.63) | 81.07 (0.37) | 84.66 (0.00) |
| A1 BERT | 80.42 (1.24) | 86.61 (0.54) | 88.21 (0.31) | 90.85 (0.21) | 93.33 (0.38) |
| A2 helpers, L only | **82.05** (0.64) | - | 88.43 (0.41) | - | - |
| A3 helpers, L + U | 80.29 (1.84) | 86.83 (0.71) | 88.64 (0.51) | 91.09 (0.43) | - |
| A4 GAN-BERT | 80.52 (1.59) | 86.51 (1.00) | 88.09 (0.40) | 90.30 (0.23) | - |
| A5 GAN + helpers | 78.12 (2.74) | 86.62 (0.53) | 88.44 (0.32) | 90.77 (0.18) | - |
| A6 self-training | **82.04** (0.75) | - | **88.81** (0.30) | - | - |
| A8 shuffled helpers | 80.03 (1.66) | - | 88.06 (0.28) | - | - |

A0 at 100% has sd 0 because the full training set is the same for every seed and logistic regression is deterministic.

## Seed-paired differences, points: mean (seeds better)
| Contrast | 1% | 5% | 10% | 30% |
|---|---|---|---|---|
| A3 - A1 helpers | -0.13 (3/5) | +0.21 (4/5) | +0.42 (4/5) | +0.24 (4/5) |
| A4 - A1 GAN | +0.10 (2/5) | -0.11 (4/5) | -0.13 (1/5) | **-0.55 (0/5)** |
| A5 - A1 both | -2.30 (1/5) | +0.00 (2/5) | +0.23 (5/5) | -0.08 (2/5) |
| A5 - A4 helpers on top of GAN | -2.40 (0/5) | +0.11 (2/5) | +0.36 (5/5) | +0.47 (5/5) |
| A2 - A1 helpers on L only | **+1.63 (5/5)** | - | +0.21 (3/5) | - |
| A3 - A2 adding U to the helpers | **-1.76 (0/5)** | - | +0.21 (3/5) | - |
| A8 - A1 shuffled helper labels | -0.40 (3/5) | - | -0.15 (2/5) | - |
| A3 - A8 helper-label content | +0.26 (3/5) | - | **+0.57 (5/5)** | - |
| A6 - A1 self-training | **+1.62 (5/5)** | - | **+0.60 (5/5)** | - |
| A0 - A1 | -10.71 (0/5) | - | -10.08 (0/5) | - |

## Reading (seed-level; see [[Bootstrap - All Passes]] for what the bootstrap confirms)
- **Helper heads, budget curve:** small positive direction at 5%, 10% and 30% (+0.2 to +0.4, 4/5 seeds each), none at 1%. Consistent with the borderline 10% result in Pass 1, so "small but consistent" is the likely wording.
- **What drives the helper gain at 10%: the label content.** Shuffled helper labels (A8) fall back to A1 and lose to A3 on every seed (+0.57). So it is not just an extra loss term or regulariser.
- **At 1% the unlabeled rows hurt.** Helper heads trained on the 228 labeled headlines only (A2) gain +1.6 points on every seed; adding the unlabeled batch to the helper loss (A3) removes that gain on every seed. A2 and A3 differ in two ways (U rows in the helper loss, and a U batch in every step), so the two are not separated. Same pattern as the A5 harm at 1% ([[Pass 1 Results]]): when L is tiny, unlabeled-side losses pull the encoder away from the sarcasm task. This reframes RQ1: at 1% the helper heads help only without the unlabeled pool.
- **Self-training works, the GAN does not.** A6 beats A1 on every seed at 1% (+1.6) and 10% (+0.6) and is the best arm at 10%. The GAN gives nothing at 1-10% and loses 0.55 points at 30% (0/5 seeds). So the null GAN result sits next to a working semi-supervised baseline on the same data, which answers "maybe SSL just cannot help here".
- **A6 pseudo-labels:** at the 0.9 threshold the teacher is confident on 95% of U at 1% (21,394-21,539 of 22,572) and 97% at 10% (19,939-20,032 of 20,520). After class balancing 16,046-20,464 (1%) and 17,326-18,180 (10%) are kept; no run fell back to L only. The teacher's dev macro-F1 is 0.78-0.81 at 1%, so roughly one pseudo-label in five is likely wrong there, and self-training still gains 1.6 points. The 0.9 threshold filters little because fine-tuned BERT is over-confident.
- **A5 harm is specific to 1%:** at 5% and 30% A5 is level with A1.
- **A0 floor:** about 10 points below A1 at 1% and 10%.
- **RQ3, iSarcasmEval ROC-AUC (x100, mean over seeds):** A0 43.9-45.7 (below chance at every budget), A1 50.0-51.7, A3 51.0 → 54.9 and A5 48.7 → 57.6 rising with budget, A6 49.4-50.7, A8 48.4-50.2. The helper arms may transfer slightly better at 30%; needs the bootstrap. The TF-IDF floor being below chance suggests that the words that mark sarcasm in News Headlines point the wrong way on tweets, consistent with the outlet-style shortcut reading ([[Risks]]).

## Cost (measured, minutes per run)
A0 under 0.1. A1 0.8 / 4.0 at 5 / 30%. A3, A4, A5 about 1.2 / 6.0 at 5 / 30%. A2 as A1. A8 as A3. A6 1.4 / 3.7 at 1 / 10% (includes retraining the teacher). A7 about 2.2 at 10%.

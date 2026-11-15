---
tags: [results]
created: 2026-11-15
updated: 2026-11-15
---

# Pass 8 Results

Hub: [[00 - Project Context]]. Plan and predictions (fixed before running): [[Experiment Plan]] "Pass 8 plan". Pass 8b (JEPA, notebook 16) added 2026-11-15, run without the 100% arms (`RUN_FULL100 = False`).

Run by the owner on Kaggle: notebook 15 (`pass8_mlm` + `pass8_mlm_100`, 25 runs, 106.3 min), notebook 17 (`pass8_diff`, 31 runs) and notebook 16 (`pass8_jepa`, 20 runs + 1 re-run, about 61 min, commit 9aa0147). Raw files: `results/pass8_mlm/`, `results/pass8_diff/` and `results/pass8_jepa/` (zip, unpacked runs and logs, analysis tables, notebook log renamed `notebook_15.log` / `notebook_17.log` / `notebook_16.log`); unpacked folders match the zips (checked with `diff`). Numbers below are VERIFIED from those tables unless marked. Method: paired bootstrap over test items, 10,000 resamples, mean over seeds 0-4, Holm within each notebook's table.

## Pass 8a: task-adaptive pretraining with MLM (notebook 15)
**Checks: PASS.** A1_p100_s0 re-run identical to Pass 1.

**Pretraining** (10,000 steps, 35.2 min): dev MLM loss 5.85 -> 3.68 (train loss at the end 1.44). Frozen linear probe, dev macro-F1: 76.2 -> 86.4 with 1% labels, 80.6 -> 87.2 with 10%.

| Contrast (test macro-F1, points) | 1% | 10% | 100% |
|---|---|---|---|
| A20 - A1 (MLM-TAPT, then plain BERT) | **+7.42** [6.55, 8.32], 5/5 seeds | **+2.80** [2.09, 3.53], 5/5 | |
| A21 - A7 (MLM-TAPT, then FGM) | **+5.75** [4.95, 6.56], 5/5 | **+1.39** [0.81, 1.98], 5/5 | -0.25 [-0.69, 0.17], n.s. |
| A21 - A1 | +7.33 | +2.94 | |

All robust (p_holm < 0.001) except the 100% row. Means: A20 87.84 at 1% (A1 80.42, A7 82.00); A21 91.15 at 10% (A1 88.21, A7 89.76); A21 93.64 at 100% (A7 93.89). Accuracy at 100%: A21 - A7 -0.25, n.s.

Verdicts: H8a.1 SUPPORTED (1% and 10%), H8a.2 SUPPORTED (1% and 10%), H8a.3 not supported (no difference at 100%).

**iSarcasmEval AUC** at 10%: A20 - A1 +3.2 (p 0.022), A21 - A7 +3.2 (p_holm 0.010); means still 54-55, near chance.

**Cue-mask check** (means over seeds, computed from the committed run tables 2026-11-15, no bootstrap): test F1 minus cue-masked test F1 is 3.8 (A20) and 3.9 (A21) at 1%, against 6.1 (A1) and 5.1 (A7); at 10%, 4.1 / 3.9 against 5.6 / 4.7. The cue-masked score itself is 84.1 (A20) vs 74.3 (A1) at 1%. So the TAPT gain survives masking the outlet cue words.

## Pass 8b: task-adaptive pretraining with JEPA (notebook 16)
**Checks: PASS.** A1_p100_s0 re-run identical to Pass 1. No collapse (target spread 0.94 at the end).

**Pretraining** (10,000 steps, 43.7 min): dev JEPA loss 1.32 -> 0.38. Frozen linear probe, dev macro-F1: 76.2 -> 78.5 with 1% labels, 80.6 -> 79.8 with 10% (MLM: 86.4 / 87.2). So the JEPA encoder's frozen features barely improve, unlike MLM's.

| Contrast (test macro-F1, points) | 1% | 10% |
|---|---|---|
| A22 - A1 (JEPA-TAPT, then plain BERT) | **+5.67** [4.72, 6.63], 5/5 seeds | **+1.63** [0.85, 2.42], 5/5 |
| A23 - A7 (JEPA-TAPT, then FGM) | **+4.30** [3.39, 5.21], 5/5 | +0.23 [-0.45, 0.93], n.s., 4/5 |
| A22 - A20 (JEPA vs MLM, plain) | **-1.75** [-2.59, -0.91], 0/5 | **-1.17** [-1.84, -0.52], 0/5 |
| A23 - A21 (JEPA vs MLM, FGM) | **-1.45** [-2.25, -0.66], 0/5 | **-1.16** [-1.78, -0.53], 0/5 |

Bold rows robust (p_holm <= 0.001). Means: A22 86.09 / 89.85, A23 86.30 / 89.99 (1% / 10%); A20 87.84 / 91.02.

Verdicts: H8b.1 SUPPORTED (A22 > A1 at 1%, also at 10%); H8b.2 SUPPORTED at 1% (no difference at 10%); H8b.3 (no direction predicted): **JEPA loses to MLM** at both budgets, on every seed, with and without FGM.

**iSarcasmEval AUC** at 10%: A22 - A1 +6.3 (p_holm < 0.001), A23 - A7 +4.4 (p_holm 0.001), A22 - A20 +3.1 (p_holm 0.010). Means 55.8-57.3, still near chance. This breaks the plan's "no arm moves it beyond 2-3 points" prediction, but from a chance-level start.

**Cue-mask check** (means over seeds, from `summary.md`, no bootstrap): test F1 minus cue-masked F1 is 4.2 (A22) and 4.3 (A23) at 1%, 4.4 / 4.2 at 10%: between MLM (3.8-4.1) and plain BERT (5.6-6.1).

## Pass 8c: diffusion model over [CLS] vectors (notebook 17)
**Checks: PASS.** A1_p10_s0 re-run identical to Pass 1 (the changed training loop leaves old arms alone).

| Contrast (points) | 1% | 10% |
|---|---|---|
| A24 - A4 (diffusion fakes vs GAN fakes) | -0.25, n.s. | -0.11, n.s. |
| A24 - A1 | -0.15, n.s. | -0.24, n.s. |
| A25 - A1 (diffusion consistency on U) | **-4.82** [-5.56, -4.09], robust | **-0.85** [-1.31, -0.40], robust |
| A26 - A1 (plain-noise consistency) | **-3.99**, robust | **-0.73**, robust |
| A25 - A26 (denoising vs noise) | -0.83, p 0.027, Holm 0.16 | -0.12, n.s. |

Verdicts: H8c.1 no difference; H8c.2 held as predicted (A24 does not beat A1; the cell prints "not supported" because it tests "beats"); H8c.3 opposite, significant (consistency hurts); H8c.4 not supported.

Health: the k+1 discriminator spots diffusion fakes easily (p_fake on fakes 0.78 at 1%, 0.97 at 10%), as with the GAN. In A25/A26 the consistency loss is near zero (flip rate 0.3%), yet both lose F1, with large seed spread at 1% (sd 4.8 and 5.6; one A26 seed -15).

## Reading (INFERRED)
- **MLM-TAPT is the largest gain in the project**: +7.4 at 1% and +2.8 at 10% over plain BERT, far above FGM, helpers or self-training (about +1.6 each at 1%). It uses only the unlabeled training texts, so it is a fair semi-supervised use of U, the first one in this study that works. It fades with every label (no gain at 100%), as expected for a low-resource method.
- The cue-mask numbers argue against an outlet-word shortcut: TAPT models lose less when cue words are masked, not more. Still to do: a bootstrap of the gap contrast.
- TAPT is a known method (Gururangan et al. 2020), so the novelty is the controlled comparison, not the method: in the same testbed, plain self-supervised use of U beats every GAN, helper, adversarial and LLM-data route.
- **The objective matters, and MLM wins.** JEPA-TAPT gives most of the gain (+5.7 of MLM's +7.4 at 1%) but trails MLM by 1.2-1.7 points on every seed. So the JEPA variant is not a novel-method headline; it is a second data point that "continue pretraining on U" is what works, with word prediction better than hidden-state prediction. Its frozen probe barely moves while fine-tuning still gains +5.7, so JEPA helps as a starting point for fine-tuning rather than by giving better ready-made features.
- One pretraining seed per objective: the MLM-vs-JEPA gap (1.2-1.7) is measured over fine-tuning seeds only; a second pretraining seed would show whether it holds.
- FGM adds nothing on top of MLM-TAPT (A21 vs A20: -0.09 at 1%, +0.13 at 10%, from the means; not a planned contrast, no bootstrap). It does add a little on top of JEPA at 1% (A23 - A22 +0.2, n.s. by eye). Pretraining absorbs FGM's gain.
- Diffusion: generating features does not help, in either role. The consistency arms hurt even though their consistency loss is tiny; a likely cause is the extra U batch passing through the encoder every step (an extra forward pass with dropout), but this is not tested.
- Pass 9 ([[Pass 9 Results]]) stacked helpers, FGM and self-training on the off-the-shelf encoder; its best stack (84.6 at 1%, 90.3 at 10%) stays below MLM-TAPT alone (87.8 / 91.0; comparison across notebooks, INFERRED). Whether to re-base it on the MLM encoder is open.

---
tags: [results]
created: 2026-11-15
updated: 2026-11-15
---

# Pass 8 Results

Hub: [[00 - Project Context]]. Plan and predictions (fixed before running): [[Experiment Plan]] "Pass 8 plan". Pass 8b (JEPA, notebook 16) not run yet.

Run by the owner on Kaggle: notebook 15 (`pass8_mlm` + `pass8_mlm_100`, 25 runs, 106.3 min) and notebook 17 (`pass8_diff`, 31 runs). Raw files: `results/pass8_mlm/` and `results/pass8_diff/` (zip, unpacked runs and logs, analysis tables, notebook log renamed `notebook_15.log` / `notebook_17.log`); unpacked folders match the zips (checked with `diff`). Numbers below are VERIFIED from those tables unless marked. Method: paired bootstrap over test items, 10,000 resamples, mean over seeds 0-4, Holm within each notebook's table.

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
- TAPT is a known method (Gururangan et al. 2020), so the novelty is the controlled comparison, not the method: in the same testbed, plain self-supervised use of U beats every GAN, helper, adversarial and LLM-data route. Pass 8b (JEPA) decides whether the objective matters.
- Diffusion: generating features does not help, in either role. The consistency arms hurt even though their consistency loss is tiny; a likely cause is the extra U batch passing through the encoder every step (an extra forward pass with dropout), but this is not tested.
- Pass 9 ([[Experiment Plan]] "Pass 9 plan") stacks ingredients on the off-the-shelf encoder; it does not include TAPT. Whether to re-base it on the MLM encoder is open.

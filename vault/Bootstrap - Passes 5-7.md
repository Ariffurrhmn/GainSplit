---
tags: [results]
created: 2026-11-15
updated: 2026-11-15
---

# Bootstrap - Passes 5-7

Hub: [[00 - Project Context]]. Plan (contrasts fixed before running): [[Experiment Plan]] "Bootstrap plan for Passes 5-7". Seed-paired readings: [[Pass 5 Results]], [[Pass 6 Results]], [[Pass 7 Results]]. Earlier bootstraps: [[Bootstrap - All Passes]] (Passes 1-2), [[Pass 4 Results]].

Run by the owner on Kaggle: notebook 14 (CPU, code commit 485a845, 391 runs loaded, about 3 min). Raw files: `results/analysis_pass5_7/` (zip, 16 tables, `notebook_14.log`); the unpacked tables match the zip (checked with `diff`). All numbers below are VERIFIED from those tables. Method: paired bootstrap over test items, 10,000 resamples, mean over seeds 0-4, Holm within each table. "Robust" = p_holm < 0.05.

**Checks: PASS.** 41 re-runs identical to their originals (`rerun_check`); 25 stage-1 models equal Pass 4's A15 (`stage1_check`).

## Test macro-F1 (points)
| Contrast | 1% | 10% | 100% |
|---|---|---|---|
| A10[weighted] - A1 (pairs at weight 0.1) | -13.86, robust | -10.30, robust | |
| A10[weighted] - A10 (weight 0.1 vs 1.0) | +2.61, robust | +4.53, robust | |
| A16 - A10[weighted] (cleaning) | -0.38, n.s. | -0.22, n.s. | |
| A17 - A16 (FGM on top) | +0.70, p 0.010, Holm 0.077 | +1.75, robust | |
| A18 - A1 (pairs first, then L) | -4.37, robust | +0.29, n.s. | |
| A19 - A7 (pairs first, then L + FGM) | -2.82, robust | -0.12, n.s. | -0.23, n.s. |
| A19 - A18 (FGM in stage 2) | +3.13, robust | +1.13, robust | |
| A7 - A1 (FGM, all labels) | | | +0.56 [0.11, 1.01], p 0.017, Holm 0.12 |
| A19 - A1 | | | +0.33, n.s. |

Accuracy at 100%: A7 - A1 +0.55 [0.10, 1.00], p 0.019, Holm 0.056; A19 - A1 and A19 - A7 n.s.

## iSarcasmEval transfer
- Every pair arm is far above its own baseline in AUC (+6 to +23 points, all robust), but the baselines sit at chance.
- **Net of word count** (`auc_vs_length`, the transfer claim that matters): **A15 (pairs only, no gold labels) +7.52 [4.47, 10.59], robust**; A11 (FGM + pairs) +4.43 at 1% (Holm 0.048) and +5.04 at 10% (Holm 0.025), robust; A13 at 10% +4.82 (Holm 0.080); the other Pass 4 arms +3-4 points, p < 0.05 before Holm only. The Pass 5 arms are level with word count (+0.2 to +1.9, n.s.). The two-stage arms at 10% and 100% fall significantly **below** word count (A18 -9.9, A19 -10.8 at 10%; A19 -11.2 at 100%).
- A7 - A1 at 100%: +2.1 AUC (p 0.022).

## Cue-mask shortcut gap
No pair arm changes the gap relative to its baseline (all 10 contrasts n.s., |diff| < 0.7).

## Reading (INFERRED)
- The expectations written before the run hold: FGM at 100% is significant before Holm only; the two-stage arms tie their baselines at 10% and lose at 1%.
- One pair claim survives the length check: zero-label transfer from the generated pairs alone (A15) is 7.5 AUC points above word count, and FGM + pairs (A11) keeps about 4-5 points. Every recipe that recovers in-domain accuracy (down-weighting, two-stage) loses that residual.
- For [[Paper Framing - Passes 1-7]]: the "explained negative" for the pairs is in-domain only; the transfer residual is a small positive finding that needs careful wording (one target dataset, AUC only).

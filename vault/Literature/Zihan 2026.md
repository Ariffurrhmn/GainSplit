---
tags: [literature]
updated: 2026-11-09
read_depth: FULL
---

# Zihan 2026

Hub: [[00 - Project Context]] · Index: [[Literature Index]] · Source table: [[Deep Dive - Paper A]]

Zihan, Ahmed, Rahman, Dewan (RUET), "Fast Gradient Method (FGM) Fine-Tuning for Bengali Sarcasm Detection", ICECTE 2026, IEEE, DOI 10.1109/ICECTE69292.2026.11429207. 6 pages. Read in full 2026-11-09 (pages 4-6 re-checked directly against the PDF; pages 1-3 via a delegated read). Whether the instructor is an author is UNVERIFIED.

## What it does (VERIFIED, PDF p4-6)
- Fully supervised BanglaBERT + FGM (embedding-layer noise, "small, normalized", citing Miyato 2017), lower 6/8/10 layers frozen, R-Drop on BanglaSarc3 only, label smoothing 0.1, grad clip 0.5, early stopping on val F1. Max length 128. T4 GPUs.
- BenSarc (25,636 Facebook comments, balanced binary). Setup 1: 70/10/20 split, 3/30/40/50% of training labels. Table I: 68.08 / 76.44 / 77.11 / 77.71 (metric unnamed). Setup 2: 100% labels, 80/10/10, Table II: 6 frozen layers 0.8900 acc and F1.
- BanglaSarc3 (12,089, ternary, Bangla + English translation as input). Table III: 6 frozen 0.8274 acc, 0.8275 macro F1; 10 frozen 0.6836.
- Table IV copies GAN-BERT and other baselines from Lora et al. 2023. Not re-run. GAN-BERT: 72.9 / 76.4 / 77.0 / 77.1.
- No seeds, no variance, no significance tests, no FGM-off run in their own setup.

## Gaps and inconsistencies (VERIFIED unless marked)
- Not reported: FGM epsilon, norm, which embeddings, loss combination; learning rates, scheduler (cosine and OneCycle both listed); split seed and indices; R-Drop weight; checkpoint name.
- Fig. 4 (50% labels) sums to 2,564 (a 10% test), but setup 1 says 20% test. Fig. 4 is symmetric (996/286/286/996), giving 77.69%, not 77.71%.
- Fig. 6 sums to 1,217, above 10% of 12,089; test set may have been upsampled although text says training only.
- Text says 6 frozen layers beat 8 and 10 by 2.35% and 6.77%; Table II differences are 0.35 and 2.77 points.
- Text says FGM at 3% "closely matched" GAN-BERT (72.9); Table IV shows 68.08. Tie at 30%, FGM +0.6 at 50%. Claim "consistently 2-3% higher F1" is not supported by the tables.
- Validation set (about 2,564 labelled items) is used for early stopping in the low-label conditions (INFERRED).

## Meaning for GainSplit
- The "GAN not needed" result exists for Bengali only as a cross-paper comparison with copied baselines, different splits and no variance (INFERRED). Our controlled A1/A4/A7 contrast is the missing ablation. See [[Experiment Plan]] and [[Risks]].
- Weak, indirect support for H1 (INFERRED): GAN-BERT's edge appears only at 3%, and is gone from 30% up. Keep p = 1% and 5% in the grid.
- Original setup unavailable: see [[Decisions]] (2026-11-09 limitation entry).

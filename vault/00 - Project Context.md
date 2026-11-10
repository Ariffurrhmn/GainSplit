---
tags: [hub, context]
updated: 2026-11-10
---

# 00 - Project Context

**START HERE.** Hub of the vault. Read this, then the newest entry of [[Session Log]], then follow links as needed. Writing rules: [[Vault Conventions]]. Standing repo rules: [[Repo Rules]].

## Summary
Project **GainSplit** (Paper A): a controlled study of *where the gain comes from* in semi-supervised sarcasm detection. The seed paper ([[Seed Paper - Irfan 2025]]) combines GAN-BERT with multi-task learning (sentiment + emotion helper heads, helper labels made by pretrained models). We separate three ingredients: adversarial training, helper heads, and helper-label quality, across label budgets (1/5/10/30/100%). Target venue: IEEE QPAIN 2027. Status: proposal accepted by the instructor (2026-11-09), who asked for the code and setup as soon as possible. Path: build code and setup, run experiments, write the paper, get instructor feedback on the draft, submit. No code and no real experiments yet (skeleton rolled back).

## Map of the vault
| Area | Notes |
|------|-------|
| The plan | [[Proposal A]] (pitch), [[Experiment Plan]] (conditions, protocol, compute) |
| What we know | [[Deep Dive - Paper A]] (full literature dive), [[Literature Index]], [[Datasets Index]], [[Seed Paper - Irfan 2025]] |
| What we don't know | [[Open Questions]] (incl. instructor questions), [[Risks]] |
| Decisions and ideas | [[Decisions]], [[Research Ideas]], [[00 - Backlog]] |
| Logistics | [[Audit - Venues]], [[Session Log]], [[Repo Rules]], [[Repo List]] |

## Research questions (short)
- RQ1: GAN vs helper heads (trained on the unlabeled pool) vs both, per label budget.
- RQ2: how helper-label quality matters (hard/soft/weighted/lexicon/teacher swap/synthetic noise; ~200-headline gold audit).
- RQ3: transfer to iSarcasmEval; are helper labels just outlet/style proxies?

## Current state (keep updated)
- Repo layout: `vault/` (this), `src/`, `experiments/`, `data/` (git-ignored), `results/`, `paper/`, `docs/proposal/` (PDF).
- Proposal accepted (2026-11-09). Priority now: rebuild code and setup fast so runs can start. All three RQs in scope; no further papers awaited from the owner.
- Next actions: top of [[Session Log]] and [[Open Questions]].
- Data: raw Headlines and iSarcasmEval in `data/` (unedited, checked 2026-11-09); derived splits removed. Code skeleton rolled back 2026-11-09; no real runs yet.
- Preprocessing: `experiments/01_preprocess.ipynb` ran 2026-11-10 (interactive Kaggle session). 28,501 headlines kept; split 22,800 / 2,850 / 2,851; outputs in `data/processed/` and uploaded by the owner as a private Kaggle dataset. Numbers in [[Headlines Dataset]] and [[iSarcasmEval]]. Punctuation removal and drop-duplicates both confirmed ([[Decisions]]).
- Training code: A1 written 2026-11-10 (`src/gainsplit/`, `experiments/02_train.ipynb`); Pass 1 design settled ([[Experiment Plan]]). `lr_tuning` ran 2026-11-10 (9/9 runs; lr 5e-5 confirmed and frozen). A3, A4, A5 still to write.
- Working mode (2026-11-09, [[Repo Rules]] rule 5): the owner does all data work and runs everything; sessions guide and explain, and write code on request.
- Limitation named 2026-11-09: original setup of the seed paper unavailable (supervisor); Zihan 2026 also omits key details; own re-implementation ([[Risks]] item 8). [[Zihan 2026]] read in full.
- Novelty check: [[Chen 2022]] read in full 2026-11-08; partial overlap only (see [[Risks]]). Strongest verified overlap: [[Dai 2021]].

## Rules of thumb
- Evidence labels: **VERIFIED** (read the source), **INFERRED**, **IDEA**, **UNVERIFIED**. Never upgrade a label without reading the source.
- Venue deadlines and formats are unchecked. Do not state them as facts.

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
| Results | [[Pass 1 Results]], [[Pass 2a Results]], [[Pass 2b Results]], [[Bootstrap - All Passes]], [[Gold Audit Results]] |
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
- Training code: A1 written 2026-11-10 (`src/gainsplit/`, `experiments/02_train.ipynb`); Pass 1 design settled ([[Experiment Plan]]). `lr_tuning` ran 2026-11-10 (9/9 runs; lr 5e-5 confirmed and frozen). A3, A4, A5 written and reviewed 2026-11-10; `session2` done (w = 0.3, GAN pilot passed). `pass1` done 2026-11-10 (45/45 runs, test scored): GAN no gain, helpers +0.4 points at 10%, no transfer to iSarcasmEval ([[Pass 1 Results]]). Analysis notebook (`experiments/03_analysis.ipynb`) and Pass 2 arms written, reviewed and merged 2026-11-10. Pass 1 bootstrap done 2026-11-10 (`results/analysis/`): GAN null, helper gain at 10% +0.42 points but p = 0.051 (not significant), GAN + helpers at 1% -2.3 points (p < 0.0001). `pass2a` done 2026-11-10 (104/104, `results/pass2a/`, [[Pass 2a Results]]): helper label content drives the 10% gain, at 1% helpers help only without U, self-training A6 works where the GAN does not (INFERRED until bootstrap). `pass2b` done 2026-11-10 ([[Pass 2b Results]]): FGM (A7) +1.55 points over A1 at 10% on every seed, the largest effect; beats the GAN by 1.5-1.7 points (INFERRED until bootstrap). Pass 2 complete. Gold audit labeled; teacher accuracy 0.53, kappa 0.24, labels accepted ([[Gold Audit Results]]). Bootstrap on all 150 runs done 2026-11-10 ([[Bootstrap - All Passes]]): FGM +1.6 points at 1% and 10% (robust), GAN never helps, self-training helps at 1%, helpers on L + U not significant. Pass 3 planned 2026-11-10 ([[Experiment Plan]] "Pass 3 plan": label-quality rows, A9, masked test, iSarcasmEval in-domain reference; about 105 Kaggle runs). Pass 3 code written, reviewed and merged 2026-11-10. Next: owner runs `01b_pass3_data`, uploads the new dataset version, runs `pass3` (105 runs) and `gan_check` (12 dev-only runs, GAN fairness); paper draft in parallel. Gold-audit app ready (`experiments/gold_audit.html`).
- Working mode (2026-11-09, [[Repo Rules]] rule 5): the owner does all data work and runs everything; sessions guide and explain, and write code on request.
- Limitation named 2026-11-09: original setup of the seed paper unavailable (supervisor); Zihan 2026 also omits key details; own re-implementation ([[Risks]] item 8). [[Zihan 2026]] read in full.
- Novelty check: [[Chen 2022]] read in full 2026-11-08; partial overlap only (see [[Risks]]). Strongest verified overlap: [[Dai 2021]].

## Rules of thumb
- Evidence labels: **VERIFIED** (read the source), **INFERRED**, **IDEA**, **UNVERIFIED**. Never upgrade a label without reading the source.
- Venue deadlines and formats are unchecked. Do not state them as facts.

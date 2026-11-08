---
tags: [hub, context]
updated: 2026-11-08
---

# 00 - Project Context

**START HERE.** Hub of the vault. Read this, then the newest entry of [[Session Log]], then follow links as needed. Writing rules: [[Vault Conventions]]. Standing repo rules: [[Repo Rules]].

## Summary
Project **GainSplit** (Paper A): a controlled study of *where the gain comes from* in semi-supervised sarcasm detection. The seed paper ([[Seed Paper - Irfan 2025]]) combines GAN-BERT with multi-task learning (sentiment + emotion helper heads, helper labels made by pretrained models). We separate three ingredients: adversarial training, helper heads, and helper-label quality, across label budgets (1/5/10/30/100%). Target venue: IEEE QPAIN 2027. Status: draft proposal, not yet shown to the instructor. No code or experiments yet.

## Map of the vault
| Area | Notes |
|------|-------|
| The plan | [[Proposal A]] (pitch), [[Experiment Plan]] (conditions, protocol, compute) |
| What we know | [[Deep Dive - Paper A]] (full literature dive), [[Literature Index]], [[Datasets Index]], [[Seed Paper - Irfan 2025]] |
| What we don't know | [[Open Questions]] (incl. instructor questions), [[Risks]] |
| Decisions and ideas | [[Decisions]], [[Research Ideas]], [[00 - Backlog]] |
| Logistics | [[Audit - Venues]], [[Session Log]], [[Repo Rules]] |

## Research questions (short)
- RQ1: GAN vs helper heads (trained on the unlabeled pool) vs both, per label budget.
- RQ2: how helper-label quality matters (hard/soft/weighted/lexicon/teacher swap/synthetic noise; ~200-headline gold audit).
- RQ3: transfer to iSarcasmEval; are helper labels just outlet/style proxies?

## Current state (keep updated)
- Repo layout: `vault/` (this), `src/`, `experiments/`, `data/` (git-ignored), `results/`, `paper/`, `docs/proposal/` (PDF).
- Next actions: top of [[Session Log]] and [[Open Questions]].
- Biggest known novelty threat: [[Chen 2022]] (full text not yet read).

## Rules of thumb
- Evidence labels: **VERIFIED** (read the source), **INFERRED**, **IDEA**, **UNVERIFIED**. Never upgrade a label without reading the source.
- Venue deadlines and formats are unchecked. Do not state them as facts.

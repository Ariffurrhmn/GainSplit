---
tags: [decisions]
updated: 2026-11-10
---

# Decisions

Hub: [[00 - Project Context]]. Append new entries at the bottom: date, decision, why.

- **2026-11-01** Reframe from "method" to controlled empirical study. Why: sample-weighting has ~20% chance to pass as a contribution (INFERRED); the ablation question is open. See [[Deep Dive - Paper A]] (f).
- **2026-11-01** Use the seed paper's configuration as the reference arm. Why: satisfies the citation goal and gives a faithful baseline ([[Seed Paper - Irfan 2025]]).
- **2026-11-01** Use iSarcasmEval for transfer/evaluation, not a budget curve. Why: small, human-labelled ([[iSarcasmEval]]).
- **2026-11-08** Target venue IEEE QPAIN 2027 (stated by the owner). Fallbacks in [[Audit - Venues]].
- **2026-11-08** Planning deadline 10 Dec 2026 (owner's choice; repo says 20 Dec, unverified). Why: leave slack ([[Audit - Venues]]).
- **2026-11-08** Experiment design: factor the seed result into GAN, helper heads, helper-heads-seeing-U, and helper-label quality, with a shuffled-label control. Why: separates the effects the seed paper confounds ([[Experiment Plan]]).
- **2026-11-08** Obsidian vault in `vault/` for cross-session memory ([[Vault Conventions]]).
- **2026-11-08** Standing rule: no AI co-author trailers and no assistant mentions in the repo ([[Repo Rules]]).
- **2026-11-09** Equal optimizer-step budget for every neural arm (`steps = max(200, 10 x epochs over L)`, GAN arms x2.5); each step = one labeled batch plus, for U-using arms, one unlabeled batch. Why: a fixed step count removes "more training" as a confound between arms; dev is not used for early stopping. Design only: the implementation was rolled back 2026-11-09. The 2.5x for GAN mirrors the seed paper's ~25 vs ~10 epochs (INFERRED, not tuned).
- **2026-11-09** Helper teachers: cardiffnlp/twitter-roberta-base-sentiment (3 classes, neutral kept; order 0 neg, 1 neu, 2 pos, VERIFIED by reading top-confidence examples) and SamLowe/roberta-base-go_emotions mapped to 5 classes (joy, sadness, anger, fear, neutral_other) by our own mapping (script rolled back 2026-11-09, to be rewritten). The seed paper's mapping is unknown ([[Open Questions]]); the mapping must be stated in the paper.
- **2026-11-09** Name the "original setup unavailable" limitation and build our own setup (supervisor's advice). Why: code and details of the seed paper and of [[Zihan 2026]] are not obtainable. Draft paper wording (neutral voice): "The original implementations, data splits and hyperparameters of the seed work and of the FGM work were not available. All arms are re-implemented under one shared protocol (same splits, backbone, budgets, seeds, tuning rule). Published numbers are cited for context and are not compared directly; absolute scores may differ from those reported. Our claims concern differences between arms within one controlled setup." See [[Risks]] item 8.
- **2026-11-09** PROPOSED, awaiting owner approval: A7 spec = single-step L2-normalised FGM on word embeddings, loss = clean + adversarial, epsilon chosen once on dev at p=10% from {0.5, 1, 2}, no label smoothing, R-Drop or layer freezing. Bengali arm out of scope (IDEA; fallback A1/A4/A7 on BenSarc as a robustness paragraph). VAT arm optional (IDEA).
- **2026-11-09** Repo goes public (owner's choice); notebooks on Kaggle pull code from GitHub, data is uploaded once as a private Kaggle dataset.
- **2026-11-09** Seed paper confirmed as the paper whose original setup is unavailable (owner). Code skeleton (`src/gainsplit/`, `experiments/*.py`, `tests/`, `requirements.txt`) deleted; rebuild from scratch. Vault notes, README edits and `data/` kept. Why: owner's call to redo the setup.
- **2026-11-09** Proposal accepted by the instructor; go ahead. Order of work: code and setup first (fast), then experiment runs, then paper draft, instructor feedback, submit.
- **2026-11-09** RQ2 (label-quality study, gold audit) back in scope. Why: the pause logged earlier the same day waited on "the instructor's paper", which is the seed paper already in hand; the owner has supplied every paper requested.
- **2026-11-09** FINAL: the GainSplit GitHub repo is public and stays public (owner). Why: Kaggle notebooks import the code straight from GitHub. Datasets stay git-ignored and go to Kaggle as a private dataset. Supersedes the "repo stays private" line in [[Session Log]].
- **2026-11-09** Preprocessing notebook `experiments/01_preprocess.ipynb` written for Kaggle; the owner runs it. Defaults, PENDING owner confirmation (switches in the settings cell): keep punctuation (seed paper removed it; punctuation may be a sarcasm cue) and drop duplicate headlines (conflicting-label duplicates removed entirely). Why: avoid train/test leakage through duplicates; keep text close to what BERT was trained on.
- **2026-11-09** Label budgets are stratified and nested per seed (1% inside 5% inside 10% ...), saved as id lists in `budgets.json`. Why: differences between budgets then reflect label amount, not a different random draw.
- **2026-11-09** Emotion helper labels: GoEmotions sigmoid scores grouped into 5 classes (GoEmotions Ekman grouping; disgust merged into anger, surprise into neutral_other), class score = max over its labels, renormalised to sum 1. Why max: joy has 12 labels and would win under a sum. Raw 28 scores also saved for RQ2. Sentiment keeps neutral (seed paper dropped it).
- **2026-11-09** CONFIRMED by owner: punctuation removed (`KEEP_PUNCTUATION = False`) for all main runs, before helper labels are made. Why: matches the seed paper, removes one outlet-style shortcut, and the literature shows small, task-dependent effects for transformers ([[Punctuation Evidence]]). Keep-punctuation is an optional robustness check (A3/A5 at 10%, 3 seeds). Supersedes the keep-punctuation default above. Drop-duplicates default still unconfirmed.
- **2026-11-10** CONFIRMED by owner: drop duplicate headlines (`DROP_DUPLICATES = True`). The 2026-11-10 run removed 118 extra copies; no duplicates had conflicting labels. Why: a headline in both train and test would inflate test scores. Supersedes the "unconfirmed" note above.

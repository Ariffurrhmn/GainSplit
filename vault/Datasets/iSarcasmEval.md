---
tags: [dataset]
updated: 2026-11-09
---

# iSarcasmEval

https://github.com/iabufarha/iSarcasmEVAL ; Abu Farha et al., Findings EMNLP 2022. Hub: [[00 - Project Context]] · Index: [[Datasets Index]].

- Repo MIT; author-labelled plus a third-party folder; English Task A test set 1,400 (VERIFIED). English train size UNVERIFIED ([[Open Questions]]).
- Small and human-labelled: use for transfer/evaluation (RQ3), not a budget curve ([[Decisions]], [[Experiment Plan]]).

## Local copy (VERIFIED 2026-11-08)
- Shallow clone of the repo at commit `dfc708b` in `data/isarcasmeval/` (git-ignored; an earlier clone had stalled with no commits and was completed).
- English train (`train/train.En.csv`): **3,468 rows, 867 sarcastic** (25%); columns tweet, sarcastic, rephrase, plus sarcasm-type flags. English Task A test (`test/task_A_En_test.csv`): **1,400 rows, 200 sarcastic**, columns text, sarcastic. Arabic files also present, unused.
- Resolves the "English train size" item in [[Open Questions]].
- 2026-11-09: clone re-checked with `git status` (no local changes, still at `dfc708b`) and kept as raw. Derived files in `data/processed/` removed (moved to the Recycle Bin). The owner will clean and split the data personally ([[Repo Rules]] rule 5).

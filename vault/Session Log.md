---
tags: [log]
updated: 2026-11-10
---

# Session Log

Newest first. Hub: [[00 - Project Context]]. Format: [[Templates/Session Entry]].

## 2026-11-10 - Preprocessing run finished
- Owner re-ran `experiments/01_preprocess.ipynb` in an interactive Kaggle session (the commit run stalled on Kaggle's side 2026-11-09). It finished; `/kaggle/working/processed/` holds about 12 MB.
- Interactive-session output is deleted when the session ends, so the owner downloads the 9 files to `data/processed/` and uploads them as a private Kaggle dataset for the training notebooks.
- Owner downloaded the 9 files to `data/processed/` (sizes match the manifest). Manifest and sentiment spot-check reviewed: all checks pass. Numbers recorded in [[Headlines Dataset]] and [[iSarcasmEval]]; emotion-label skew added to [[Risks]] (item 11).
- Owner uploaded the processed files as a private Kaggle dataset and confirmed drop-duplicates ([[Decisions]]). All 2026-11-09 and 2026-11-10 repo changes committed and pushed.
- Pass 1 training-code plan explained to the owner and written to [[Experiment Plan]] (PROPOSED). Open: A5 helper design, equal steps vs GAN x2.5, helper loss weights, processed dataset name on Kaggle.
- Processed dataset on Kaggle is `gainsplit-processed`. Draft 2x2 figure of the Pass 1 arms added at `paper/figures/pass1_design.svg`. Decisions (a)-(c) explained in plain terms; owner confirmed all three recommendations ([[Decisions]]): shared helper heads in A5, equal steps for every arm, helper weight tuned once on dev.
- Owner chose one Kaggle session on GPU T4 x2, two runs at a time, over separate sessions ([[Decisions]]). Quota rate for T4 x2 still to check.
- A1 code written on the owner's request: `src/gainsplit/`, `tests/`, `experiments/02_train.ipynb` ([[Experiment Plan]]). Implementation choices in [[Decisions]]. Data tests pass locally on made-up data; the training smoke tests run on Kaggle (no torch locally). Not committed yet.
- Owner pushed the code (commit 29e4220) and ran `lr_tuning` interactively on GPU T4 x2: 9/9 runs done, 1.3 min each. Dev macro-F1 0.855-0.875; best mean lr 5e-5 (0.8673), confirmed by the owner and set as the code default ([[Decisions]]). Notebook's expected-score line corrected (0.85-0.88 at 10%, not 0.9).
- Owner placed `results_lr_tuning.zip` and its unpacked files in `results/lr_tuning/` (all 28 files match the zip). The 9 `.log` files are git-ignored by the `*.log` rule; they stay inside the zip, which is tracked.
- Roadmap to submission written into [[Experiment Plan]]. All changes committed and pushed.
- Next: write A3 (helper heads) and the helper-weight tuning plan.

## 2026-11-09 - Preprocessing notebook
- Wrote `experiments/01_preprocess.ipynb` for Kaggle: verifies raw file hashes, drops `article_link`, light cleaning, empty/duplicate removal, token-length check, stratified 80/10/10 split, nested label budgets, teacher helper labels, iSarcasmEval cleaning, manifest. Each cell explains what and why.
- Defaults pending owner confirmation: keep punctuation, drop duplicates ([[Decisions]]). Emotion mapping and max-aggregation recorded there too.
- Punctuation: literature checked ([[Punctuation Evidence]]); owner chose removal, set in the repo notebook and on Kaggle. Owner started the first Kaggle commit run.
- End of day: the Kaggle run stalled; Kaggle reports processing halted on its side (platform issue, being fixed). No output yet. Owner resumes 2026-11-10: check the run, re-run if needed, then review its output. Repo changes from today not yet committed or pushed.
- Not run (owner runs it). Next: owner uploads the three raw files as a private Kaggle dataset, imports the notebook from GitHub, runs it, and copies the summary numbers into [[Headlines Dataset]] and [[iSarcasmEval]].

## 2026-11-09 - Proposal accepted, status corrected
- Owner: instructor has seen and accepted the proposal and asked for code and setup as soon as possible. Plan: code and setup, runs, paper draft, instructor feedback, submit. Hub, [[Proposal A]], README and [[Open Questions]] status lines updated; logged in [[Decisions]].
- RQ2 un-paused: the earlier pause waited on "the instructor's paper", which is [[Seed Paper - Irfan 2025]], already in hand. Owner has supplied every paper requested; none outstanding.
- Repo is public and stays public (owner): Kaggle notebooks import code from GitHub. Old "stays private" note marked superseded.
- Data check: Headlines JSON hash matches the download record; iSarcasmEval clone has no local changes. Both kept as raw. `data/processed/` (splits made by the deleted script) moved to the Recycle Bin.
- New standing rule ([[Repo Rules]] rule 5, also in the root instructions file): the owner gets, cleans and uploads data and runs all code; sessions guide step by step and write code only on request, with explanations.
- Next: rebuild the code skeleton from scratch ([[Experiment Plan]]). Still open: A7 spec approval, QPAIN deadline check.

## 2026-11-09 - Limitation named, FGM paper analysed
- Owner: the limitation concerns the seed paper. Setup rolled back: deleted untracked `src/gainsplit/`, `tests/`, `experiments/*.py`, `requirements.txt`; vault, README and `data/` kept. To be rebuilt from scratch.
- Supervisor: expect no access to the original setup; build our own and name this as a limitation. Recorded in [[Decisions]] and [[Risks]] (item 8).
- [[Zihan 2026]] read in full (delegated read, pages 4-6 re-checked). Main findings: no FGM-off ablation, baselines copied, no seeds, several internal inconsistencies, low-label claim overstated. It supports keeping A7 and our fixed-step, multi-seed protocol.
- Open: which paper the supervisor means, A7 spec approval, Bengali arm out of scope ([[Open Questions]]). No code changed.

## 2026-11-09 - Code skeleton and smoke tests
- Wrote split, models, training loop for A0-A8, runner, teacher script and tests (see [[Experiment Plan]] implementation notes). 15 tests pass on a tiny random BERT; no real training or timing yet. Decisions logged in [[Decisions]].
- Found and fixed two bugs during review: FGM used a different batch for the adversarial pass; self-training later rounds used more steps than other arms.
- Teacher script checked on 200 headlines; sentiment label order verified. Full helper-label run still to do.
- Next: run `make_helpers.py` on Kaggle GPU, upload `data/processed` + helper labels as a private Kaggle dataset, tune once on dev, then Pass 1.

## 2026-11-09 - Owner answers
- Kaggle quota: owner states 30 GPU-h per week (own account info, not independently verified). Planning deadline stays 10 Dec 2026. Instructor has seen and accepted the proposal. Model downloads approved.
- Emotion teacher undecided by owner; default is a GoEmotions model mapped to 5 classes (teacher swap is a Part 2 arm anyway).
- RQ2 (gold audit, label-quality rows) PAUSED by owner until the instructor's paper arrives; owner will supply it. Part 1 and Part 3 continue.
- Repo stays private for now (unpublished work, venue anonymity rule unchecked, Headlines licence unchecked). SUPERSEDED same day: repo is public, see [[Decisions]]. Kaggle delivery plan: bundle src, configs and data split into a private Kaggle dataset rather than cloning from GitHub.

## 2026-11-08 - Datasets gathered
- Downloaded Headlines (28,619 rows) and iSarcasmEval (EN train 3,468, EN Task A test 1,400) into `data/`; details and hashes in [[Headlines Dataset]], [[iSarcasmEval]]. Headlines file is v2-sized despite the name; mismatch with v1 count noted.
- Next: build the fixed 80/10/10 split script in `src/`, drop `article_link`, run teachers for helper labels.

## 2026-11-08 - Experiment plan rewritten
- Rewrote [[Experiment Plan]]: 9-arm core grid (adds A2 heads on L only, A3 heads on L+U, A8 shuffled-label control), label-quality study, transfer and shortcut checks, written-down hypotheses, 6-page layout. Decision logged in [[Decisions]].
- Owner runs on free Kaggle GPU. Experiment Plan now has three passes sized for ~25 GPU-h, 3 seeds first. Kaggle quota still unverified.

## 2026-11-08 - QPAIN 2027 rules and instructor status
- Read the owner's QPAIN venue-instructions notes and source snapshots; summarised in [[Audit - Venues]] (all UNVERIFIED, second-hand). Deadline conflict 10 vs 20 Dec to resolve.
- Owner chose 10 Dec 2026 as the planning deadline (about 4.5 weeks from 2026-11-08).
- Instructor status and unanswerable questions recorded in [[Open Questions]]. Zihan 2026 link found via Consensus; PDF still needed from the owner's library.

## 2026-11-08 - Chen 2022 read in full
- Read the Chen 2022 PDF end to end; rewrote [[Chen 2022]] (read_depth FULL). Verdict: partial pre-emption of RQ2 only; their "soft label" is a lexicon BoW distribution, models are GloVe-LSTM, no SSL or label budgets. Updated [[Risks]], [[Literature Index]], hub, [[Open Questions]].
- Consensus search confirmed the seed paper metadata. Zihan 2026 did not surface there.
- Next: Zihan 2026 full text; send instructor questions; check QPAIN 2027 call.

## 2026-11-08 - Repo List and reusable setup prompt
- Added [[Repo List]] with this repo as the template for new project repos; linked from the hub; rule 9 in [[Vault Conventions]].
- A setup prompt for other repos was written in chat (not stored in the repo). It tells the new session to read this repo's rules and conventions first.

## 2026-11-08 - Repo and vault setup
- Organized files: proposal PDF in `docs/proposal/`, notes moved into `vault/`; added README, .gitignore, empty `src/ experiments/ data/ results/ paper/`.
- Built this vault from [[Proposal A]] and [[Deep Dive - Paper A]]; added literature, dataset, decision, question, risk and rules notes ([[Repo Rules]]).
- Next: get the Chen 2022 PDF ([[Chen 2022]]); send instructor questions ([[Open Questions]]); check the QPAIN 2027 call ([[Audit - Venues]]).

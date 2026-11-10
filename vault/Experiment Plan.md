---
tags: [experiments, plan]
updated: 2026-11-10
status: planned, not started
---

# Experiment Plan

Hub: [[00 - Project Context]]. Pitch: [[Proposal A]]. Prior work: [[Deep Dive - Paper A]]. Deadline and page limit: [[Audit - Venues]] (6 pages, 10 Dec planning date).

Everything below is a design (IDEA) until run. Compute figures are INFERRED, not benchmarked.

## The question
The seed paper ([[Seed Paper - Irfan 2025]]) reports one number for GAN-BERT + sentiment/emotion helper heads. It never separates (a) the GAN, (b) the helper heads, (c) the extra unlabeled data the helper heads may see, or (d) the quality of the machine-made helper labels. We separate them.

## Fixed setup (all arms)
- Data: [[Headlines Dataset]], one fixed 80/10/10 split (train/dev/test) by seed-independent hash; test never touched until the end. Seeds change only the labeled subset, init and batch order.
- Backbone: bert-base-uncased, max_len 48, fp16. Same optimizer, lr, batch size, epochs for every arm unless the arm's definition says otherwise (GAN-BERT gets its own epochs, as in the seed paper).
- Label budget p in {1, 5, 10, 30, 100}% of train. Labeled set = stratified random p%; **unlabeled pool U = the remaining train texts** (labels hidden). At p=100% U is empty, so arms that need U are skipped there.
- Helper labels (sentiment, emotion) are produced once by the teachers for every train text, labeled or not (teachers per [[Seed Paper - Irfan 2025]]: cardiffnlp twitter-roberta sentiment, GoEmotions-style model mapped to 5 classes; our own mapping, documented, because the seed's is unknown).
- Hyperparameters tuned once on dev at p=10% with the supervised BERT arm, then frozen. Dev is never used for early stopping per run: fixed epoch count. (Avoids leaking more labels than the budget allows.)
- Metrics: accuracy, macro-F1, ROC-AUC on test. 5 seeds; mean +- std; paired bootstrap (same test set) for every comparison claimed in the text.

## Part 1 (RQ1): who gives the gain? Core grid
Notation: L = labeled, U = unlabeled pool, H = helper heads.

| Arm | Description | Uses U? |
|---|---|---|
| A0 | TF-IDF + logistic regression (floor) | no |
| A1 | BERT, labeled only | no |
| A2 | BERT + H, heads trained on **L only** | no |
| A3 | BERT + H, heads trained on **L + U** (helper labels exist for U) | yes, via heads |
| A4 | GAN-BERT, no helpers ([[GAN-BERT]]) | yes, via GAN |
| A5 | GAN-BERT + H (seed-style, our re-implementation) | yes, both |
| A6 | Self-training (pseudo-label U, one or two rounds) | yes |
| A7 | BERT + FGM adversarial fine-tuning, labeled only (supervised adversarial control, [[Zihan 2026]]) | no |
| A8 | A3 with **shuffled** helper labels (negative control) | yes |

What each contrast tells us:
- A1 vs A2: do helper heads help when they add no data? (pure multi-task effect)
- A2 vs A3: does the gain come from the heads seeing U? (the question the seed paper leaves open)
- A3 vs A8: is it the label content or just extra heads/regularisation?
- A1 vs A4: does the GAN help at all? A4 vs A7: is it "adversarial" or just regularisation?
- A5 vs max(A3, A4): is there any interaction, or is the combo no better than its best part?
- A6: does a trivial SSL baseline match the GAN?

Runs: 9 arms x 5 budgets x 5 seeds, minus U-arms at 100% = about 200 runs. Priority order if time runs out: A1, A3, A4, A5 first (the 2x2), then A2, A8, A6, A7, A0. Seeds: run 3 seeds first, extend to 5 for the headline budgets (1, 10, 100).

## Part 2 (RQ2): does helper-label quality matter?
Take the best helper arm from Part 1 (expected A3 or A5) and vary only the helper labels, at budgets 1%, 10%, 100%, 5 seeds:
1. hard teacher labels (reference)
2. soft teacher probabilities
3. confidence-weighted loss (weight by teacher max-prob)
4. lexicon labels (VADER or the Hu-Liu lexicon used by [[Chen 2022]]); no model teacher
5. teacher swap: a second, different sentiment teacher
6. synthetic noise dial: flip 10/25/50% of helper labels
7. random labels (= A8)
Gold audit: the owner hand-labels ~200 randomly drawn headlines for sentiment (3 classes); report teacher accuracy against gold and plot downstream F1 vs teacher accuracy across rows 1-6. This is the only step that needs human effort.

## Part 3 (RQ3): transfer and shortcut checks
- Zero-shot transfer: models trained on Headlines (all arms, p=100% and p=10%) evaluated on the [[iSarcasmEval]] English Task A test set (1,400, VERIFIED). Expect a large drop ([[Headlines Dataset]]); the question is whether any arm drops less.
- Mutual information between teacher labels and the sarcasm label (CPU, hours): if helper labels predict sarcasm because they encode outlet style, say so.
- Source-masked test: strip outlet-specific cues (casing patterns, punctuation, names) and re-evaluate the main arms.
- Optional only if time: RoBERTa-base repeat of the 2x2 at p=10%.

## Predictions written down before running (so null results are honest)
- H1: GAN effect is about zero for p >= 10% (INFERRED from [[GAN-BERT]]'s own curves; [[Risks]]).
- H2: A3 >= A2 at low budgets, i.e. part of the "multi-task gain" is the heads seeing U.
- H3: A8 (shuffled labels) is close to A2 but below A3.
- H4: label quality has a weak, non-monotone effect until noise is large (>=25%).
Any outcome is reportable: the paper is a controlled study, not a new-method claim ([[Decisions]]).

## Compute (INFERRED, not benchmarked)
BERT-base, max_len ~48, fp16, T4: ~1-1.5 min/epoch at 100% labels. GAN arms and U-arms cost more because U is large at low p. Rough total for Parts 1-3: 15-30 GPU-hours.

**Hardware (owner, 2026-11-08): free Kaggle GPU.** Weekly quota and per-session limit are UNVERIFIED (recollection: roughly 30 GPU-h per week, sessions of about 9-12 h; check the Kaggle account page). Plan for ~25 GPU-h over the first three weeks:
- Pass 1 (must, ~6-8 GPU-h): arms A1, A3, A4, A5 at p = 1, 10, 100 (no U-arms at 100), **3 seeds**. This alone answers RQ1.
- Pass 2 (~6-8 GPU-h): add A2, A8, A7, A6, A0 at the same budgets; add p = 5, 30 only for A1/A3/A4/A5 for the curve.
- Pass 3 (~5-8 GPU-h): Part 2 label-quality rows at p = 1 and 10 only, 3 seeds; Part 3 transfer reuses saved checkpoints (no extra training). Extend to 5 seeds on the headline cells only if quota remains.
- Rules for Kaggle: one notebook run per arm-budget-seed batch, save results JSON and test predictions to `/kaggle/working` and download after each batch; checkpoint every epoch so a 9-12 h cut-off loses nothing; upload the data once as a private Kaggle dataset. Code comes from the public GitHub repo (cloned or pip-installed in the notebook), so each run uses a pushed commit; record the commit hash in the results JSON.

## Paper layout (6 pages incl. references)
Fig. 1: F1 vs label budget, arms A1/A3/A4/A5 (+A6/A7 dashed). Table 1: main grid at p = 1, 10, 100 with std and significance marks. Table 2: label-quality rows. Table 3: iSarcasmEval transfer. One short paragraph for the audit, MI and masking checks.

## Known reproduction details (VERIFIED from the seed PDF)
bert-base-uncased; 80/20 split; 70% of train labelled, 30% unlabelled; sentiment from cardiffnlp/twitter-roberta-base-sentiment (neutral eliminated); emotion from a GoEmotions model or pysentimiento mapped to 5 classes; L = L_s + l1 L_sent + l2 L_emo + L_adv; per-head G/D pairs; punctuation removed; ~25 epochs GAN-BERT vs ~10 BERT. Unknowns, now permanent: [[Open Questions]]. Our A5 is therefore a re-implementation, compared with the seed's reported 93.88 acc / 94.15 F1 only qualitatively.

## Pass 1 implementation plan (2026-11-10; A1 written, A3/A4/A5 not yet)
- Arms as a 2x2 of switches in one shared training loop: A1 (neither), A3 (helper heads on L+U), A4 (GAN, no helpers), A5 (GAN + the same helper heads). Arms differ only in these switches; data, backbone, optimizer, batch, steps and seeds are shared.
- Runs: A1 at 1/10/100%, A3/A4/A5 at 1/10% (U empty at 100%), 3 seeds = 27 runs.
- Files (to write on request): `src/gainsplit/data.py` (load `processed/`, build L/U/dev/test from `budgets.json`), `models.py` (BERT encoder, sarcasm head, helper heads, generator/discriminator), `train.py` (one loop with the switches, fixed step count, fp16, resume from checkpoint), `evaluate.py` (accuracy, macro-F1, ROC-AUC, saved predictions), a CPU smoke test on a tiny BERT, and `experiments/02_train.ipynb` (clones the repo at a pushed commit, attaches the processed dataset, runs a batch of runs, writes results JSON + test and iSarcasmEval predictions).
- Save predictions, not model weights (27 x ~440 MB would not fit); RQ3 transfer predictions on iSarcasmEval test are made at the end of each p = 10/100 run.
- Build order: A1 first; owner runs one A1 seed at 10% to time it and sanity-check F1; tune lr once on dev at 10% with A1; then A3, then A4/A5; then the batch runner.
- Data on Kaggle: private dataset `gainsplit-processed`. Pretrained models are public on the Hugging Face Hub, so no HF token is needed (Internet on).
- Written 2026-11-10: `src/gainsplit/` (`config.py` arms and step rule, `data.py` find/verify/split, `models.py`, `train.py`, `evaluate.py`, `runner.py` two-GPU queue, `plans.py` run lists), `tests/` (data tests pass locally on made-up data; training smoke tests need torch and run in the notebook), `experiments/02_train.ipynb`. A3, A4, A5 added the same day (reviewed; loss formulas in [[Decisions]]); plans `w_tuning`, `gan_pilot`, `session2`, `pass1` (45 runs, needs `PASS1_HELPER_WEIGHT`). Torch code checked by reading only; the smoke tests run in notebook step 2. First plan: `lr_tuning` (9 runs), run 2026-11-10 on Kaggle (owner, interactive, GPU T4 x2, commit 29e4220): all 9 finished, 1.3-1.4 min each, dev only. Scores in [[Decisions]]; lr 5e-5 confirmed by the owner and frozen. Measured cost: 720 steps = about 1.3 min on a T4 incl. model loading, so Pass 1 (27 runs) is roughly 1-1.5 GPU-h (INFERRED from this timing; U-arms assumed about 2x per step), far below the 6-8 GPU-h budgeted.
- Runner: one Kaggle notebook on GPU T4 x2, a list of runs, two at a time (one per GPU) ([[Decisions]] 2026-11-10).
- Figure: `paper/figures/pass1_design.svg` (2x2 of arms with the contrasts; draft for the paper).
- DECIDED 2026-11-10 ([[Decisions]]): (a) shared helper heads; (b) equal steps for every arm; (c) w tuned once from {0.1, 0.3, 1.0}. Original options for the record: (a) A5 helper design: shared helper heads on the encoder (recommended; cleanly A3 + A4) vs the seed paper's per-head generator/discriminator pairs; (b) step budget: same steps for every arm (recommended) vs GAN arms x2.5 as logged 2026-11-09, which gives GAN arms more training and re-adds the confound the fixed budget was meant to remove; (c) helper loss weights: fixed 0.5/0.5 or tuned once on dev with A3 at 10%.

## Roadmap to submission (2026-11-10, planning deadline 10 Dec)
Owner runs everything on Kaggle; code is written on request. Steps in the same block can run side by side.
1. **Code, one go:** A3 (helper heads on L + U), A4 (GAN-BERT), A5 (A4 + A3 heads), plus plans `w_tuning` and `gan_pilot`.
2. **DONE 2026-11-10** (12/12 runs, ~2 min each; w = 0.3, GAN pilot passed, [[Decisions]]). **Kaggle session 2 (~30-40 min):** `w_tuning` (A3, p = 10%, w in {0.1, 0.3, 1.0}, seeds 0-2, dev only) and `gan_pilot` (A4, p = 10%, seeds 0-2, dev only: checks the GAN trains). Owner picks w.
3. **DONE 2026-11-10** (45/45 runs; [[Pass 1 Results]]). **Was: NEXT, ready** (`PLAN = "pass1"`, w = 0.3 set; measured timings give ~1.5-2 GPU-h, ~1 h on two GPUs). **Kaggle session 3:** Pass 1 with test scored: A1 at 1/10/100%, A3/A4/A5 at 1/10%, **5 seeds** (45 runs; DELEGATED, [[Decisions]]).
4. **DONE 2026-11-10, owner to run** (`experiments/03_analysis.ipynb`, `src/gainsplit/analysis.py`). **Code:** analysis notebook: mean +- std tables, paired bootstrap per contrast, F1-vs-budget figure.
5. **`pass2a` DONE 2026-11-10** (104/104, [[Pass 2a Results]]); epsilon 0.5; **`pass2b` DONE 2026-11-10** ([[Pass 2b Results]]). Pass 2 complete. **Pass 2** (specs and split into `pass2a` / `pass2b` in [[Decisions]] 2026-11-10): A2, A8, A6, A0, A7 (A7 spec approved, DELEGATED 2026-11-10), budgets 5/30 for the 2x2 arms.
6. **Pass 3: PLANNED 2026-11-10** ("Pass 3 plan" below): CPU analysis (3a), new label and masked-test files (3b), Kaggle plan `pass3` (3c). Code written and merged 2026-11-10; **NEXT: owner runs `01b_pass3_data`, then `pass3`**.
Side tracks, any time: gold audit (owner hand-labels 200 random training headlines in `experiments/gold_audit.html`; protocol in [[Decisions]]); check the QPAIN 2027 deadline and anonymity rule; draft the paper's data and setup sections (setup is now fixed); instructor feedback on the draft; submit.

## Pass 3 plan (2026-11-10, DELEGATED; replaces the Part 2 / Part 3 lists above where they differ)
Revised after Passes 1-2 ([[Bootstrap - All Passes]]): helpers on L + U give no significant gain, helpers on L only help at 1%, label content matters against shuffling at 10%, every arm is at chance on iSarcasmEval. Reasons per item in [[Decisions]].

**Step 3a: CPU analysis, no training (owner runs locally).** New cells in `03_analysis.ipynb`:
1. *Helper labels vs sarcasm (RQ3 shortcut):* normalised mutual information and Cramer's V between teacher sentiment / emotion labels and the sarcasm label on the training set; a probe (logistic regression on the 8 teacher probabilities, 5-fold CV ROC-AUC) predicting sarcasm from helper labels alone. Same probe on the 200 gold items with the owner's labels vs the teacher's labels: if the teacher's labels predict sarcasm and the owner's do not, the teacher encodes style, not surface sentiment.
2. *Transfer with intervals:* bootstrap 95% interval of iSarcasmEval ROC-AUC per arm and budget (is anything above 0.5?), and A3 - A1, A5 - A1 at 30% on iSarcasmEval.
3. *Gold table:* teacher accuracy, macro-F1, kappa against the owner's labels for every helper-label source (rows R1-R5 below, computed once the labels exist).

**Step 3b: new label and test files (owner runs `experiments/01b_pass3_data.ipynb` on Kaggle, then adds them to `gainsplit-processed` as a new version).**
1. VADER sentiment for every headline (compound >= 0.05 positive, <= -0.05 negative, else neutral; standard thresholds).
2. Second sentiment teacher, 3 classes, trained on different data from the cardiffnlp tweet model. Candidate `j-hartmann/sentiment-roberta-large-english-3-classes` (UNVERIFIED: the owner checks the model card for 3 classes and training data before running; fallback `finiteautomata/bertweet-base-sentiment-analysis`, UNVERIFIED).
3. Outlet-cue masked test: train a word-unigram logistic regression (as A0) on the full training set; take the 100 words with the most positive and the 100 with the most negative weights (200 words, fixed before any masked score is seen); replace them with `[MASK]` in the test headlines. Control: in each test headline mask the same number of randomly chosen words (seeded). Report the share of test headlines touched.

**Step 3c: Kaggle plan `pass3` (one session, about 105 runs, ~1.5-2 h on two GPUs; INFERRED from Pass 2 timings).** Every run also scores the masked and random-masked test sets.
- RQ2 label rows, each at A3 / 10% and A2 / 1%, seeds 0-4 (70 runs): R1 soft teacher probabilities (both heads); R2 25% and R3 50% of helper labels flipped to another class at random (both heads, seeded); R4 VADER sentiment; R5 second sentiment teacher; R6 sentiment head only; R7 emotion head only. R4/R5 change only the sentiment labels; the emotion head keeps its teacher labels. Reference rows already exist: hard labels (A3, A2) and shuffled (A8; A2-shuffled below).
- A9 = A2 with shuffled helper labels at 1% (5 runs): is A2's +1.6 at 1% the label content or the extra heads?
- Masked-test re-runs: A1, A3, A4, A6, A7 at 10%, seeds 0-4 (25 runs). Training is deterministic, so these rebuild the Pass 1-2 models; their plain test scores must match the earlier runs exactly (built-in check).
- iSarcasmEval in-domain reference: A1 trained on iSarcasmEval train (3,467), tested on iSarcasmEval test, seeds 0-4 (5 runs). Shows how far the Headlines models are from what the task allows, so "chance level" reads as a transfer failure, not an impossible test set.

**Dropped from the old Part 2 list:** confidence-weighted loss (close to soft labels; one fewer row), budgets 100% (no U), RoBERTa repeat. **Optional, only if time:** GAN at 1% with more steps; a small GAN hyperparameter check on dev ([[Risks]] item 13).

**Code (three sessions in parallel, disjoint files):** (A) `experiments/01b_pass3_data.ipynb` + `src/gainsplit/masking.py`; (B) training switches R1-R7, A9, extra eval splits, iSarcasmEval-train arm, `pass3` plan, tests (`config.py`, `data.py`, `train.py`, `evaluate.py`, `plans.py`); (C) step 3a analysis (`analysis.py`, `03_analysis.ipynb`, tests). File formats shared by A and B are fixed in the briefs.

## Status
Code rolled back 2026-11-09 by the owner's decision: `src/`, `experiments/` and `tests/` are empty again and the implementation will be rebuilt from scratch. The design above (arms, budgets, protocol) still stands unless changed in [[Decisions]]. Earlier implementation notes (split sizes, A5 reading, smoke tests) were discarded with the code. Log runs in [[Session Log]].

- 2026-11-09: preprocessing rebuilt as one Kaggle notebook, `experiments/01_preprocess.ipynb` (owner runs it; GPU + internet on). Outputs: headlines train/dev/test, `budgets.json` (5 seeds x 5 budgets, nested), helper labels (hard + soft, plus raw 28 emotion scores), iSarcasmEval train/test, `manifest.json` with counts and hashes. Choices in [[Decisions]]. Not yet run; code cells syntax-checked and the cleaning and budget functions tested on synthetic data only.

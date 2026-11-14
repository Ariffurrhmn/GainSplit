---
tags: [experiments, plan]
updated: 2026-11-14
status: Passes 1-4 done (Pass 4 results in [[Pass 4 Results]]); Pass 5 plan written, not run
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

## Pass 4 plan (run 2026-11-13, results in [[Pass 4 Results]]; started 2026-11-12; training arms PROPOSED 2026-11-13, DELEGATED per [[Repo Rules]] rule 7)
Data: `pairs_train.csv` from [[Pairgen Full Run Results]] (unfiltered main set, 22,417 pairs) and `pairs_train_filtered.csv` (judge v2, 17,315 pairs; ablation only). Not written or run yet.

**Why every train seed may be used at every budget (VERIFIED from `pairgen.build_prompt`):** the generation prompt shows only the seed text, never its label, and every generated side is new text (no copy of any train, dev or test headline). So a pair carries no gold label of its seed. Pairs from all 22,800 train seeds are the semi-supervised signal made from the unlabeled pool; no run sees a gold label outside its own L.

**How pairs enter training.** Same loop, same step rule as every other arm (equal steps: 200 at 1%, about 720 at 10%, about 7,130 at 100%). Each step adds one pair batch after the L batch, as the pool batch does for the U arms. The pair batch holds whole pairs (both sides of a pair in the same batch), so each update sees the two sides of the same topic side by side. Pair rows have their intended label and join the one sarcasm loss over L + pair rows, the same way A6 uses pseudo-labeled rows. No new loss term, no new weight. Consequence: at 1% a run sees about 6,400 pair rows and at 10% about 23,000, i.e. part of the set; the set size is not the limiting factor at these budgets.

**Arms** (5 seeds, 0-4, test scored; baselines A1, A6, A7 reused from Passes 1-2 because training is deterministic, checked exactly in Pass 3):
| Arm | What it is | Budgets | Runs | Question |
|---|---|---|---|---|
| A10 | A1 + unfiltered pairs | 1, 10, 100 | 15 | Do the pairs help, and where? |
| A11 | A7 (FGM, Pass 2b epsilon) + pairs; FGM covers L and pair rows | 1, 10 | 10 | Do pairs stack with the best existing method (the candidate recipe)? |
| A12 | unpaired control: one side per seed (side picked at random, seed 2027), rows sampled singly; same rows per batch as A10 | 1, 10 | 10 | Does the topic-matched pairing matter, or only the extra generated text? |
| A13 | A10 with the judge-filtered pairs | 1, 10 | 10 | Does the judge filter help? |
| A14 | A10 with a random subset of the unfiltered pairs, same per-model counts as A13 (Gemma 7,533, Mistral 4,986, Qwen 4,796; seed 2027) | 1, 10 | 10 | Control for A13: separates filtering from the change in model mix |
| A15 | pairs only, no gold labels; run as budget 10 so it gets the 10% step count (720) | 10 | 5 | How far do LLM pairs go with zero human labels? |
| check | A1 at 10% re-run with explanations (tag `explain`), seeds 0-4 | 10 | 5 | Must match Pass 1 exactly (new code leaves the old path unchanged); also the A1 side of the explainability step |
| mask | A1 and A7 at 1% re-runs with masked test sets (tag `mask`) | 1 | 10 | Masked-test baselines at 1% (Pass 3 had them at 10% only); must match Pass 1 / 2b |
| sens | A10 at 1% with `step_mult = 3` (tag `steps3x`) | 1 | 5 | Do the pairs need more steps at 1%? Sensitivity only, not a main contrast |

~~`pair_pilot`~~ dropped 2026-11-13 ([[Decisions]]): the pair batch is fixed at 32 rows (16 pairs), the size of the pool batch in the helper and GAN arms, so the whole pass runs in one commit with nothing waiting on an earlier result.

**As coded (2026-11-13): plan `pass4`, 75 test-scored runs** (A10-A14 at 1% and 10%, A15 at 10%, the re-runs and the sensitivity runs above), A10 at 100% moved to the optional plan `pass4_full100` (5 runs). Time (INFERRED from measured Pass 1-3 run times: pair arm about the cost of a helper arm, 0.7 min at 1% and 2.1 min at 10%; A11 about 3.3 min; explanations about +0.7 min): about 115 run-minutes plus start-up, so **about 1.3-1.5 h of a T4 x2 session** for `pass4`; `pass4_full100` adds about 55 min. Owner's remaining quota on 2026-11-13: about 3 GPU-h.

**Evaluation (same as Pass 3):** Headlines test macro-F1; cue-masked and random-masked test (`eval_masked`); iSarcasmEval test AUC with bootstrap intervals (in-domain reference 65.8, [[Pass 3 Results]]).

**Pre-set contrasts** (paired bootstrap, 10,000 resamples, as before; Holm over the Pass 4 contrasts as robustness column):
- A10 - A1 at 1/10/100; A11 - A7 and A11 - A1 at 1/10; A11 - A6 at 1 (the strongest 1% baseline).
- A10 - A12 at 1/10 (pairing); A13 - A14 at 1/10 (filtering); A14 - A10 at 1/10 (subset and model mix).
- Cue-mask gap and iSarcasmEval AUC: A10 and A11 vs A1 and A7. A15 reported descriptively next to A1 at 1% and 10%.

**Predictions written before running (IDEA):** (H4.1) A10 > A1, largest at 1%, near zero at 100%. (H4.2) A10 has a smaller cue-mask gap than A1. (H4.3) A10's iSarcasmEval AUC is above A1's, below 65.8. (H4.4) A10 > A12; if not, the claim is "generated text helps", not "counterfactual pairs help", and the paper says so. (H4.5) A13 ≈ A14 (the judge is noisy; [[Pairgen Full Run Results]]). (H4.6) A11 ≥ A7.

**Main Pass 4 model for the explainability step:** A10 at 10% (the cleanest pair effect), fixed now; A11 too if time allows.

**Extra check (Mahato 2026 concern, [[Novelty Check - LLM Counterfactuals]]):** TF-IDF + logistic regression that separates generated from real train headlines (CPU, minutes). Reports how visible "LLM style" is; within the pairs it cannot predict the label, since both sides are generated. Optional: Unfun human counterfactual test set, only if the owner downloads it (licence and overlap check first).

**Code (written 2026-11-13, not run on Kaggle yet):** `config.py` (arms A10-A15, `pair_batch`, `explain`), `data.py` (pair loader with SHA-256 checks, `matched` / `unpaired` subsets with seed 2027, leak checks, `preflight`), `train.py` (`GroupBatchStream` keeps pairs whole; A15 trains on pairs only; explain step), `explain.py` (Integrated Gradients written directly in torch, all-padding baseline, 32 steps, 500 fixed test headlines, completeness check), `plans.py` (`pass4`, `pass4_full100`), `analysis_pass4.py` (contrasts + Holm, gap and AUC differences, cue-share bootstrap, re-run check), `pair_checks.py` (CPU: style probe, real-trained reader, cue-word and length shortcuts). Notebooks: `08_pass4.ipynb` (GPU, one commit, CPU pre-flight first) and `09_pass4_analysis.ipynb` (CPU on Kaggle, after: pair checks + analysis; notebook 07 merged into it 2026-11-13). Tests: `tests/test_pass4.py`, `tests/test_pass4_train.py`.

**Fixed step: explainability check (CONFIRMED by owner 2026-11-12).** After Pass 4 training, on Kaggle:
- Method: Integrated Gradients token attributions (Captum) for the predicted class, all-padding baseline, on a fixed, seeded, class-balanced sample of the Headlines test set (size set with the training plan).
- Models: A1 (baseline) and the main Pass 4 model, same budget and seeds 0-4; others if time allows.
- Main number per model: share of absolute attribution on the top-100 outlet cue words (`cue_words.json`), averaged over headlines and seeds, with a bootstrap interval for the difference from A1.
- Prediction written down before running (IDEA): the pair-trained model puts a smaller share on cue words than A1, in line with a smaller cue-mask gap ([[Pass 3 Results]]).
- Paper: the cue-share number in the results table, plus 2-3 highlighted example headlines (same headlines for both models).
- Not used: attention weights as explanations; LIME/SHAP (slower, noisier on short text). Background: [[Research Ideas]] "Explainability".

## Pass 5 plan (written 2026-11-14, not run; owner's idea, DELEGATED design per [[Repo Rules]] rule 7)
Pass 4's main problem ([[Pass 4 Results]]): mixing the generated pairs in at full weight, unfiltered, costs 15-16 points of in-domain macro-F1 at 1% and 10% (A10/A11 vs A1/A7), because the pairs carry a different, AI-specific notion of "sarcastic" ([[Pass 4 Results]] "Shortcut-pattern hunting in the generated pairs": the AI leans on stock registers like "finally", "groundbreaking", "shockingly" vs. "discusses", "explores", "reportedly", near-absent from real headlines). Pass 5 asks whether removing that signal from the data, or simply trusting the pairs less during training, recovers the in-domain loss without losing the (partly length-explained, [[Pass 4 Results]] "iSarcasmEval length-shortcut check") transfer gain.

**Two independent knobs, both added to the config (not new arms by themselves):**
1. `pair_weight` (0-1, default 1.0 = Pass 4's unweighted behaviour): multiplies the pair rows' share of the sarcasm loss. The real L rows always keep weight 1.
2. `cleaned_drop_pct` (new pair set `"cleaned"`, arms A16 = A10 + cleaning, A17 = A11 + cleaning): drops the worst `cleaned_drop_pct`% of seeds by a per-seed shortcut score (`pair_checks.shortcut_scores`, the same model as `shortcut_word_table`; the max of a seed's two rows, so a pair is always dropped or kept whole). The score file (`pair_shortcut_scores.csv` + manifest) is a **new precomputed data artifact**, built once in notebook 09 section A5.3, added to `gainsplit-processed` the same way `cue_words.json` was (no model refit on GPU time).

**Tuning (dev only, before the main plan, same pattern as `w_tuning`/`fgm_tuning`):**
- `pair_weight_tuning`: A10 at 10%, `pair_weight` in {0.1, 0.3, 1.0}, 3 seeds (9 runs). Winner sets `PASS5_PAIR_WEIGHT`.
- `cleaned_drop_tuning`: A16 at 10% and the frozen `pair_weight`, `cleaned_drop_pct` in {10, 20, 30}, 3 seeds (9 runs). Winner sets `PASS5_CLEANED_DROP_PCT`.

**Main plan `pass5` (30 test-scored runs, 5 seeds, budgets 1% and 10%; needs both constants set first):**
| Arm | What it is | Isolates |
|---|---|---|
| A16 | A10 (pairs + plain BERT) with the cleaned set and the frozen weight | both fixes together |
| A17 | A11 (pairs + FGM) with the cleaned set and the frozen weight | both fixes, stacked with FGM |
| A10 (tag `weighted`) | A10, unfiltered pairs, frozen weight only, no cleaning | the weight knob alone, isolated from cleaning |
A16/A17 vs A10(weighted) then separates "does cleaning add anything on top of the weight" from "does the weight alone already fix it". Baselines A1, A7 reused from Passes 1-2 (training deterministic, checked in every pass since Pass 3).

**Predictions (IDEA, written before running):** (H5.1) some pair_weight < 1 recovers most of the in-domain loss (A10-weighted much closer to A1 than Pass 4's A10). (H5.2) cleaning adds a further, smaller recovery on top of the weight (A16 > A10-weighted). (H5.3) the iSarcasmEval transfer gain shrinks as pair_weight drops, trading in-domain accuracy against transfer, i.e. no free lunch; Pass 5 picks a point on that trade-off rather than eliminating it. Any outcome is reportable, same rule as Pass 1 ([[Decisions]]).

**Code (written 2026-11-14, not run on Kaggle; local non-torch tests 156 passed/2 skipped, torch smoke tests in a scratch CPU install 219 passed, 0 skipped incl. a byte-identical check that `pair_weight=1.0` reproduces every existing Pass 4 run exactly):** `config.py` (`pair_weight`, `cleaned_drop_pct` fields; pair set `"cleaned"`; arms A16, A17), `data.py` (`load_shortcut_scores`, `cleaned_subset`), `train.py` (per-row loss weight threaded through `fit`/`plain_step`/`fgm_step`, `None` whenever `pair_weight == 1.0` so Pass 1-4 code paths are byte-identical), `pair_checks.py` (`_fit_shortcut_model` shared fit, `shortcut_scores`), `plans.py` (`pair_weight_tuning`, `cleaned_drop_tuning`, `pass5`). Notebook 09 section A5.3 exports `pair_shortcut_scores.csv`.

**Tuning notebook (`experiments/10_pass5_tuning.ipynb`, written 2026-11-14):** GPU, two stages run one after the other (`cleaned_drop_tuning` needs `pair_weight_tuning`'s winner frozen into `PASS5_PAIR_WEIGHT` and pushed first, same dependency as `fgm_tuning` -> `pass2b`). Interactive session, no commit needed (18 runs total, small and quick); reuses `02_train.ipynb`'s structure (dev-only summary table), not `08_pass4.ipynb`'s (which assumes test scoring and would break on a dev-only plan's missing `test`/`isarc_test` score keys).

**`pair_weight_tuning` result (run 2026-11-14 by the owner, notebook 10 stage 1, commit b7018f6, 9/9 runs, 13.2 min, 219 tests passed on Kaggle; dev only, VERIFIED from `summary.csv` in `results_pair_weight_tuning.zip`):**
| pair_weight | dev macro-F1 seed 0 / 1 / 2 | mean (sd) |
|---|---|---|
| 0.1 | 76.63 / 76.08 / 78.36 | **77.02** (1.19) |
| 0.3 | 73.55 / 74.18 / 74.96 | 74.23 (0.71) |
| 1.0 | 71.72 / 71.46 / 73.14 | 72.11 (0.90) |
| A1 reference (Pass 1, same seeds, no pairs) | 86.45 / 86.39 / 87.34 | 86.73 |
- Sanity check: the `pair_weight=1.0` runs reproduce Pass 4's A10 dev scores for seeds 0-2 exactly (71.72 / 71.46 / 73.14), so the new weighting code leaves the old path unchanged on GPU too.
- By the pre-set rule, 0.1 wins (best mean, best on every seed, no tie).
- Reading (INFERRED): dev F1 rises steadily as the weight falls, and 0.1 is the edge of the grid. 0.1 recovers about a third of the in-domain loss (4.9 of 14.6 points) and is still 9.7 points below A1, so H5.1 ("most of the loss recovered") looks unlikely at this weight. Because the tuning score is in-domain dev F1 only, any lower weight would probably score higher still, ending at weight 0 = A1. This rule therefore cannot find the trade-off point H5.3 is about. The transfer side (iSarcasmEval AUC) is only measured in the test-scored `pass5` runs.
- Frozen 2026-11-14: `PASS5_PAIR_WEIGHT = 0.1` (pre-set rule kept; the trade-off is reported from the test-scored `pass5` runs). Notebook 10 now defaults to `cleaned_drop_tuning`.

**`cleaned_drop_tuning` result (run 2026-11-14 by the owner, notebook 10 stage 2, commit 1c4ce93, 9/9 runs, 13.2 min, 219 tests passed on Kaggle; dev only, VERIFIED from `results/cleaned_drop_tuning/summary.csv`):**
| cleaned_drop_pct (weight 0.1) | dev macro-F1 seed 0 / 1 / 2 | mean (sd) |
|---|---|---|
| 10 | 77.11 / 76.49 / 77.11 | 76.90 (0.36) |
| **20** | 77.42 / 77.54 / 77.41 | **77.45** (0.07) |
| 30 | 76.12 / 76.69 / 76.89 | 76.57 (0.40) |
| 0 (no cleaning, from stage 1) | 76.63 / 76.08 / 78.36 | 77.02 (1.19) |
- By the pre-set rule, 20 wins (best mean; 10 is 0.55 below, more than 1 sd, so no tie). Frozen as `PASS5_CLEANED_DROP_PCT = 20.0`. The 20% cleaned set has 17,934 seeds (35,868 rows).
- Reading (INFERRED): cleaning adds very little on top of the weight. 20% is +0.43 over no cleaning on average, better on 2 of 3 seeds (+0.79, +1.46, -0.95), and all three drop levels sit within about 1 point of each other. The effect is noise-sized on dev; `pass5` (5 seeds, test, A16 vs A10-weighted) will say whether it is real. H5.2 looks weak.

**Main-run notebook (`experiments/11_pass5.ipynb`, written 2026-11-14):** copy of `08_pass4.ipynb`'s structure with `PLAN = "pass5"` (30 runs), prints the two frozen settings in the pre-flight, and the first-look cell shows the Pass 5 arms next to A1/A7 (Passes 1-2) and unweighted A10/A11 (Pass 4) read from the repo clone. The baseline part was dry-run locally against the committed summaries (A1 10% 88.21, A10 10% 73.38, matching [[Pass 4 Results]]). No analysis notebook for Pass 5 contrasts yet.

**Main run done 2026-11-14** (notebook 11, 30/30 runs; [[Pass 5 Results]]). **Not yet started:** a CPU analysis of its contrasts (A16 vs A10-weighted, A10-weighted vs A10 and A1, A17 vs A11 and A7, iSarcasmEval AUC vs the length-only baseline).

## Pass 6 plan (written 2026-11-14, run 2026-11-14, results in [[Pass 6 Results]]; owner chose two-stage training after [[Pass 5 Results]]; design DELEGATED per [[Repo Rules]] rule 7)
Passes 4-5 mixed the generated pairs into every training step; the model lost 10-16 points on Headlines at every weight tried, and once down-weighted its iSarcasmEval transfer fell to the length-only baseline. Pass 6 separates the two sources in time: **stage 1** trains on the pairs alone, **stage 2** fine-tunes the same model on the gold labels alone, so the gold labels have the last word. Idea listed in [[Research Ideas]] "After Pass 4" ("two stages").

**Arms (new switch `pretrain`, new config field `pretrain_steps`):**
| Arm | Stage 1 | Stage 2 | Compare with |
|---|---|---|---|
| A18 | unfiltered pairs only, 720 steps (exactly A15's training) | L only, plain BERT, usual steps | A1 (in-domain), A15 and the length baseline (transfer) |
| A19 | the same | L only, BERT + FGM (epsilon 0.5), usual steps | A7 |

**Design choices (DELEGATED):**
- Stage-1 length 720 steps at both budgets = A15's steps at 10%. Stage 1 is then the exact Pass 4 A15_p10 model for each seed (AUC 76.2 on iSarcasmEval), so no tuning run is needed, and stage 2 starts from a known model. Every run records its stage-1 dev score; the notebook checks it equals A15's (a re-run check, like Pass 3-4's).
- Stage 2 keeps the stage-1 sarcasm head (same task) and starts a fresh optimizer and warm-up schedule; step count by the usual rule, so stage 2 is A1/A7's training budget exactly.
- Unfiltered pairs at full weight in stage 1 (cleaning and weighting did nothing useful in Pass 5); no tuning plan.
- Plan `pass6`: 20 test-scored runs (2 arms x {10%, 1%} x 5 seeds), masked test sets scored; about 40 min on T4 x2, one commit. Notebook `experiments/12_pass6.ipynb`.

**Predictions (IDEA, written before running):** (H6.1) A18 stays within 1 point of A1 on Headlines test macro-F1 at 10% and 1%, because stage 2 is A1's full training. (H6.2) A18's iSarcasmEval AUC falls well below A15's 76.2 as stage 2 overwrites the pair training; it is a gain only if it stays clearly above the length-only baseline 68.7. (H6.3) A19 is within 1 point of A7 in-domain. A useful method needs H6.1 and an AUC clearly above 68.7 at the same time; otherwise the pair results go into the paper as a negative finding. Any outcome is reportable ([[Decisions]]).

**Code (written 2026-11-14, not run on Kaggle):** `config.py` (arms A18/A19, `pretrain_steps` with checks), `train.py` (`fit(model=...)` continues an existing model; `train_run` runs stage 1, scores it on dev, then stage 2 on L only; `result.json` gets a `pretrain` block), `plans.py` (`PASS6_PRETRAIN_STEPS = 720`, plan `pass6`). Tests: local non-torch suite 163 passed, 2 skipped; with torch (scratch CPU install) 229 passed, including a check that stage 1 gives exactly A15's dev scores and that stage 2 changes the model and differs from A1; 1 failure from matplotlib missing in that scratch install only (the same test passes in the normal local suite and on Kaggle). Notebook 12's first-look cell dry-run locally on stand-in files.

## Pass 7 plan (written 2026-11-14, run 2026-11-14; owner's request: can a full-label run pass 94% accuracy on Headlines; design DELEGATED per [[Repo Rules]] rule 7)
Every run so far except Pass 1's A1 used 1-30% of the labels. A1 at 100% reaches 93.35% test accuracy (best seed 93.90%, VERIFIED from `results/pass1/summary.csv`), level with the seed paper's reported 93.88% on its own 80/20 split ([[Seed Paper - Irfan 2025]]; not like-for-like, see [[Decisions]] 2026-11-11). FGM and the two-stage pair recipe have never been tried with all labels.

**Plan `pass7` (11 test-scored runs at budget 100, masked tests scored, one commit, about 2 h on T4 x2, notebook `experiments/13_pass7.ipynb`):**
| Arm | What it is | Compare with |
|---|---|---|
| A7 x5 | all labels, FGM (epsilon 0.5) | A1 at 100% (Pass 1) |
| A19 x5 | stage 1 pairs only (= A15, 720 steps), stage 2 all labels + FGM | A7 at 100% (isolates the pair stage) |
| A1 seed 0, tag `rerun` | all labels, plain BERT | must equal Pass 1's A1_p100_s0 exactly (re-run receipt) |

**Rules (DELEGATED):** nothing is tuned and nothing is picked on test; epsilon 0.5 and the stage-1 length stay frozen from Passes 2b and 6. The result is reported as the 5-seed mean with its spread, not the best seed. The seed paper's 93.88% is quoted as a reference on a different split, not as a beaten baseline. A stronger backbone (e.g. DeBERTa-v3) is left out: it changes the model, not the method, and would need its own lr check on dev.

**Predictions (IDEA, written before running):** (H7.1) A7 beats A1 at 100%, but by less than at 10% (+0.3 to +0.8 points, so about 93.7-94.2% accuracy). (H7.2) A19 is within 0.3 points of A7: with 7,130 stage-2 steps the pair stage is mostly overwritten (Pass 6: the pair effect shrank as stage 2 grew). (H7.3) iSarcasmEval AUC of A7 and A19 stays near chance, as for A1 at 100% (51.7). Reaching 94.0% mean accuracy is possible but not expected with bert-base; any outcome is reportable.

**Outcome (2026-11-14, [[Pass 7 Results]]):** 11/11 runs, re-run and stage-1 checks PASS. A7 93.90% mean accuracy (+0.55 over A1, 5/5 seeds; best seed 94.32%), A19 93.67% (-0.23 vs A7), iSarcasmEval AUC 53.8 / 57.5. H7.1 holds, H7.2 holds narrowly, H7.3 mostly (A19 keeps a small pair imprint). 94% is not reached as a 5-seed mean.

## Bootstrap plan for Passes 5-7 (written 2026-11-14, before running; not run)
Same method as [[Bootstrap - All Passes]] and Pass 4: paired bootstrap over test items, 10,000 resamples, one resample shared by every seed, statistic = mean over seeds 0-4; Holm correction within each table. Code `src/gainsplit/analysis_pass5_7.py` (tests `tests/test_pass5_7.py`), CPU notebook `experiments/14_bootstrap_pass5_7.ipynb`, output `results/analysis_pass5_7/`. All contrasts are fixed here; none is added after seeing results.

| Table (Holm family) | Contrasts |
|---|---|
| `contrasts_f1` (19) | Pass 5 at 1% and 10%: A10[weighted]-A1, A10[weighted]-A10, A16-A10[weighted] (cleaning), A17-A16 (FGM on top), A17-A7. Pass 6 at 1% and 10%: A18-A1, A19-A7, A19-A18. Pass 7 at 100%: A7-A1, A19-A1, A19-A7 |
| `contrasts_accuracy_100` (3) | the three Pass 7 contrasts on accuracy |
| `auc_contrasts` (12) | iSarcasmEval AUC: A10[weighted]-A10, A10[weighted]-A1, A17-A7, A18-A1, A19-A7 at 1% and 10%; A7-A1 and A19-A7 at 100% |
| `auc_vs_length` (22) | iSarcasmEval AUC of every pair arm of Passes 4-7 minus word count alone, paired on the same items (new: the transfer claim net of length) |
| `gap_contrasts` (10) | cue-mask gap: A10[weighted], A16, A18 vs A1; A17, A19 vs A7; at 1% and 10% |

Checks first: 41 re-runs identical to their originals, 25 stage-1 models equal to A15 (both confirmed on the committed files with a structure check that printed row counts only).

**Expectations (IDEA, from the seed-paired means):** A7-A1 at 100% (+0.56, 5/5) is significant before Holm, borderline after; A19-A7 at 100% is not. A18-A1 and A19-A7 at 10% are not significant; at 1% they are negative and significant. Only A15 (pairs only, 76.2) may sit clearly above word count (68.7); the arms with gold labels in Passes 5-7 do not.

## Status
Code rolled back 2026-11-09 by the owner's decision: `src/`, `experiments/` and `tests/` are empty again and the implementation will be rebuilt from scratch. The design above (arms, budgets, protocol) still stands unless changed in [[Decisions]]. Earlier implementation notes (split sizes, A5 reading, smoke tests) were discarded with the code. Log runs in [[Session Log]].

- 2026-11-09: preprocessing rebuilt as one Kaggle notebook, `experiments/01_preprocess.ipynb` (owner runs it; GPU + internet on). Outputs: headlines train/dev/test, `budgets.json` (5 seeds x 5 budgets, nested), helper labels (hard + soft, plus raw 28 emotion scores), iSarcasmEval train/test, `manifest.json` with counts and hashes. Choices in [[Decisions]]. Not yet run; code cells syntax-checked and the cleaning and budget functions tested on synthetic data only.

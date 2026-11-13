---
tags: [log]
updated: 2026-11-13
---

# Session Log

Newest first. Hub: [[00 - Project Context]]. Format: [[Templates/Session Entry]].

## 2026-11-13 - iSarcasmEval length-shortcut check written
- Owner asked to write the length check flagged as open in [[Pass 4 Results]]. Written: `pair_checks.isarc_length_shortcut` (descriptive AUC of word count vs the real iSarcasmEval test labels) and `analysis_pass4.isarc_length_auc` (bootstrap 95% CI over the exact items a run was scored on), wired into notebook 09 (section A prints the descriptive table, section E's `run_all` writes `isarc_length_shortcut.csv/.md`). Logged in [[Pass 4 Results]] "iSarcasmEval length-shortcut check".
- 9 new/extended tests; full suite 137 passed, 2 skipped (torch, as before), locally.
- Not run: needs the owner to push and re-run notebook 09 (CPU only, no new data, same inputs as the Pass 4 analysis run).

## 2026-11-13 - Pass 4 results read
- Owner ran notebook 08 (commit 51fea9b, GPU T4 x2, 75/75 runs, 76.2 min, 193 tests passed, re-run check PASS) and notebook 09 (Kaggle CPU, 330 runs) and sent `results_pass4.zip`, `analysis_pass4.zip` and both logs. Read in [[Pass 4 Results]].
- Pairs as extra labeled rows: -16.5 F1 at 1%, -14.8 at 10% (A10 vs A1); FGM + pairs likewise. Pairing adds nothing; judge filter +0.8 / +1.2 (significant after Holm). H4.1, H4.4, H4.5, H4.6 rejected.
- iSarcasmEval AUC: every pair arm 72-74 vs about 50 for A1/A7; pairs-only A15 76.2, above the in-domain reference 65.75. Possible length shortcut (generated sarcastic sides longer, length-only AUC 0.785 in the pairs): check before claiming.
- IG completeness gap 0.37-0.54 (32 steps too few); cue share A10 0.095 vs A1 0.115, approximate.
- Zips and logs moved from Downloads into `results/pass4/` and `results/analysis_pass4/` at the owner's request and unpacked; every file matches its zip. The two `download*.txt` files were exact copies of the logs and stay in Downloads.
- Next: length check on iSarcasmEval test (CPU); choose the next step ([[Research Ideas]] "After Pass 4").

## 2026-11-13 - Kaggle upload folder prepared
- Owner asked to run everything on Kaggle and to get only the files for the input and the notebooks to import. Owner's request, so the files were assembled in this session (copy only, no content changed).
- `data/kaggle_upload/gainsplit-processed/` (git-ignored, 20 files, 22 MB): the 16 files of `data/processed/` plus `pairs_train.csv` + `manifest_pairs.json` (from the finalize zip, `pairs_train.csv` SHA-256 6412c1fa..., same as judge v2 checked) and `pairs_train_filtered.csv` + `manifest_judge_v2.json` (from `judge_v2.zip`). All files pass `verify`, `verify_pass3` and the Pass 4 pre-flight (unfiltered 44,834 rows; filtered and matched 34,630 with Gemma 7,533 / Mistral 4,986 / Qwen 4,796; unpaired 22,417).
- Notebook 07 merged into 09: `09_pass4_analysis.ipynb` now runs the pair checks and the full Pass 4 analysis on Kaggle CPU (accelerator None), with inputs `gainsplit-processed` + the output of notebook 08; baselines come from the repo clone.
- Next (owner): upload the folder as a new version of `gainsplit-processed`; push; import `08_pass4.ipynb` (GPU T4 x2, commit) and then `09_pass4_analysis.ipynb` (CPU).

## 2026-11-13 - Pass 4 code written (one commit)
- Owner asked for all Pass 4 work that can run in parallel in one push, nothing that waits on an earlier result, and a time budget (about 3 GPU-h of quota left). `pair_pilot` dropped, pair batch fixed at 32 rows; A10 at 100% moved to optional `pass4_full100` ([[Decisions]]).
- Written: arms A10-A15 (`config.py`), pair loader with hash and leak checks and the seeded `matched` / `unpaired` sets (`data.py`), `GroupBatchStream` and pairs-only training (`train.py`), Integrated Gradients inside the run (`explain.py`), plans `pass4` (75 runs) and `pass4_full100` (5), CPU analysis (`analysis_pass4.py`) and CPU pair checks (`pair_checks.py`); notebooks `07_pair_checks` (CPU), `08_pass4` (GPU, CPU pre-flight first), `09_pass4_analysis` (CPU). Details in [[Experiment Plan]] "Pass 4 plan".
- Checked locally: 135 tests pass without torch; with a CPU torch install in a scratch environment, 91 tests incl. tiny-BERT training of every Pass 4 arm pass; the old and new code give identical predictions (difference 0) for A1, A3, A4, A6, A7 on a tiny BERT. Not run on Kaggle; nothing committed.
- Time (INFERRED from measured Pass 1-3 run times): `pass4` about 1.3-1.5 h of a T4 x2 session; `pass4_full100` about 55 min more.
- Next (owner): file `pairgen` and `judge_v2` zips in `results/`; add `pairs_train.csv`, `manifest_pairs.json`, `pairs_train_filtered.csv`, `manifest_judge_v2.json` to a new `gainsplit-processed` version; run notebook 07 locally (CPU); commit and push; run notebook 08 (Save & Run All, then stop the interactive session); send `results_pass4.zip` and the log; run notebook 09 locally.

## 2026-11-13 - Pass 4 training arms planned
- Owner asked for the Pass 4 plan. Written to [[Experiment Plan]] "Pass 4 plan" and logged in [[Decisions]] (DELEGATED).
- Pairs from all train seeds at every budget (prompt never shows the seed label); pair batch after the L batch, whole pairs together, one sarcasm loss, equal steps.
- Arms: A10 pairs (1/10/100%), A11 FGM + pairs, A12 unpaired control, A13 judge-filtered, A14 model-matched random subset, A15 pairs only; one A1 re-run check; 1% sensitivity with 3x steps. `pair_pilot` (6 dev runs) picks pair batch 16 vs 32 first. 71 test-scored runs, about 4 GPU-h (INFERRED). Contrasts and predictions written before any run.
- Next: write the Pass 4 code (config, data, train, plans, analysis, tests) and review it; owner adds `pairs_train.csv`, `pairs_train_filtered.csv` and both manifests to `gainsplit-processed`, then runs `pair_pilot`.

## 2026-11-13 - Judge contest v2 results
- Owner ran notebook 06 (commit ac5609f, 5.0 h) and sent `judge_v2.zip` and the log. Read in [[Pairgen Full Run Results]]; all hashes match.
- Scoring removed the SARCASTIC bias. Only Gemma with prompt v2 passes (report balanced accuracy 0.773, AUC 0.862); next best Gemma v1 0.745, Mistral v2 0.713; irony classifier at chance.
- Filtered set: 17,315 of 22,417 pairs (Qwen 65% and Mistral 67% kept; Gemma pairs unjudged, all kept). Rejections are mostly serious sides read as sarcastic, at about the judge's false-positive rate on real serious headlines (0.235).
- Ablation caveat: filtering also shrinks the set and raises Gemma's share (34% to 43%); a size- and model-matched random subset is needed as a control (IDEA).
- Next (owner): file the zip and log in `results/judge_v2/`; add `pairs_train_filtered.csv` + `manifest_judge_v2.json` to `gainsplit-processed`. Then plan the Pass 4 training arms.

## 2026-11-13 - Judge contest v2 written
- Owner asked for a better judge. Written: `src/gainsplit/judge.py`, `tests/test_judge.py` (24 tests pass with pairgen) and `experiments/06_judge_v2.ipynb`: score-based judges (log-probability of SARCASTIC minus SERIOUS) with a threshold set on 400 real train headlines and pass/fail on 400 others, prompt v2 with a format note and 3 + 3 real examples, plus an off-the-shelf irony classifier. Details in [[Pairgen Full Run Results]].
- Dry run on made-up data passes both paths; GPU cell not run.
- Next (owner): run notebook 06 on Kaggle (inputs `gainsplit-processed` + the notebook 05 finalize output; T4 x2, Internet, `HF_TOKEN`; Save & Run All (Commit); about 1-4 h); send `judge_v2.zip` and the log. Push the code first (Kaggle clones the repo).

## 2026-11-12 - Explainability added to Pass 4
- Owner asked for explainable models and confirmed the proposal: Integrated Gradients attributions for A1 and the main Pass 4 model; main number is the share of attribution on outlet cue words ([[Experiment Plan]] "Pass 4 plan", [[Decisions]]). Not written or run yet; it needs the Pass 4 models.

## 2026-11-12 - Pair dataset finalized
- Owner ran finalize (2.9 h) and sent the zip and log. Read in [[Pairgen Full Run Results]].
- No judge reached 0.75 on real headlines (best SmolLM3 0.507); all judges say SARCASTIC to almost everything. By the rule set before the run, no judge filter: `pairs_train.csv` has all 22,417 checked pairs (44,834 rows). Hashes, balance and no-leak checks pass.
- Judges still separate generated sides in the intended direction (Mistral: 0.88 vs 0.17 SARCASTIC rate), more clearly than real headlines (0.79 vs 0.45).
- Next (owner): file the zip in `results/pairgen/`, add `pairs_train.csv` + `manifest_pairs.json` to a new `gainsplit-processed` version; then Pass 4 training plan and code.

## 2026-11-12 - Full pair-generation output inspected
- Owner sent `pairgen.zip` from the full run. Read in [[Pairgen Full Run Results]].
- Generation complete: 22,800 / 22,800 seeds, 98% pass the automatic checks (22,417 pairs), no refusals; about 6.4 h of generation.
- Judge contest and finalize left no output (no `judge_contest.json`, `pairs_train.csv`, `manifest_pairs.json`); the zip also has a `timing.csv` the repo notebook does not write, so the Kaggle copy was probably edited.
- Owner explained: the run used an edited two-stage notebook (memory fix for Gemma) with `STAGE = "generate"`, so finalize had not run yet. Notebook read and compared with the repo version in [[Pairgen Full Run Results]]: same seed plan, prompt and checks; Gemma batch 16 and out-of-memory splitting; finalize collects votes from all four judges.
- Owner asked: the notebook that ran replaced `experiments/05_pairgen_full.ipynb` (same content; one-click version kept in git history at 7893def).
- Next (owner): same notebook, generation output added as input, `STAGE = "finalize"`, Save & Run All (Commit); send `pairgen.zip`. Leave `pairgen.py` unchanged until then.

## 2026-11-11 - Full run made one-click
- Owner asked for a single unattended run. `05_pairgen_full.ipynb` now runs in one commit: judge contest on 400 real train headlines first (SmolLM3-3B vs cross-voting; reused if a previous run saved it), then generation of all 22,800 seeds, then judging with the winner only, then the frozen `pairs_train.csv` + `manifest_pairs.json` (scores, counts, cue rates, 8 example pairs, SHA-256). Generation stops itself after 9.5 h so output is kept; a second commit with the first output as input finishes the job. No settings to change. Dry-run on made-up data (stop-and-resume path and finish path).
- Next (owner): Save & Run All (Commit); download `pairgen.zip`; send `manifest_pairs.json`.

## 2026-11-11 - Judge contest and sarcasm prompt v2
- Owner declined any human labelling ([[Decisions]]) and proposed SmolLM3-3B (HuggingFaceTB, Apache 2.0, not gated) as judge.
- Done: `pairgen.py` prompt version 2 (asks for obvious satire; four sharper invented examples) and judge-contest helpers (balanced accuracy with unclear = wrong, cross verdict, pick with threshold); 18 tests pass. `05_pairgen_full.ipynb` finalize now runs a judge contest on 400 real train headlines (200 per class): SmolLM3-3B vs cross-voting by the two generators that did not write the pair; best balanced accuracy >= 0.75 becomes the filter, otherwise no judge filter (all checked pairs kept). Blind audit sheet removed. Thinking mode turned off in chat templates (`enable_thinking=False`). Both paths dry-run on made-up data; GPU cells not run.
- Next (owner): trial (`N_SEEDS = 200`, generate then finalize), send the contest scores and tables; then the full run.

## 2026-11-11 - Pair-generation pilot results
- Owner ran the pilot (commit 11ee9ae); filed in `results/pairgen_pilot/`, read in [[Pairgen Pilot Results]].
- Generation works for all three models (no refusals, about 1-2.6 h per model for the full split); outlet-cue gap disappears in generated pairs. Phi judge is too weak (0.635 on real headlines, says SERIOUS by default); kept 26-33% mostly because of it. Sarcastic sides are mixed in strength.
- Next: owner fills the blind audit sheet; replace the judge and strengthen the sarcasm prompt before the full run.

## 2026-11-11 - Full pair-generation notebook written (pilot skipped by the owner)
- Owner chose to go straight to production (session usage limits). Written: `experiments/05_pairgen_full.ipynb` (whole train split by default, 3 generators, chunks of 500 saved per model, resumable across Kaggle sessions via earlier outputs added as input, seed-plan fingerprint check, finalize stage with checks, outlet-cue table, Phi judge on all generated headlines + 400 real train headlines, frozen `pairs_train.csv` + `manifest_pairs.json` with SHA-256, 200-item blind audit sheet). Helpers added to `src/gainsplit/pairgen.py`; 14 tests pass. Dry-run on made-up data across three simulated sessions; GPU cells not run.
- The notebook recommends a 200-seed trial in the same notebook (`N_SEEDS = 200`) before the full run.
- Time per model unknown until measured; printed per chunk.
- Next (owner): trial run, then one model per committed session, then finalize; send tables from steps 6-7 and the manifest.

## 2026-11-11 - Pass 4 direction approved; pair-generation pilot written
- Instructor approved the paired-generation direction (owner, 2026-11-11). Generators: Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct-v0.3 (open weights, Kaggle); judge: Microsoft Phi-3.5-mini-instruct (open, not gated, a family not used for generation). Claude Haiku was first planned as judge but dropped: the owner has no Anthropic API key, only a chat subscription. Hugging Face check: only Gemma is gated.
- Done: `src/gainsplit/pairgen.py` (seed picking, model assignment, prompt with invented few-shot examples, parsing, refusal and quality checks, outlet-cue rate, judge prompt, blind audit sheet) + `tests/test_pairgen.py` (10 tests pass locally); `experiments/04_pairgen_pilot.ipynb` (200 train seeds, 3 models, checks, Phi judge with greedy decoding, incl. judge accuracy on the real seeds, blind audit sheet, time estimate). Non-GPU cells dry-run on made-up data end to end; GPU cells not run.
- Open risk: Gemma 2 in float16 on T4 may give poor output (bf16 not supported on T4).
- Next (owner): commit and push (Kaggle clones the repo), attach secret `HF_TOKEN`, run the pilot (GPU T4 x2, internet on, input `gainsplit-processed`), download `pairgen_pilot.zip`, fill the blind audit sheet.

## 2026-11-11 - Instructor feedback: impact
- Instructor feedback on the draft: showing that the analysed model is weak is not enough; what new method gives more gain, and why should it be published?
- Done: reviewed results for positive directions. Proposed (IDEA, [[Research Ideas]] "Pass 4 candidates"): (1) stack FGM + self-training + helper heads on L; (2) VAT on U + FGM on L as the replacement for the GAN; (3) cue-dropout training against the outlet shortcut, tested on iSarcasmEval.
- Vault changes: [[Research Ideas]], hub.
- Follow-up request from the instructor: use generative AI (variants of GAN/BERT) and make the detector generalise to other datasets. Options logged in [[Research Ideas]] "Generative AI and generalisation".
- Novelty check of LLM counterfactual augmentation: [[Novelty Check - LLM Counterfactuals]]. Technique exists for sarcasm (Jia 2024 multimodal, Hong 2025 dialogues) and for headline humor (Horvitz 2024); gap left is the outlet-shortcut, low-label, cross-dataset angle. Unfun.me (human Onion-to-serious edits) found as a possible counterfactual test set.
- Second novelty search: Horvitz 2024 (ACL, full PDF read) pre-empts the Headlines counterfactual-test angle. Surviving candidate: label-free paired generation (LLM writes sarcastic + serious version of each unlabeled headline) as the SSL signal replacing the GAN generator. Details and risks in [[Novelty Check - LLM Counterfactuals]].
- Next (owner): pick a direction with the instructor; then Pass 4 plan, code, runs. Deadline 20 Dec 2026.

## 2026-11-11 - Engineering Research Express rules checked
- Done: read the IOP publishing-support pages for Engineering Research Express (author guidelines, About, ethical policy, data policy); compared them with the current draft.
- Vault changes: [[Audit - Venues]] (new VERIFIED block).
- Findings: scope note excludes AI/ML work without a clear engineering application (main risk); no page limit (the 9 pages are the instructor's rule); single- or double-anonymous by choice; AI disclosure, data availability statement and checked references required; no colour in tables; cannot be under review at QPAIN at the same time.
- Decision (owner): QPAIN first, ERX later as an extended version ([[Decisions]]).
- Next (owner): QPAIN submission as planned; later, agree an engineering framing for the ERX version with the instructor and decide on the disclosure statement.

## 2026-11-11 - Research diary and playbook
- Done: wrote up the whole research process from the vault history, for reuse on the next paper.
- Vault changes: new [[Research Diary]] (stages, mistakes and fixes, planning numbers) and [[Research Playbook]] (roles, setup, vault habits, protocol, Kaggle code pattern, delegated code sessions, paper checklist, starting brief); linked from the hub and [[Repo List]].
- Findings: none new.
- Next: unchanged (seed-paper reference, Overleaf compile, instructor review). For a new paper: paste the starting brief in [[Research Playbook]] section 9 into its first session.

## 2026-11-11 - Paper draft and Overleaf package
- Done: QPAIN 2027 rules checked on the live site (deadline 20 Dec 2026, 4-6 pages, IEEE template, double-blind, similarity <= 20% / 4%). Full draft written in `paper/` (IEEEtran, anonymous), with a TikZ model diagram, three data figures and two tables generated from `results/analysis/`; compiles to 6 pages. `paper/gainsplit_overleaf.zip` built for Overleaf.
- Vault changes: new [[Paper Draft]]; [[Audit - Venues]] (VERIFIED block), [[Decisions]] (2 DELEGATED), hub.
- Findings: none new; numbers in the text re-checked against the CSVs.
- Next: owner fills the seed-paper title and authors in `refs.bib`, uploads the zip to Overleaf (pdfLaTeX), reads the draft and sends it to the instructor; check the bibliography entries marked CHECK; run a similarity check before submission.

## 2026-11-11 - Pass 3 and GAN check filed
- Done: owner ran `pass3` (twice by mistake; both runs identical in all 1,575 metric cells) and `gan_check` on Kaggle, and re-ran `03_analysis.ipynb` locally on 255 runs. Zips filed in `results/pass3/` and `results/gan_check/` (105 and 12 runs, paths checked). Mask re-runs 25/25 exact; helper-label counts as expected; `gan_check` scored dev only.
- Vault changes: new [[Pass 3 Results]], [[GAN Check Results]]; updated [[Bootstrap - All Passes]], [[Risks]] (11, 13; new 14, 15), [[Decisions]] (5 DELEGATED entries), hub.
- Findings: shuffled helper labels at 1% still +1.17 over A1 (robust); at 10% only heavy label damage hurts; gold kappa does not order the label sources; outlet-cue gap ~3 points in every arm; in-domain iSarcasmEval AUC 65.8; GAN check failed the pre-set rule (no robustness run). Prediction files round probabilities (AUC source fixed in [[Decisions]]).
- Next: commit results + vault (owner approval); paper: check QPAIN 2027 rules ([[Audit - Venues]]), IEEE template in `paper/`, draft Data, Setup, RQ1; code session for the label-quality figure and paper tables.

## 2026-11-10 - Pass 3 data files made
- Owner checked the teacher-2 model card (3 classes) and ran `01b_pass3_data` on Kaggle (commit 654de15). Five files checked: hashes match `manifest_pass3.json`; masked tests keep ids, order and labels; 62% of test headlines have >= 1 cue word masked (1.6 words on average when hit); top cue words are outlet style (Onion: nation, onion, area, local, report; HuffPost: donald, huffpost, allegedly, why, the).
- Gold accuracy of the new sources added to [[Gold Audit Results]]: VADER kappa 0.26 (0.37 on sure items), teacher 2 kappa 0.14.
- Upload folder assembled on the owner's request: `data/gainsplit-processed-v2/` (14 files: the 9 original + 5 Pass 3; all hashes checked; git-ignored). Owner uploads it as a new version of `gainsplit-processed`, then runs `pass3`.
- GAN fairness check decided ([[Decisions]], DELEGATED) and coded in a separate session: `gan_check` plan, 12 dev-only runs (A4 generator lr 1e-5 / 2e-4 at 10%; A4 and A1 with 3x steps at 1%). Reviewed, merged; 87 tests pass. Owner runs it alongside or right after `pass3` (`PLAN = "gan_check"`).

## 2026-11-10 - Pass 3 planned
- Pass 2b and the full bootstrap committed and pushed (8847fd7).
- Pass 3 designed and logged ([[Experiment Plan]] "Pass 3 plan", [[Decisions]] DELEGATED): 3a CPU analysis (helper-to-sarcasm MI and probe, iSarcasmEval AUC intervals, gold table per label source); 3b new files (VADER, second sentiment teacher, outlet-cue masked test + random-mask control); 3c Kaggle plan `pass3`, about 105 runs (7 label rows x {A3 10%, A2 1%}, A9 = A2 shuffled at 1%, masked-test re-runs of A1/A3/A4/A6/A7 at 10%, A1 trained on iSarcasmEval).
- Three code sessions wrote the Pass 3 code: `experiments/01b_pass3_data.ipynb` + `src/gainsplit/masking.py`; training variants, A9, masked-test scoring, iSarcasmEval training and the `pass3` plan (105 runs); analysis for tagged runs, mask-rerun check, shortcut gap, label-quality contrasts, transfer AUC intervals, helper/gold checks. Reviewed, merged into main with two small fixes ([[Decisions]]); 77 tests pass. Notebook default `PLAN = "pass3"`.
- Next (owner): check the second teacher's model card, run `01b_pass3_data` on Kaggle, add its five files to `gainsplit-processed` as a new version, run `pass3` (~1.4 h), download `results_pass3.zip`.

## 2026-11-10 - Bootstrap on all passes
- Owner re-ran `03_analysis.ipynb` locally on pass1 + pass2a + pass2b: 150 test-scored runs (the "184" given earlier was wrong; the 9 dev-only `fgm_tuning` runs are excluded). Pass 1 contrasts reproduce exactly.
- VERIFIED ([[Bootstrap - All Passes]]): FGM +1.6 points over A1 at 1% and 10% (p < 0.0001, robust to Holm); GAN never helps (null at 1/5/10%, -0.55 at 30%); self-training +1.6 at 1% (robust), +0.6 at 10%; at 1% helpers help only on L (A2 +1.6; adding U -1.8); GAN + helpers at 1% -2.3; helpers on L + U not significant at any budget; A3 - A8 at 10% +0.57 (p 0.011, not robust to Holm).
- Holm robustness column added (DELEGATED, [[Decisions]]).
- Next: Pass 3 (RQ2 label-quality rows, RQ3 shortcut checks) and the paper draft (data, setup and RQ1 results can start now).

## 2026-11-10 - Pass 2b filed
- Owner committed the gold-audit vault edits (62d0020) and ran `pass2b` on Kaggle at that commit: 10/10 A7 runs, epsilon 0.5. Filed in `results/pass2b/`.
- Result ([[Pass 2b Results]], INFERRED): FGM gains +1.55 points over A1 at 10% on every seed (sd 0.15) and +1.6 at 1%; it beats GAN-BERT by 1.5-1.7 points while using no unlabeled data. Largest method effect in the study. No transfer to iSarcasmEval.
- The owner's analysis re-run used `RESULTS_DIRS = [pass2b]` only, so `results/analysis/` now holds A7-only tables and empty contrasts (A7 has nothing to compare with). The committed Pass 1 versions are in git. Not committed; to be replaced by a re-run on all three folders.
- Next (owner): re-run `03_analysis.ipynb` with `RESULTS_DIRS` = pass1 + pass2a + pass2b. Then Pass 3 and the paper draft.

## 2026-11-10 - Pass 2a and gold audit filed
- Filed `results_pass2a.zip` (unpacked) in `results/pass2a/` and `gold_sentiment_audit.csv` in `results/gold_audit/`. Pass 2a: 104/104 runs, 5 seeds in every planned cell.
- FGM epsilon 0.5 chosen by the pre-set tie rule (1.0 best mean, within 1 sd); `PASS2_FGM_EPSILON = 0.5`, notebook `PLAN = "pass2b"`; the plans test now resets the epsilon to None before checking the guard. 34 tests pass locally.
- Pass 2a readings (INFERRED, seed-paired; [[Pass 2a Results]]): helper label content drives the 10% gain (A3 - A8 +0.57, 5/5); at 1% helpers help only on L (A2 +1.6, 5/5) and adding U removes it; self-training A6 beats A1 at 1% (+1.6) and 10% (+0.6) on every seed; GAN null at 5% and -0.55 at 30% (0/5); helper direction +0.2 to +0.4 at 5/10/30%.
- Gold audit ([[Gold Audit Results]]): teacher accuracy 0.53, kappa 0.24 against the owner. Owner confirmed the unsure flag is literal; labels accepted (DELEGATED). First 50 items agree much less (0.30) than the rest (0.56-0.64): warm-up effect, reported as a limitation; owner declined a re-label of items 1-50. Teacher labels stay the helper training signal.
- Next (owner): run `pass2b` on Kaggle (10 runs, ~15 min); re-run `03_analysis.ipynb` locally with `RESULTS_DIRS` = pass1 + pass2a (+ pass2b when done); answer the unsure-flag question.

## 2026-11-10 - Pass 1 bootstrap
- Owner ran `experiments/03_analysis.ipynb` on `results/pass1`. A first Kaggle attempt failed with `KeyError: 'arm'`: on Kaggle `RESULTS_DIRS` is empty by default, so no runs were loaded. Re-run locally: 45 runs, 5 seeds per cell; output in `results/analysis/`.
- Bootstrap ([[Pass 1 Results]]): GAN null at 1% and 10% (VERIFIED); helper gain at 10% is +0.42 points, 95% interval [-0.00, 0.85], p = 0.051, so **not significant at 5%**; A5 harm at 1% is -2.3 points, p < 0.0001, with a significant negative interaction. No arm transfers to iSarcasmEval. GAN curves: at 1% (200 steps) the fake-probability on generated vectors is still rising (INFERRED caveat).
- Decisions logged ([[Decisions]], DELEGATED): report the 10% helper gain as not significant, no extra seeds; commit analysis outputs and the executed notebook.
- Pending: `results_pass2a.zip` is not on this machine yet (owner ran `pass2a`; download from the Kaggle output tab). Then: file it in `results/pass2a`, pick the FGM epsilon from `fgm_tuning`, set `PASS2_FGM_EPSILON` and `PLAN = "pass2b"`, read A2/A8/A6 contrasts and A6 pseudo-label counts.

## 2026-11-10 - Pass 1 results
- Owner ran `pass1` on Kaggle (commit c0a44a5): 45/45 runs, 114 GPU-min. Zip and unpacked files in `results/pass1/`. Numbers and reading in [[Pass 1 Results]].
- Main reading (INFERRED until the paired bootstrap runs): the GAN gives no gain at 1% or 10%; the helper heads give +0.4 points at 10% (4/5 seeds) and nothing at 1%; GAN + helpers at 1% loses 2.3 points; no arm transfers to iSarcasmEval (ROC-AUC about 0.5 everywhere).
- Decisions logged ([[Decisions]], DELEGATED): Pass 1 accepted, no reruns; Pass 2 split into `pass2a` and `pass2b`; Pass 2 priority (A2 and A8 first); arm specs for A2, A8, A6, A0; analysis notebook scope.
- Two code sessions wrote `src/gainsplit/analysis.py` + `experiments/03_analysis.ipynb` + `tests/test_analysis.py`, and the Pass 2 arms (A0 `tfidf.py`, A2, A6, A7, A8 in `train.py`; `metrics.py`, `util.py`; plans `fgm_tuning`, `pass2a` 104 runs, `pass2b`). Both reviewed; one fix: the analysis leaves dev-only tuning runs out of the tables (otherwise the 3 FGM epsilons per seed would count as extra seeds). Merged into main, 34 tests pass locally (torch tests run on Kaggle). Notebook default `PLAN = "pass2a"`.
- Next (owner): (1) run `experiments/03_analysis.ipynb` on `results/pass1` (CPU, minutes) and send the output folder; (2) run `pass2a` on Kaggle (T4 x2, Internet on, `gainsplit-processed`, Save & Run All; ~1 h 40 min, ~3.3 GPU-h) and send `results_pass2a.zip`; (3) keep labeling the gold audit. Then: pick the FGM epsilon from `fgm_tuning` (dev), set `PASS2_FGM_EPSILON` in `plans.py`, run `pass2b` (10 runs).

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
- Roadmap to submission written into [[Experiment Plan]]. All changes committed and pushed (96a2414).
- Owner delegated decisions ([[Repo Rules]] rule 7). Taken and logged in [[Decisions]]: 5 seeds in Pass 1, A7 spec, A3/A4/A5 implementation, `session2` plan, gold-audit protocol. Two code sessions wrote A3/A4/A5 + plans and `experiments/gold_audit.html`; both reviewed (one fix: feature-matching loss averaged, [[Decisions]]). Local pandas tests pass (9); torch smoke tests run on Kaggle. Committed and pushed.
- Gold audit: owner labels alone (extra-annotator idea dropped, [[Decisions]], [[Risks]] item 12). App fix: Forward now looks disabled until the current headline has a label (it only steps over labeled items; choosing a label moves on).
- `session2` ran (owner, commit 770f690): 12/12 runs, about 2 min each; results in `results/session2/`. w = 0.3 chosen (best on every seed); GAN pilot healthy (dev macro-F1 0.866 vs A1 0.867; p_fake 0.006 on real, ~0.6 on fake). Both DELEGATED, reasoning in [[Decisions]]. `PASS1_HELPER_WEIGHT = 0.3`, notebook default `PLAN = "pass1"`, summary now grouped by budget and shows test scores.
- Next: owner runs `pass1` (45 runs, test scored, Save & Commit, ~1 h) and keeps labeling the gold audit.
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

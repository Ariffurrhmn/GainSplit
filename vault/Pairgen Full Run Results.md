---
tags: [results, pass4]
created: 2026-11-12
---

# Pairgen Full Run Results

Hub: [[00 - Project Context]]. Earlier pilot: [[Pairgen Pilot Results]]. Output `pairgen.zip` from the owner's Kaggle run of 2026-11-12, inspected 2026-11-12 (not yet filed in `results/`). Numbers read from the files (VERIFIED); readings INFERRED.

## What the zip holds
- `seed_plan.csv` (22,800 seeds: 7,600 per model; 11,960 serious, 10,840 sarcastic), `plan_fingerprint.txt` (`5ff0dead7b8738c3`), `timing.csv`, and 16 raw chunks per model (Qwen, Gemma, Mistral).
- **Missing:** `judge_contest.json`, `judged_items.csv`, `pairs_all.csv`, `pairs_train.csv`, `manifest_pairs.json`. So neither the judge contest (step 1) nor the finalize step (step 3) left output.
- Explained (owner, 2026-11-12): the run used an edited notebook, `full-run-dataset (memory fix).ipynb` (now `experiments/05_pairgen_full.ipynb`), because the repo version ran out of GPU memory on Gemma. Read in full 2026-11-12 (VERIFIED). It is the earlier two-stage design (`STAGE = "generate"` / `"finalize"`) and ran with `STAGE = "generate"`, so the missing files are expected: the judge contest sits in the finalize stage.

## Edited notebook vs repo version (read 2026-11-12)
- Memory fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; Gemma batch size 16 (others 32); on CUDA out-of-memory a batch is split in half and retried. Both versions already spread a model over both GPUs (`device_map="auto"`).
- Same as repo: seed plan (same function, SEED, model list; fingerprint `5ff0dead7b8738c3`), prompt (from repo `pairgen.py`, version 2), checks, 400-headline contest, 0.75 threshold, sampling settings.
- Effects: Gemma's sampled text differs from what batch 32 would have given (batch size changes the random draws), not a quality issue; batch sizes are recorded in the manifest (`models`). No 9.5 h stop in generate, unneeded here (6.4 h).
- Finalize differs from the repo version: all four judges (SmolLM3 + three generators, none judging its own text) vote on every generated headline and on the 400 real ones; `votes.csv` keeps every vote, so the paper can report every judge's agreement. Manifest lacks the cue-rate block and example pairs (the cue table is displayed instead).
- Code check: every `pairgen` helper it calls exists in the repo with the expected behaviour; item ids of real (`<id>_real`) and generated (`<id>_s` / `<id>_n`) headlines cannot collide. The manifest will record the code commit at finalize time, not at generation; generation used the repo head at 04:06 on 2026-11-12, most likely 7893def (INFERRED). Keep `src/gainsplit/pairgen.py` unchanged until finalize is done.

## Generation
- Complete: all 22,800 seeds generated, one row each, model and seed text match the plan.
- Time per 500-seed chunk: Qwen about 340 s, Gemma about 505 s, Mistral about 655 s; total generation about 6.4 h.
- Automatic checks (`pairgen.check_pair`, run in memory on the raw chunks, no files written): pass rate Gemma 0.991, Mistral 0.982, Qwen 0.977; 22,417 pairs pass. No refusals, no empty or over-long outputs; failures are mostly copies of the seed (0.9-1.9%) and a few parse failures (Qwen, Mistral 0.4-0.5%).
- Mistral often writes several JSON pairs in one answer (2,143 outputs over 400 characters); the parser keeps the first, so this does not hurt the pass rate.
- Pair overlap (Jaccard) 0.17-0.20; sarcastic sides about 11-13 words, serious sides about 8.5-10.

## Finalize run (2026-11-12, `pairgen (1).zip` + log, read 2026-11-12)
- Code commit 7893def (same as generation); plan fingerprint matches; 22,800 seeds, none missing. Run time 2.9 h: judging SmolLM3 39 min (45,234 headlines), Qwen 42 min (30,390), Gemma 50 min (30,168), Mistral 44 min (30,310). Unclear answers 0-1.6%.
- **Judge contest: no candidate usable.** Balanced accuracy on 400 real train headlines: SmolLM3 0.507, cross without Qwen 0.427, without Gemma 0.483, without Mistral 0.498 (all below the 0.75 threshold). Per the rule fixed before the run, no judge filter: all 22,417 pairs that pass the checks are kept.
- Cause (from `votes.csv`): every judge answers SARCASTIC for almost everything. Share of SARCASTIC on real headlines: SmolLM3 0.98, Gemma 0.97, Qwen 0.87, Mistral 0.62. The pilot judge (Phi, same prompt) leaned the other way (SERIOUS). Single-judge balanced accuracy on real (computed here, not in the notebook): Mistral 0.67, Qwen 0.57, Gemma 0.53, SmolLM3 0.51. Cross-voting scores below 0.5 because a disagreement counts as wrong.
- Not a reading bug (INFERRED): the judges still separate the classes in the right direction, on real headlines (Mistral SARCASTIC rate 0.79 on real sarcastic vs 0.45 on real serious) and more strongly on generated sides (Mistral 0.88 on generated sarcastic vs 0.17 on generated serious; Qwen 0.99 vs 0.48). A batch/padding fault would scramble answers, not keep the direction. Likely cause: lowercase headlines without punctuation look odd and get read as satire (IDEA, untested).
- Reading (INFERRED): by the best single judge, the generated pairs differ in the intended direction more clearly than real headlines do. Whether the pairs help is decided by the Pass 4 training results, not by the judges.
- Dataset checks (run here on the files): all four SHA-256 values match the manifest; `pairs_train.csv` 44,834 rows, 22,417 per label, every seed has exactly one sarcastic and one serious row, all seeds from the train split; no generated headline is identical to any train, dev or test headline.
- Kept per model: Gemma 7,533, Mistral 7,462, Qwen 7,422.

## Judge improvement options (IDEA, 2026-11-12)
- Rule: a new judge is chosen only on real labelled train headlines (never on generated ones, never on dev/test), with the same 0.75 bar; threshold set on one set of real headlines, accuracy reported on a second.
- (1) Score instead of a one-word answer: probability of SARCASTIC vs SERIOUS from the next-token logits, threshold tuned on real headlines. Targets the observed bias directly (the judges rank in the right direction but all lean SARCASTIC). One forward pass per headline, no generation.
- (2) Better prompt: say that every headline, real or satirical, is lowercase without punctuation; add a few labelled real train headlines as examples.
- (3) Off-the-shelf sarcasm/irony classifier trained on other data (independent of our labels); accuracy on headlines unknown until tested.
- (4) Own fine-tuned detector (A1, 100% labels): likely passes 0.75 but leans on outlet words, which the generated pairs lack, and filtering with it is circular. Evidence only, not a filter.
- (5) Larger model of an unused family in 4-bit (e.g. a 14B model) if it fits two T4s; unverified.
- Keep the unfiltered set as the main dataset; a filtered set, if a judge passes, becomes a Pass 4 ablation (does filtering matter?).

## Judge contest v2: code written (2026-11-13, not run)
- `src/gainsplit/judge.py` (no GPU: real-headline sets, prompts v1/v2, threshold, AUC, judge per author, pair filter) + `tests/test_judge.py` (6 tests; 24 pass with `test_pairgen.py`). Existing `pairgen.py` functions unchanged.
- `experiments/06_judge_v2.ipynb`: 9 candidates (SmolLM3, Qwen, Gemma, Mistral, each with prompt v1 and v2, plus `cardiffnlp/twitter-roberta-base-irony`). Score = log P(SARCASTIC) - log P(SERIOUS) from one forward pass, no generation. Real sets drawn with seed 2027: 3 + 3 few-shot examples, tune 400, report 400, all from train, disjoint. Pass = report-set balanced accuracy >= 0.75; a generator never judges its own pairs; authors without a passing judge stay unfiltered and flagged. Checks the SHA-256 of `pairs_train.csv` against `manifest_pairs.json`.
- Outputs: `judge_v2_contest.json`, `real_scores.csv`, `gen_scores.csv`, `pairs_judged.csv`, `pairs_train_filtered.csv`, `manifest_judge_v2.json`, `judge_v2.zip`. The unfiltered `pairs_train.csv` stays the main Pass 4 dataset; the filtered one is an ablation.
- Dry run on made-up data with fake scorers: both paths work (a judge passes; none passes). The GPU scoring cell was not run (no torch locally). Expected time 1-4 h (INFERRED).

## Still to do
- Owner files `pairgen (1).zip` as `results/pairgen/pairgen.zip` (and the log), and adds `pairs_train.csv` + `manifest_pairs.json` to a new version of the Kaggle dataset `gainsplit-processed` for Pass 4.
- Owner runs `06_judge_v2.ipynb` and sends `judge_v2.zip` and the log.
- Paper: report the judge contest honestly (no judge reached 0.75; strong SARCASTIC bias of zero-shot LLM judges on this text format) and the single-judge separation as descriptive evidence.
- Done 2026-11-12: the notebook that ran replaced `experiments/05_pairgen_full.ipynb` (content unchanged, JSON re-indented). The one-click version (commit 7893def) stays in git history.

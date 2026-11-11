---
tags: [diary, process, meta]
created: 2026-11-11
updated: 2026-11-11
---

# Research Diary (GainSplit, Paper A)

Hub: [[00 - Project Context]]. The reusable process distilled from this diary: [[Research Playbook]]. Day-by-day detail: [[Session Log]]; every choice with its reason: [[Decisions]].

What happened, in order, how it was done, and what each stage taught. Written after the experiments finished and the first full draft existed (2026-11-11), so a later project can see the whole arc in one place.

## The project in one paragraph
A controlled empirical study of *where the gain comes from* in a published semi-supervised sarcasm-detection model (GAN-BERT + sentiment/emotion helper heads, [[Seed Paper - Irfan 2025]]). Instead of proposing a new method, we split the seed result into its ingredients (adversarial GAN training, helper heads, the unlabeled data the heads see, and the quality of the machine-made helper labels) and tested each under one protocol, across label budgets 1/5/10/30/100%, with controls (shuffled labels, a supervised adversarial arm, self-training). Target: IEEE QPAIN 2027. Outcome: 255 runs on free Kaggle GPUs, a 6-page draft and an Overleaf package, about four working days from empty repo to draft (2026-11-08 to 2026-11-11).

## Timeline

### Stage 0: Choosing the question (before 2026-11-08)
- Done: a literature deep dive ([[Deep Dive - Paper A]]) and a proposal ([[Proposal A]]). The first idea was a new method (sample weighting); it was judged to have about a 20% chance of passing as a contribution, so the project was reframed as a controlled ablation study ([[Decisions]], 2026-11-01).
- Why it worked: an ablation study of a published, cited result is reportable whatever the outcome. A null result ("the GAN gives nothing") is a finding, not a failure.
- Lesson: when a method idea is weak, ask "which published result has never been taken apart?" That question is cheaper to answer and harder to reject.

### Stage 1: Repo, vault and novelty check (2026-11-08, day 1)
- Repo layout fixed on day 1: `vault/`, `src/`, `experiments/`, `results/`, `data/` (git-ignored), `paper/`, `docs/`. Root instructions file plus [[Repo Rules]] and [[Vault Conventions]].
- Vault built from the proposal: hub note, literature notes, datasets, decisions, open questions, risks, session log, templates.
- Novelty check: the closest prior paper ([[Chen 2022]]) was read **in full**, not from the abstract; verdict "partial overlap only". The strongest overlap found was [[Dai 2021]]. Both recorded in [[Risks]].
- Datasets downloaded with hashes recorded ([[Headlines Dataset]], [[iSarcasmEval]]); a row-count mismatch against the dataset paper was noted, not ignored.
- [[Experiment Plan]] rewritten with arm table, "what each contrast tells us", and **predictions written down before running** (H1-H4), so a null result could be reported honestly.
- Venue rules collected second-hand and marked UNVERIFIED ([[Audit - Venues]]); a 10 vs 20 Dec deadline conflict was found and a conservative planning date chosen.

### Stage 2: False start and the working-mode rule (2026-11-09, day 2)
- A full code skeleton (models, training loop for all arms, teacher script, 15 tests) was written in one go. Review found two real bugs (FGM used a different batch for the adversarial pass; self-training later rounds got more steps than other arms).
- The owner then rolled the whole skeleton back: code written ahead of the owner's understanding could not be defended. New standing rule ([[Repo Rules]] rule 5): **the owner gets, cleans, uploads data and runs everything; sessions guide, explain, and write code only on request, with explanations.**
- Supervisor confirmed the seed paper's original code and settings were unavailable. Turned into a named limitation with draft paper wording on the same day ([[Risks]] item 8).
- A pause on RQ2 turned out to rest on a misreading ("the instructor's paper" was the seed paper, already in hand). Lesson: resolve ambiguous references at once instead of parking work on them.
- Repo made public so Kaggle notebooks can `git clone` the code; data stays private as a Kaggle dataset.
- Preprocessing notebook written cell by cell with a "what and why" explanation per cell. Contested choice (keep or remove punctuation) settled by a short literature check ([[Punctuation Evidence]]) and the owner's call.
- Kaggle's "Save & Commit" run stalled on Kaggle's side. Lesson: platform failures happen; have the interactive-session fallback ready.

### Stage 3: First real runs (2026-11-10, day 3 morning)
- Preprocessing re-run interactively. Interactive output is deleted when the session ends, so the owner downloaded the files, checked them against the manifest (SHA-256 per file), and uploaded them as the private dataset `gainsplit-processed`.
- A1 (plain BERT) written. Design questions explained in plain terms; the owner confirmed: equal optimizer steps for every arm (removes "trained longer" as a confound), helper weight tuned once on dev, shared helper heads.
- `lr_tuning` (9 runs, dev only, 3 seeds, about 1.3 min each): lr 5e-5 chosen and frozen. The pre-set tie rule turned out ambiguous; the reasoning was written down. Lesson: phrase tie rules exactly ("within 1 sd of the best mean, measured on the best setting's sd") before running.
- Measured cost replaced guesses: one run is minutes, not hours, so 5 seeds became affordable and the "checkpoint every epoch" plan became "skip finished runs".

### Stage 4: Delegation and the main passes (2026-11-10, day 3)
- The owner delegated every decision a session can reasonably take ([[Repo Rules]] rule 7). Each is logged in [[Decisions]] as **DELEGATED** with its reasoning so the owner can defend it later. Code is written in separate sessions (cheaper model, own branch) and reviewed by the coordinating session before merge.
- `session2` (12 dev-only runs): helper weight w = 0.3 (best on every seed) and a GAN pilot that checks the GAN actually trains (discriminator separates real and fake without collapse) before it enters the study.
- `pass1` (45 runs, 5 seeds, test scored, 114 GPU-min): GAN no gain; helpers +0.4 points at 10%. Accepted as run; no re-tuning after seeing test scores.
- Analysis notebook: paired bootstrap over test items (10,000 resamples, same resample across seeds). The 10% helper gain came out at p = 0.051 and was reported as **not significant**: no extra seeds, no switching tests. Lesson: decide the test before the numbers exist and live with borderline results.
- Pass 2 split into `pass2a` (104 runs, including FGM epsilon tuning on dev) and `pass2b` (10 FGM runs after epsilon was chosen). Arms chosen by what Pass 1 showed: the controls that explain the only gain went first (helpers on labeled data only; shuffled helper labels).
- Gold audit: the owner hand-labeled 200 headlines in an offline HTML app (seeded sample, blind to sarcasm and teacher labels, written guideline fixed first). Teacher kappa 0.24; a warm-up effect in the first 50 items was reported as a limitation, not hidden.
- Bootstrap over all 150 runs: FGM (a supervised adversarial control) +1.6 points at 1% and 10%, robust; GAN never helps. A Holm robustness column was **added beside** the pre-set analysis, not instead of it.
- Pass 3 designed from the results (label-quality rows, a new shuffled control at 1%, outlet-cue masking with a random-mask control, an in-domain reference model). Three code sessions wrote it in parallel on separate branches; reviewed, two small fixes, merged; 77 tests.
- A GAN fairness check (12 dev-only runs) answered "was the GAN under-tuned next to the tuned FGM arm?", with the pass/fail rule fixed before running.

### Stage 5: Results complete (2026-11-11, day 4 morning)
- `pass3` (105 runs) and `gan_check` (12) filed. `pass3` was accidentally run twice; both copies matched in all 1,575 metric cells, which doubled as a determinism check. Masked-test re-runs reproduced Pass 1-2 exactly.
- Analysis re-run on all 255 runs. Readings: most of the helper gain survives shuffled labels (a regulariser, not label content); gold agreement does not predict which label source helps; every model leans on outlet words; the GAN check failed its pre-set rule, so no extra test runs.
- Found a data-handling flaw: saved probabilities were rounded to 6 decimals, which creates AUC ties. Fixed by choosing one AUC source for the paper and stating it ([[Risks]] item 14).

### Stage 6: Paper draft (2026-11-11, day 4)
- Venue rules checked on the live site (deadline, 4-6 pages, double-blind, similarity limits): moved from UNVERIFIED to VERIFIED only after reading the source.
- Draft in IEEEtran, anonymous. Figures and tables are generated by scripts in `paper/figures/` and `paper/tables/` straight from `results/analysis/*.csv`; every number in the text re-checked against the CSVs. Effects shown as paired differences with intervals (not raw curves), filled vs hollow markers for Holm-robust results, colour-blind-safe palette plus marker shapes.
- Overleaf zip built. Open at the end: the seed paper's full bibliographic entry was never recorded when it was read, so the draft still says "CHECK" there ([[Paper Draft]]).

## Mistakes and what fixed them
| What went wrong | Cost | Fix or rule now |
|---|---|---|
| Whole code skeleton written before the owner understood it | One day of code thrown away | Owner runs everything; code on request only, with explanations |
| RQ2 paused on a misread reference | Half a day of wrong status | Ask what an ambiguous reference means immediately |
| Kaggle commit run stalled | One evening | Interactive session; download outputs before it closes |
| Analysis notebook run on Kaggle with no results folders | One failed run (`KeyError`) | Run analysis locally on CPU; notebook fails loudly when no runs load |
| Analysis re-run on one pass only | Committed tables overwritten | Always re-run on every results folder; re-commit after each pass |
| A run count quoted from memory (184 vs 150) | A wrong number in the log | Count from files, never from memory |
| Same plan run twice | GPU time | Harmless only because runs were deterministic; keep them deterministic |
| Ambiguous tuning tie rule | Judgement call needed | Write tie rules precisely before the runs |
| Probabilities saved to 6 decimals | AUC ties, two AUC sources | Save full-precision scores |
| Seed-paper citation not captured at reading time | Draft cannot go out as is | Fill the bib entry the moment a paper is read |
| Gold-audit export contains dataset text, repo is public | Licence risk | Such files git-ignored and backed up privately |

## Numbers worth remembering for planning
- BERT-base, max_len 48, fp16, T4: about 1.3 min per 720-step run; two runs at a time on Kaggle T4 x2.
- 255 runs in total over 7 plans; the largest Kaggle session (`pass2a`, 104 runs) took about 1 h 40 min.
- Calendar: 4 days from empty repo to full draft, with the owner running every notebook.

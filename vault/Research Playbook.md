---
tags: [playbook, process, meta]
created: 2026-11-11
updated: 2026-11-11
---

# Research Playbook

Hub: [[00 - Project Context]]. Where these habits came from: [[Research Diary]]. Template repo: [[Repo List]].

The working process that took GainSplit from an empty repo to a full paper draft in four days. Written to be handed to a new session on a **different** paper: copy the process, not the project content. Section 9 is a ready-to-paste starting brief.

## 1. Roles
- **Owner:** gets, cleans and uploads data; runs every notebook and script (locally or on Kaggle); makes the calls only they can make (scope, venue, data choices with paper-level consequences); must be able to defend every part of the paper.
- **Coordinating session:** reads the vault, plans, explains each step in plain terms (what, how, why it matters for the paper), takes delegated decisions and logs them, reviews code, files results, writes the vault and the paper.
- **Code sessions:** write a well-specified piece of code on their own branch (a cheaper model is fine). The coordinating session reviews and merges; nothing unreviewed reaches `main`.
- Never: download, clean, transform or upload data, or run experiments on the owner's behalf. Never mention the assistant anywhere in the repo or add co-author trailers ([[Repo Rules]]).

## 2. Day-one setup (about an hour)
1. Repo layout: `vault/`, `src/<package>/`, `experiments/` (notebooks), `tests/`, `results/<plan>/`, `data/` (git-ignored), `paper/` (with `figures/` and `tables/` generator scripts), `docs/`.
2. Root instructions file (no assistant name in its contents) pointing to the vault, plus [[Repo Rules]] and [[Vault Conventions]] copied from this repo.
3. Vault notes: hub (`00 - Project Context`, with a "Current state" block), `Session Log` (newest first), `Decisions` (append-only: date, choice, why), `Open Questions`, `Risks`, `Experiment Plan`, `Literature Index` + `Literature/`, `Datasets Index` + `Datasets/`, `Audit - Venues`, `00 - Backlog`, `Templates/`.
4. `.gitignore`: data, model weights, LaTeX build files, any export that contains dataset text.
5. Record the venue's deadline, page limit, template and anonymity rule. Mark them UNVERIFIED until read on the live call-for-papers page, and check that page in week one, not at the end.

## 3. Vault habits (the memory between sessions)
- Start of every session: hub, then the newest [[Session Log]] entry, then the linked notes the task needs.
- One idea per note; every note links to the hub and one related note.
- Evidence labels on every claim: **VERIFIED** (source read, link given), **INFERRED**, **IDEA**, **UNVERIFIED**. Never upgrade a label without reading the source. Read the closest prior papers **in full**.
- Every choice goes into `Decisions` with date and reason. Delegated choices are tagged **DELEGATED**; owner choices **CONFIRMED by owner**; replaced choices say what superseded them.
- End of every session: new Session Log entry (Done / Vault changes / Findings / Next) and a refreshed "Current state" in the hub.
- When a paper is read, record its full citation at once (authors, title, venue, year, pages). A missing reference at draft time blocks submission.

## 4. Shaping the question
- Prefer a question that is publishable whatever the result. "Where does the gain of published method X come from?" with controls beats a new method with a 20% chance of being judged novel.
- Build the design around **contrasts**: list the arms, then for each pair write what their difference tells you. Add negative controls (shuffled labels, random masks) and a simpler alternative that could explain the gain (supervised adversarial training, self-training, a TF-IDF floor).
- Hold everything else equal: same splits, backbone, steps, seeds and tuning rule for every arm. Equal optimizer steps stop "trained longer" from posing as an effect.
- Write the predictions (H1, H2, ...) into the Experiment Plan **before** any run.
- If the original code of the paper you build on is unavailable, say so early, re-implement under one protocol, and claim only differences between arms in your own setup. Draft the limitation wording the same day.

## 5. Experiment protocol (the rules that keep results defensible)
- Fixed train/dev/test split made once; the test set is not scored until the settings are frozen.
- Tuning: one hyperparameter at a time, a small grid (3 values), 3 seeds, **dev only**, on one reference budget, then frozen for all arms. The selection rule, including an exact tie rule, is written down before the runs.
- Pilot any fragile component (a GAN) on dev and check its training curves before it enters the study.
- Main runs: 5 seeds, test scored, accepted as run. No re-tuning after seeing test scores, no extra seeds to push a borderline p-value across 0.05.
- Statistics fixed before the results: paired bootstrap over test items (same resample for every seed and arm), 95% interval and two-sided p for every contrast; a multiple-testing (Holm) column added beside it as a robustness check, never replacing it.
- Work in **passes**: Pass 1 answers the main question cheaply; each later pass is designed from what the earlier one showed (controls that explain the observed gain first). A fairness check for an arm that may look under-tuned, with its pass/fail rule fixed in advance.
- Deterministic runs: the same config and commit must reproduce the same numbers. Re-runs are then a free correctness check.
- Save full-precision scores and per-item predictions (ids, not text); model weights are optional.

## 6. Code and compute pattern (free Kaggle GPU)
- Code in a package in a **public** repo; data in a **private** Kaggle dataset with a `manifest.json` of SHA-256 hashes made by the preprocessing notebook.
- One training notebook for every run. Its cells: (1) `git clone` the repo and print the commit; (2) run the test suite as a smoke test (tiny random model, made-up data); (3) list the GPUs, find the dataset, verify every file against the manifest; (4) pick a named **plan**; (5) run it; (6) zip the outputs.
- A **plan** is a named Python function returning a list of run configs (arm, budget, seed, settings, dev-only or test-scored, tag). Frozen settings live as constants next to the plans, with a comment saying which tuning run chose them. A plan that needs a not-yet-chosen setting raises an error instead of running with a default.
- The runner trains two runs at a time (one per GPU on T4 x2), skips runs whose results already exist (cheap resume), and writes per run: `result.json` (config, commit, metrics, training curves), prediction CSVs, a log.
- Every notebook cell has a markdown cell above it saying what it does and why, so the owner can run and explain it.
- Analysis runs locally on CPU over **all** results folders at once; its tables and figures and the executed notebook are committed after each pass.
- Kaggle notes: "Save & Run All (commit)" can stall on Kaggle's side; an interactive session works but deletes its output on close, so download first. Check the weekly GPU quota on the account page.

## 7. Delegated code sessions
1. Coordinating session writes a precise spec: files to touch, function names, exact behaviour, tests to add, what not to change.
2. Each code session works on its own branch; parallel sessions get disjoint files.
3. Review the diff against the spec and the protocol rules (same steps, same batches, gradients flowing only where intended, dev/test separation). Real bugs found this way: an adversarial pass on the wrong batch, uneven step counts, a summed loss that should be averaged, gradients leaking between GAN optimizers, a CSV reader turning the word "null" into an empty string.
4. Trial merge, run the tests, merge, log the review findings in `Decisions`.

## 8. Filing results and writing the paper
- Owner downloads the output zip; it is filed unpacked in `results/<plan>/` with run counts and paths checked against the plan. Then a results note in the vault: numbers, readings labelled INFERRED until the bootstrap confirms them.
- Every count and number is read from the files, never quoted from memory.
- Paper figures and tables are produced by scripts from the analysis CSVs; re-check every number in the text against them before the draft goes out.
- Show small effects as paired differences with intervals, not raw score curves; mark which results survive the robustness correction; colour-blind-safe colours plus marker shapes.
- Before sending: venue template, page count including references, anonymity (no repo link if double-blind), bibliography complete, similarity check, supervisor or instructor review.

## 9. Starting brief for a new paper (paste into the first session)
> New research project. Before anything else, read the setup of the template repo https://github.com/Ariffurrhmn/GainSplit in this order: `CLAUDE.md`, `vault/Repo Rules.md`, `vault/Vault Conventions.md`, `vault/Research Playbook.md`, `vault/Research Diary.md`, `vault/Templates/`, `.gitignore`. Copy its structure, rules and process into this repo; do not copy GainSplit's project content, and do not edit the template repo.
>
> Working rules: I (the owner) get, clean and upload all data and run all code and experiments myself, locally or on Kaggle. You guide me step by step, explain what each step does and why it matters for the paper, and write code when I ask, with a plain explanation. Take every decision you reasonably can and log it in `vault/Decisions.md` as DELEGATED with the reasoning; ask me only for calls only I can make. No assistant attribution or mention anywhere in the repo.
>
> Day-one tasks: (1) set up the repo layout and vault; (2) record the paper's question, target venue and deadline (mark venue facts UNVERIFIED until read on the live site); (3) list the closest prior papers and which ones I must supply as PDFs; (4) draft the Experiment Plan with arms, contrasts, controls and written-down predictions; (5) end with a Session Log entry and the hub's "Current state".
>
> Project details: [question / seed paper / datasets / venue / deadline / compute].

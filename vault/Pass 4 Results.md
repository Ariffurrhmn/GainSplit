---
tags: [results]
created: 2026-11-13
updated: 2026-11-13
---

# Pass 4 Results

Hub: [[00 - Project Context]]. Plan and predictions: [[Experiment Plan]] ("Pass 4 plan"). Pair data: [[Pairgen Full Run Results]]. Earlier results: [[Pass 3 Results]], [[Bootstrap - All Passes]]. Decisions: [[Decisions]].

Run by the owner on Kaggle: notebook 08 (GPU T4 x2, code commit 51fea9b, 193 tests passed first, data matched `manifest.json`), 75/75 runs in 76.2 min, re-run check PASS. Analysis: notebook 09 (Kaggle CPU) on 330 runs (pass1, pass2a, pass2b, pass3, pass4), pair set unfiltered SHA-256 6412c1fa... Raw files: `results/pass4/` (zip, unpacked runs and logs, `notebook_08.log`) and `results/analysis_pass4/` (zip, 13 tables, `notebook_09.log`); every unpacked file matches its zip.

Status: numbers VERIFIED (read from `analysis_pass4.zip` tables and run `result.json` files on 2026-11-13). Readings marked INFERRED are interpretation. Units: points of macro-F1 or AUC x 100. Paired bootstrap, 10,000 resamples; Holm over the Pass 4 contrasts.

## Checks (VERIFIED)
- Re-runs exact: A1 1% and A7 1% (mask), A1 10% (explain) and the Pass 3 mask re-runs all give F1 difference 0 and identical predictions (`rerun_check`, 40 rows, all True).
- Every pair run loaded the expected set (unfiltered: 44,834 rows, 22,417 seeds, Gemma 7,533 / Mistral 7,462 / Qwen 7,422).
- Pair runs fit their training mix easily (A10 10% seed 0: sarcasm loss 0.51 at step 100, 0.11 at step 720).

## Main result: pairs hurt in-domain, help transfer
| Arm | 1% test F1 | 10% test F1 | iSarc AUC 1% | iSarc AUC 10% |
|---|---|---|---|---|
| A1 (labels only) | 80.42 | 88.21 | 50.2 | 51.0 |
| A7 (FGM) | 82.00 | 89.76 | 47.8 | 51.5 |
| A6 (self-training) | 82.04 | 88.81 | 49.4 | 50.7 |
| A10 (A1 + pairs) | 63.95 | 73.38 | 71.7 | 72.4 |
| A11 (FGM + pairs) | 64.40 | 74.43 | 73.1 | 73.7 |
| A12 (unpaired control) | 63.73 | 73.11 | 72.2 | 72.6 |
| A13 (judge-filtered) | 64.82 | 74.36 | 72.5 | 73.5 |
| A14 (matched random subset) | 64.03 | 73.19 | 72.3 | 72.4 |
| A15 (pairs only, 0 gold labels) | n/a | 55.47 | n/a | 76.2 |
| A10, 3x steps | 60.23 | n/a | 73.4 | n/a |
In-domain iSarcasmEval reference (A1 trained on iSarcasmEval train, Pass 3): AUC 65.75 [61.6, 69.8].

Contrasts (`contrasts.md`, `auc_contrasts.md`):
- A10 - A1: **-16.5** at 1% [-18.1, -14.8], **-14.8** at 10% [-16.2, -13.4]; both p < 0.0001 (Holm 0.000).
- A11 - A7: -17.6 at 1%, -15.3 at 10%. A11 - A6 at 1%: -17.6. All p < 0.0001.
- Pairing, A10 - A12: +0.22 at 1% (p 0.41), +0.28 at 10% (p 0.30). No effect.
- Judge filter, A13 - A14: **+0.79** at 1% [0.24, 1.32] (Holm 0.023), **+1.17** at 10% [0.63, 1.71] (Holm < 0.001).
- Subset and model mix, A14 - A10: +0.08 / -0.19, both n.s.
- 3x steps at 1%: -3.7 F1 (p < 0.0001), +1.6 iSarc AUC.
- iSarcasmEval AUC: A10 - A1 +21.5 at both budgets; A11 - A7 +25.4 / +22.3; A15 - A1 (10%) +25.3; all CIs exclude 0. Every pair arm's AUC CI excludes 50 and lies above the in-domain reference 65.75.
- iSarcasmEval macro-F1 stays low for pair arms (A10 10% seed 0: 44.2; A15 seed 0: 34.9), so the ranking transfers but the 0.5 threshold does not (INFERRED: class balance differs, iSarcasmEval test is mostly non-sarcastic).

## Predictions written before the run
| | Prediction | Outcome |
|---|---|---|
| H4.1 | A10 > A1, largest at 1% | **Rejected**: -16.5 / -14.8 |
| H4.2 | A10 smaller cue-mask gap than A1 | Not supported: -0.56 (p 0.25) / -0.31 (p 0.50). Only A13 at 1% (-1.24, p 0.008) and A11 vs A7 at 1% (-1.05, p 0.027) reach p < 0.05, uncorrected |
| H4.3 | A10 iSarc AUC above A1, below 65.8 | Half: above A1 (+21.5), but **above** the in-domain reference too |
| H4.4 | A10 > A12 (pairing matters) | **Rejected**: pairing adds nothing; claim becomes "generated text", not "counterfactual pairs" |
| H4.5 | A13 ≈ A14 (judge noisy) | **Rejected**: filter helps +0.8 / +1.2 |
| H4.6 | A11 ≥ A7 | **Rejected**: -17.6 / -15.3 |

## Explainability (`explain.md`, `explain_contrasts.md`)
- Cue share (share of absolute Integrated Gradients attribution on the top-100 outlet cue words, 500 fixed test headlines, 5 seeds): A1 0.115, A10 0.095; difference **-0.021** [-0.028, -0.014], p < 0.0001.
- Caveat (VERIFIED): median relative completeness gap is 0.54 (A1) and 0.37 (A10). The attributions sum to roughly half to two thirds of the score difference they should equal, so 32 steps are too few; numbers are approximate. The comparison uses the same method on both models, so the direction is usable with this caveat stated (INFERRED).
- Caveat (INFERRED): A10 is a much weaker Headlines classifier, so a lower cue share does not mean a better model.

## Pair checks on CPU (`pairs_*.md`)
- Style probe (TF-IDF, generated vs real): AUC 0.859 (Gemma 0.888, Mistral 0.854, Qwen 0.866). LLM style is easy to see.
- Real-trained reader (TF-IDF on real train, AUC 0.931 on real dev) on the generated sides: AUC **0.526**; sarcastic side scored higher in 54% of pairs. What the generators call sarcastic is almost invisible to a Headlines-trained reader.
- Cue words only: real train AUC 0.851; generated pairs 0.626.
- Length only: real train AUC 0.530; generated pairs **0.785** (sarcastic sides 11.8 words, serious 9.2; Qwen 0.823). Length is a strong shortcut inside the pairs.

## Reading (INFERRED)
- The generated pairs carry a different notion of sarcasm from the Headlines label (outlet: The Onion vs HuffPost). Mixing them 1:1 into the batch pulls the model away from the Headlines label; more steps make it worse.
- The same pairs move the model toward the iSarcasmEval notion (intended sarcasm in tweets): zero-label A15 reaches AUC 76.2, above a model trained on iSarcasmEval's own train set (65.75). This is the strongest Pass 4 signal and the most surprising one; it needs a shortcut check before any claim: does length (or another surface feature) alone separate iSarcasmEval test labels? If a length-only score reaches about 0.7 AUC there, the transfer may be a length artifact.
- The judge filter is the only data-side knob that helps, consistent with the filter removing pairs whose serious side reads as sarcastic ([[Pairgen Full Run Results]]).
- As a method for in-domain gain, pairs as plain extra labeled rows fail. Options for a next step are in [[Research Ideas]] "After Pass 4": lower pair weight, pair-ranking loss instead of shared labels, pre-train on pairs then fine-tune on L, length-matched pairs.

## iSarcasmEval length-shortcut check (written 2026-11-13, not run)
Code: `pair_checks.isarc_length_shortcut` (descriptive: word count by class and AUC on the real iSarcasmEval test labels, printed in notebook 09 section A) and `analysis_pass4.isarc_length_auc` (bootstrap 95% CI of the same AUC over the exact items a run was scored on, wired into `run_all` as the `isarc_length_shortcut` table, section E). 29 tests cover both (`tests/test_pass4.py`), full suite 137 passed / 2 skipped locally.
- Reads whether word count alone already separates the real iSarcasmEval test labels (not the generated pairs, which is `pairs_length`, already run). If this AUC is itself high (close to the 72-76 the pair arms reach), the transfer gain in this note may be partly a length shortcut rather than sarcasm, and the claim needs to say so.
- Not yet run: needs the owner to re-run notebook 09 (CPU only, same inputs as before) once code is pushed.

## Open
- iSarcasmEval length-shortcut check above: written, not run.
- Whether to run `pass4_full100` (A10 at 100%): given the in-domain result, INFERRED low value.

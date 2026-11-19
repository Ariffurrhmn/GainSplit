# PLAN: next phase after the Draft v2 reviews

Written 2026-11-19. Planning only: no run was launched, no paper file was changed. Predictions skeleton: `vault/Pass 11 Predictions.md` (DRAFT until frozen, see D.4).
Evidence labels as in `vault/Vault Conventions.md`. "Verified" below means checked against code, configs, result files or the cited source in this session.

Inputs read: `reviews/google.txt` (identical in content to `paper/v2/review.md`), `reviews/Act_as_a_rigorous_peer_reviewer_doing_a_LITERATURE.pdf` (Consensus, 14 pp.), `paper/v2/main.tex`, `refs.bib`, `src/gainsplit/*` (config, plans, train, ssl_pretrain, data, masking, models, runner), notebooks 15 and 20, `results/*/summary.csv`, `gan_check`, pretraining receipts, vault notes (Project Context, Session Log head, Open Questions, Experiment Plan, Pass 2a/3/10 results, Bootstrap notes, Zihan, Seed paper). No separate merged fix list exists in `reviews/`. No requirements file is tracked; the environment is the Kaggle image (torch/transformers versions are recorded in every `result.json`).

Hardware (repo, `vault/Decisions.md` 2026-11-10): Kaggle GPU T4 x2, two runs at a time via `runner.py`. Local machine has no GPU and no torch. "GPU-h" below = summed run minutes; "wall" = elapsed time on T4 x2.

---

## 0. What the verification changed (read this first)

1. **Candidate 4 (helper heads on L + U) is already run.** Arm A3 = helper heads on L + U, at 1/5/10/30% (Pass 1, 2a). 1%: A3 80.29, A3 - A1 = -0.13, A3 - A2 = -1.76 (robust, Holm 0.004). The paper reports only A2 (L only). The headline "helper heads +1.6 at 1%" is therefore for a design the seed paper did not use. Text edit, no run.
2. **Candidate 3 (LLM-pair mixing ratio) is mostly answered.** Per step, A10 draws 32 L rows + 32 pair rows (`pair_batch=32`), i.e. 1:1, not Google's "200:1". Pass 5 already swept `pair_weight` (0.1/0.3/1.0, dev), and weight 0.1 recovers only about 30% of the loss; cleaning the pairs (A16) added nothing (-0.2), A12 (unpaired) and A13/A14 (filter vs random) exist. The paper's "likely reason: length and stock phrasing" is weakened by the cleaning result. Text edit, no run.
3. **Candidate 7 (do frozen MLM features cluster by outlet) is a no-op as a run.** Label = outlet, so clustering by outlet is label separability. The existing receipt already has the frozen linear probe at 1% labels: dev macro-F1 76.2 (BERT) -> 86.4 (MLM) -> 78.5 (JEPA). Report these numbers; drop the run.
4. **"Baseline undertrained at 1%" is refuted** (my own contrarian check): `gan_check` A1 at 1%, 3x steps, dev: 78.85 vs 79.52 at 1x (seeds 0-2). More steps do not help plain BERT, so the +7.4 is not "TAPT rescues a starved 200-step run".
5. **Exposure gap arithmetic is correct and wider than stated.** A4, A5, A24, A25 read one U batch of 32 per step: 200 steps x 32 = 6,400 of 22,572 = 28.4% of one pass at 1%. The A6 student is capped the same way (step count follows |L|, so at 1% it reads at most 6,400 of about 18k pseudo-labeled rows). The dev "3x steps" check reaches only 85%. MLM/JEPA read about 28 passes. Not only GAN and diffusion are affected.
6. **A paper error found:** Section IV-E says "the 200 words most indicative of each outlet". `cue_words.json` holds 100 per class (200 in total, `k_per_class = 100`). Fix the sentence.
7. **Holm wording is inaccurate for early passes.** Passes 1-3: Holm was computed post hoc over 38 contrasts as a robustness check (`vault/Bootstrap - All Passes.md`); the pre-set test was uncorrected. Passes 8-10 have per-table families (Pass 10: stack_f1 16, stack_interactions 14, seeds_f1 6, curve_f1 7, gap_contrasts 14, auc_vs_length 10). The paper says "inside each pre-planned family". Must be rewritten per pass, not just "add m".
8. **Seed-paper 70%**: `vault/Deep Dive - Paper A.md` marks "70% of train labeled" as read from the PDF, so the `% CHECK 70%` comment in `main.tex` can be resolved. Open Question 6 (their Table II supports sum to the whole dataset) argues for keeping "not comparable".

---

## A. Triage table

Source: G = Google, C = Consensus. Status: CONFIRMED / REFUTED / UNVERIFIABLE (needs owner) / SPECULATION. Type: T = text edit, E = cheap experiment, O = optional. Effort in minutes (text) or hours.

| # | Item | Src | Status (evidence) | Sev | Type | Effort | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Title and abstract broader than the evidence (one outlet-confounded testbed, one encoder) | G, C | CONFIRMED (title has no "headline"; abstract "simplest route wins" has no scope). Conflict: C wants the "Just Keep Pretraining" title dropped; G wants only "Headline" added. Resolution: keep the title's punch, add scope (decision Q1) | High | T | 20 | Do today |
| 2 | "Absorbs the gains" overstated | C, G | CONFIRMED as scope: pool U = pretraining text (closed pool). Li et al. 2021 find TAPT and self-training complementary; Du et al. 2021 (from memory, UNVERIFIED) use an external retrieved pool. Soften to "on this dataset, encoder and pool" | High | T | 30 | Do today |
| 3 | Exposure gap (0.28 vs 28 passes), incl. A6 student | G | CONFIRMED (item 0.5). Needs reframing plus evidence | High | T + E2 (+ E3 optional) | 30 text; 0.7 GPU-h | Text today; E2 first in queue; E3 optional |
| 4 | A2 trained on L only; is A5 trained on U? | G | CONFIRMED: A5 applies helper loss to L+U (`helpers_on_pool` default, `gan_step`); A2 is `l_only`. A3 exists and is unreported (item 0.1) | High | T | 40 | Do today: add A3, A3-A2, state A5 on L+U, explain A2 |
| 5 | In-domain low-label run on an unconfounded set | C (must-have), G (implicit) | CONFIRMED gap. iSarcasmEval train = 3,467 tweets, 25% sarcastic; after the existing 10% dev split about 3,120. 1% = 31 tweets, too small | High | E1 | 2.1 GPU-h + coding | Do today (Tier A) |
| 6 | RoBERTa-base as second encoder | C | CONFIRMED (single encoder only). `model_name` already in `RunConfig`; masked-test sets contain a literal `[MASK]` that BPE splits, so skip masked evals for RoBERTa | Med-High | E4 | 1.35 GPU-h | Do today if time (Tier B) |
| 7 | DeBERTa / ModernBERT | C | CONFIRMED absent | Low | T | 5 | Skip: one limitation clause |
| 8 | Missing SSL baselines (MixText, FixMatch-NLP, PET, SetFit) | C | CONFIRMED absent. UDA/FixMatch need text augmentation (not built) | Med | T | 20 | Skip runs; Related Work sentence plus limitation. Risk acknowledged (C.2) |
| 9 | LLM zero/few-shot baseline | C | CONFIRMED absent; needs API access and cost; does not touch the claims | Low | T | 5 | Skip; cite SarcasmBench (verify) |
| 10 | DAPT on external news (CC-News) | C | CONFIRMED absent. Speculation: CC-News may contain HuffPost/Onion text overlapping the test set; download rule applies | Low | - | - | Skip: leakage risk plus owner downloads |
| 11 | Supervised contrastive learning arm | C | CONFIRMED absent | Low | - | - | Skip |
| 12 | "Outlet-balanced evaluation" | C | REFUTED as a no-op: test set is 47.5% sarcastic and label = outlet; balancing changes nothing | - | - | - | Skip |
| 13 | TAPT + self-training precedent (Li et al. 2021) and why self-training saturates | G | CONFIRMED precedent (EMNLP Findings 2021, pp. 1006-1015, arXiv 2109.06466). **Google's author list is wrong**: actual authors Li, Yavuz, Chen, Yan | Med | T | 20 | Do today; fix authors in bib |
| 14 | Diffusion consistency vs text-level UDA | G | CONFIRMED scope point | Low | T | 5 | Do today |
| 15 | MLM warm start (BERT's MLM head kept) vs cold JEPA predictor | G | CONFIRMED: `AutoModelForMaskedLM.from_pretrained` keeps the head; `build_predictor` is random init. Fine-tuning discards both heads, so only the pretraining phase is asymmetric | Med | T + O | 10 text; 1.8 GPU-h | Text today; JEPA seeds E5 optional |
| 16 | Single JEPA pretraining seed | G | CONFIRMED (already in Limitations) | Low-Med | T / E5 | | As item 15 |
| 17 | Holm families unstated | G | CONFIRMED and larger (item 0.7) | High | T | 40 | Do today; list per pass |
| 18 | Table II lacks s.d.; base checkpoint; A31 undefined | G | CONFIRMED (stack uses seed-0 encoder, `PASS8_CKPT["mlm"]`; A31 = A20 + helpers) | Low | T | 20 | Do today (data in `results/pass10_stack/summary.csv`) |
| 19 | Test-only bootstrap ignores training-split variance | G | CONFIRMED (limitation iii half-covers it) | Med | CPU analysis | 1 h | Do today: cluster bootstrap over seeds and items on existing predictions (E0b) |
| 20 | Only 1% and 10% reported | G | CONFIRMED. A1, A20 at all five budgets exist (Pass 10); A4/A5 at 5, 30% exist (Pass 2a); **A4/A5 at 100% do not exist** (Pass 1 ran 1/10% only), so G's 100% request is partly impossible | Med | T | 40 | Do today: a compact 5-budget table for A1, A3-A5, A20 |
| 21 | Log-linear interpolation not stated | G | CONFIRMED (`curve` table uses log-linear) | Low | T | 5 | Do today |
| 22 | Rounding differences between tables and text | G | CONFIRMED (e.g. 80.4 vs 64.0 = 16.4, text 16.5) | Low | T | 5 | Do today: caption note |
| 23 | Reproducibility details (LLM names and prompts, diffusion, JEPA predictor/EMA, helper teachers, audit sampling) | G | CONFIRMED omitted. Values exist: LLMs Qwen/Gemma/Mistral (`Pairgen Full Run Results`), JEPA predictor 2 layers, EMA 0.996 to 1, MSE (`ssl_pretrain.py`), audit seed 20261110, 200 sampled | Med | T | 45 | Do today, one sentence each; page budget decides |
| 24a | A10 "200:1 dominance" | G | REFUTED (1:1 per step; item 0.2) | - | T | 5 | State the mixing ratio in III-B |
| 24b | "Likely reason: length/phrasing" unsupported | C | CONFIRMED, stronger than C says: Pass 5 cleaning (drop 20% worst) added nothing | Med | T | 15 | Replace with the evidenced statement: generated text is detectably different (shortcut CV AUC 0.95; length alone AUC 0.785), but removing the worst seeds and down-weighting did not repair it; cite Li 2023 / Veselovsky 2023 (verify) |
| 25 | Cue-mask method details; "200 words per outlet" | G | CONFIRMED (item 0.6). Method: binary-count unigram logistic regression, balanced, top 100 per class, whitespace tokens, `[MASK]` replacement, random control masks the same count per headline. The random control uses the same `[MASK]` token, which answers G's MLM/[MASK] interaction worry | Med | T | 20 | Do today |
| 26 | Why ROC-AUC in transfer; A1 transfer score; protocol of the 65.8 reference | G | CONFIRMED omitted (test 14.3% sarcastic vs 47.5%). 65.8 = A1 trained on iSarcasmEval at 100%, 10% stratified dev split, `train_data="isarcasm"` | Low-Med | T | 15 | Do today |
| 27 | Transfer failure mixes genre and label-source mismatch | G | CONFIRMED | Low | T | 10 | Do today; E1 partly separates them |
| 28 | Polarity-flip placeholder; DiD design | G | CONFIRMED design is right; arithmetic 107+474+774 = 1,355 checks | Med | CPU analysis | 1.5 h | Do today (E0a) |
| 29 | Seed-paper 70% | G | CONFIRMED resolvable (item 0.8) | Low | T | 5 | Do today; move to Introduction |
| 30 | Name QPAIN 2025 origin of [3] | G | CONFIRMED harmless in third person | Low | T | 5 | Do today |
| 31 | Non-NLP framing: GAN/diffusion act on [CLS]; spell out FGM and iSarcasmEval; "predictions" wording | G | CONFIRMED | Low | T | 15 | Do today |
| 32 | Reference fixes: initials in [3] and [16], `Marasovi{'c}`, duplicate 2023 in [5], page numbers | G | CONFIRMED in `refs.bib` (`Marasovi{'c}` line 233; `irfan2025seed` = "Irfan and Das and others"). Author lists for [3] from reviews, to be checked on IEEE Xplore (UNVERIFIABLE here) | Low | T | 30 | Do today; owner confirms names |
| 33 | Arm codes non-sequential | G | CONFIRMED | Low | T | 5 | One sentence |
| 34 | Figure 2 axis "Labelled"; table header style | G | CONFIRMED (figure made by `paper/v2/figures/make_figures.py`) | Low | T | 10 | Do today; regenerate figure (owner runs) |
| 35 | Single-annotator audit; sampling | G | CONFIRMED omitted | Low | T | 5 | Do today |
| 36 | GAN null may be dataset-specific | C | CONFIRMED (add one sentence) | Low | T | 5 | Do today |
| 37 | Add McHardy 2019, Oprea and Magdy 2020, Li et al. 2023, Veselovsky 2023, data2vec, Ruder 2017, Bingel and Sogaard 2017 | C | PARTLY: relevant and plausible; every DOI/venue unverified (see B.2). Page limit 6 | Med | T | 60 | Do later, after results; only those that carry a claim |
| 38 | Shi and Lipani 2023 (TAPT can hurt) | C | CONFIRMED but loose: abstract says harm "for sentence-pair tasks or when prompt-based FT is used". Our setting is single-sentence standard fine-tuning | Low | T | 5 | Optional caveat clause only |
| 39 | ULMFiT "100 labels match 10x more data" | C | **REFUTED: abstract says 100x** (checked on arXiv 1801.06146) | Low | T | 2 | Do not use C's figure; cite 100x if ULMFiT is added |
| 40 | Consensus reference-accuracy lines, author names, "Rahman" attributions | C | REFUTED / garbled: parentheticals pair wrong references (e.g. Sun 2019 labelled as FGM source); "(Rahman)" is a de-anonymising guess, do not copy into the paper | - | - | - | Ignore |
| 41 | Cut diffusion/LLM paragraphs to make room | C | CONFIRMED useful: 6-page limit, new content needed | Med | T | 30 | Do today (compress, keep numbers) |
| 42 | Shrink the GAN dev-check paragraph | C | CONFLICT with item 3 (G asks for more on GAN exposure). Resolution: one sentence, with the exposure numbers | Low | T | 5 | Do today |
| 43 | "FGM matched GAN-BERT for Bengali" | C | CONFIRMED overreach: our note on the source (Zihan 2026) says GAN-BERT numbers were copied from Lora 2023 and the text's "closely matched" is not backed by the table (68.08 vs 72.9). C's own reason ("paper not in corpus") is weak, the vault evidence is better | Low | T | 5 | Soften |
| 44 | "Predictions fixed before every experiment" | own | CONFIRMED with nuance: Passes 4-10 have per-pass blocks in `Experiment Plan.md`; Passes 1-3 have the general block (line 66) and Pass 3 items. Wording should say "written before each pass" | Med | T | 10 | Do today |
| 45 | Baseline undertrained | own | REFUTED (item 0.4) | - | T | 5 | Add one clause (3x steps do not help A1) |
| 46 | Weak-baseline risk (Oliver 2018): one-round, one-threshold self-training | own | SPECULATION. MLM teacher pseudo-labels are 92-96% correct, so the room is small | Low | - | - | Limitation clause |

---

## B. Experiments

### B.1 Selection

Ranked by "changes a claim per GPU-hour". Tier A runs today; Tier B if time; Tier C only after A and B. Dropped: candidates 3, 4, 7 (item 0), DAPT, contrastive, LLM zero-shot, extra SSL baselines.

| Tier | ID | Experiment | GPU-h | Wall (T4 x2) | Cum. wall |
|---|---|---|---|---|---|
| CPU | E0a, E0b | Polarity DiD; seed-aware bootstrap | 0 | 1.5-2.5 h of owner time, parallel | - |
| A | E2 | MLM dose curve at 1% | 0.7 | 0.4 h | 0.4 h |
| A | E1 | In-domain A1 vs A20 on iSarcasmEval, with size-matched control | 2.1 | 1.2 h | 1.6 h |
| B | E4 | RoBERTa-base A1 vs A20 | 1.35 | 1.0 h | 2.6 h |
| C | E3 | GAN with about 10 passes over U, paired with A1 at the same steps | 2.5 | 1.3 h | 3.9 h |
| C | E5 | Two more JEPA pretraining seeds | 1.8 | 1.0 h | 4.9 h |
| | | Total | 8.5 | 4.9 h (about 5.2 h with notebook overhead) | |

Tier A alone: 2.8 GPU-h, about 1.6 h wall.

Cost basis (measured, `results/*/summary.csv`, `pretrain_info.json`): A1 1%/10% 0.64/1.69 min; A4 1% 1.05 min at 200 steps (0.165 s/step); A20 1%/10% 0.53/1.43 min; A1 100% 7.8 min; MLM pretraining 10,000 steps 35.2 min, JEPA 43.7 min (T4, probes included). Estimates for new runs are INFERRED from these.

### B.2 Common setup for every new run

- Same recipe as the paper: bert-base-uncased, lr 5e-5, batch 32, max_len 48, AdamW, 10% warm-up, `max(200, 10 passes over L)` steps, fp16, no early stopping, seeds from `budgets.json` (0-4), `eval_test=True`. Settings stay frozen; nothing is tuned on test; E4 is the only run family with a dev-only selection (rule fixed below).
- Test scoring: macro-F1, accuracy, F1-sarcastic, ROC-AUC (as `evaluate.scores`), saved predictions per run (`preds_<split>.csv`) for paired bootstrap.
- Analysis: paired bootstrap over test items, 10,000 resamples, same resample for all seeds, statistic = mean over seeds; Holm within each named table; a prediction is "supported" only if the direction holds and p_holm < 0.05; "no difference" predictions are supported if p_holm >= 0.05 and the interval is reported (as in Pass 10). Add the seed-aware interval from E0b to the headline contrasts.

### E0a. Polarity-flip DiD (CPU, existing predictions)

- Question: does a sentiment helper pull the encoder the wrong way on sarcastic headlines with positive surface sentiment (107 of 1,355 sarcastic test headlines)?
- Setup: per-seed predictions of A1, A2 (Pass 1/2a) and A20, A31 (Pass 8a/10) on the test set; groups from `helper_labels.csv` sentiment (positive / negative / neutral) restricted to sarcastic headlines. Statistic: difference-in-differences of error rate: [err(A2, pos) - err(A1, pos)] - [err(A2, rest) - err(A1, rest)], and the same for A31 vs A20. Bootstrap over test items within each group (group sizes fixed), 10,000 resamples. Exploratory; n = 107.
- Reuses: `analysis.py` bootstrap helpers; needs a new function (about 60 lines) and a test. Output: `results/analysis_pass11/polarity_did.csv`.
- Cost: CPU, minutes. No dependency.

### E0b. Seed-aware uncertainty (CPU, existing predictions)

- Question: do the headline contrasts survive when the five training splits are resampled too, not only the test items?
- Setup: two-level (cluster) bootstrap: resample seeds with replacement (5 of 5), then test items, 10,000 resamples. Contrasts: A20-A1, A2-A1, A6-A1, A7-A1, A22-A20, A30-A1 at 1% and 10%; A20(1%) vs A1(10%). Also report the per-seed difference vector and a sign test.
- Output: `results/analysis_pass11/cluster_bootstrap.csv`. Cost: CPU.

### E2. MLM dose curve (exposure)

- Question and claim tested: is the MLM advantage over the GAN-type routes a matter of exposure to U (about 28 passes vs 0.28), or of the objective? Also tests how fast the +7.4 appears.
- Setup: pretrain MLM with seed 0 and every other setting as Pass 8a, for steps in {100, 350, 1,000, 3,000}. 100 steps x 64 = 6,400 rows = exactly the GAN's U exposure at 1%; 350 steps = one pass. Then A20 at 1% labels, seeds 0-4, from each encoder (20 runs, tags `dose100`, `dose350`, `dose1000`, `dose3000`). The 10,000-step row is Pass 8a (reused). Linear probe receipts on (default). Note: warm-up and decay scale with the step count (6% warm-up, linear decay), so short runs are complete schedules, not truncated ones.
- Reuses: `ssl_pretrain.py` via `--config '{"steps": N}'`, `plans._p10`, `analysis_pass10` bootstrap helpers. Must write: plan `pass11_dose` in `plans.py` (about 15 lines), `analysis_pass11.dose_table` (gain vs steps, with intervals), notebook 22 (copy of notebook 20 structure), a test.
- Files: `results/pass11_core/` (shared with E1, see D), receipts `pretrain_info_dose*.json`, tables `analysis/dose_f1.csv`.
- Cost: pretraining 4,450 steps (about 25 min with probes, two GPUs in parallel) + 20 runs x 0.55 min = 0.7 GPU-h.
- Dependency: none. First in the queue.

### E1. In-domain low-label comparison on iSarcasmEval (author-labeled)

- Question and claim: does the label-efficiency gain of continued MLM pretraining replicate when the label is not the outlet? Tests contribution (iii) and the "dataset-specific" reading. This is the run most likely to change the paper's claims.
- Data facts (verified): `isarcasm_train.csv` 3,467 rows, 25.0% sarcastic; `isarcasm_test.csv` 1,400 rows, 14.3% sarcastic (200 positives). Existing split code `load_isarcasm_splits` holds out a 10% stratified dev set (fixed seed) and uses the rest, about 3,120 rows, as L with no U.
- Label budgets (absolute counts, nested per seed, drawn from the 3,120 non-dev rows): N = 228 (same count as Headlines 1%), 780 (25%), 3,120 (all). 1% (31 tweets) is dropped as too small. **10 seeds** per cell (runs are small; test has only 200 positives, so average over seeds).
- Arms (all fine-tune with the A1 recipe; `max(200, 10 passes)` steps):
  1. A1: BERT.
  2. A20-iS: MLM-pretrained on the iSarcasmEval train texts minus dev (about 3,120 texts, no labels, no test). **Length rule, fixed now: 1,400 steps at batch 64 (about 28.7 passes), matching the "about 28 passes" of the main experiment.** Other settings as `PretrainConfig`.
  3. A20-iS100: same with 4,900 steps (about 100 passes, the TAPT default for small corpora). Sensitivity only, N = 228 and 780.
  4. A20-HS: the existing Headlines MLM encoder (seed 0, Pass 8a), fine-tuned on iSarcasmEval. Cross-domain control.
  5. **Size-matched control (on Headlines):** MLM on a random 3,120-headline subset of the Headlines training texts (fixed subset seed), 1,400 steps, then A20 at Headlines 1% and 10%, seeds 0-4 (10 runs). Reads: if a 3.1k-text pool still gives most of the Headlines gain, a small iSarcasm gain cannot be blamed on pool size.
- Metrics: **primary = ROC-AUC on iSarcasmEval test** (test is 14% sarcastic; same metric as the transfer section); secondary macro-F1 and F1-sarcastic (the SemEval metric). Dev only for sanity.
- Reuses: `train.py` unchanged apart from the data path; `ssl_pretrain.pretrain` (text source hard-coded to `headlines_train.csv`); `runner.py`. **Must write:** (a) `data.load_isarcasm_splits(data_dir, budget_n, seed)` using a new owner-generated `budgets_isarc.json` (nested subsets of the non-dev rows) and returning U = remaining non-dev texts without labels; (b) relax `RunConfig` guards (`train_data="isarcasm"` is currently A1 at budget 100 only; `init` arms require Headlines); (c) `ssl_pretrain --texts-source {headlines, isarc, headlines_subset}` reading only train minus dev, never test; (d) plan `pass11_isarc`; (e) `analysis_pass11.isarc_*` tables; (f) tests (no test ids in pretraining texts; no labels read; byte-identical Pass 1/8a re-runs); (g) notebook 22 cells. About half a day of code, reviewed before the run commit.
- Owner steps (data rule): generate `budgets_isarc.json` on Kaggle from `isarcasm_train.csv` and the fixed dev seed, add it to a new `gainsplit-processed` version with its manifest entry.
- Cost: pretraining about 32 GPU-min (1,400 + 4,900 + 1,400 steps with probes); runs: 3 main arms x 10 seeds x 3 budgets at about 0.6/0.6/1.4 min = 78 min, + A20-iS100 20 runs = 12 min, + 10 control runs = 10 min. About 2.1 GPU-h.
- Output: `results/pass11_core/`: `analysis/isarc_auc.csv`, `isarc_f1.csv`, `size_control.csv`, receipts `pretrain_info_isarc28.json`, `..._isarc100.json`, `..._hsub.json`.
- Dependency: code and `budgets_isarc.json` (about half a day). Do not block E2 on it.

### E4. RoBERTa-base, A1 vs A20

- Question: is the headline gain specific to bert-base-uncased? (RoBERTa's pretraining corpus includes news, so the gain may be smaller; that is the interesting case.)
- Setup: `model_name = "roberta-base"`. Stage 1 (dev only, tuning rule fixed now): A1 at 10%, lr in {2e-5, 3e-5, 5e-5}, seeds 0-2, `eval_test=False`; pick the best mean dev macro-F1, ties within one s.d. go to the smaller lr (same rule as every `_tuning` plan); the notebook freezes it in memory, no second commit. Stage 2: MLM pretraining of roberta-base, 10,000 steps, same `PretrainConfig` (seed 0); then A1 and A20 at 1% and 10%, seeds 0-4, `eval_test=True`, `eval_masked=False` (the masked test CSVs contain a literal `[MASK]`, which RoBERTa's tokenizer splits). Check max_len 48 covers about 99% of headlines under BPE (add the check to the notebook pre-flight).
- Reuses: `RunConfig.model_name`, `ssl_pretrain` (`--config '{"model_name": "roberta-base"}'`), `plans._p10`. Must write: plan `pass11_roberta` (tuning + main), guard that `cfg.model_name` equals the tokenizer base recorded in `pretrain_info.json` for init arms, a test, notebook 23.
- Cost: tuning 9 x 1.7 min + pretraining about 40 min + 20 runs about 25 min = 1.35 GPU-h; wall about 1.0 h (pretraining on one GPU while the other tunes).
- Output: `results/pass11_roberta/`.

### E3 (optional). GAN-BERT with many passes over U, paired with plain BERT

- Question: is the GAN null an artefact of 0.28 passes over U? Note `step_mult` also multiplies the passes over L, so a GAN-only long run is confounded; the control is A1 at the same step count and the contrast is A4 - A1 at matched steps.
- Setup: A4 and A1 at 1% labels with `step_mult = 35` (7,000 steps, about 9.9 passes over U), seeds 0-4, test scored (10 runs). Optionally A24 (diffusion fakes) as well.
- Reuses: `RunConfig.step_mult`, existing arms. Must write: plan `pass11_ganlong` (5 lines), analysis rows.
- Cost: A4 about 19 min x 5, A1 about 11 min x 5: 2.5 GPU-h. Promoted to must-run if E2 shows the 100-step MLM encoder gains at most 1 point (exposure would then be a plausible explanation); demoted to cut if E2 shows at least 4 points.
- Existing evidence that covers part of this: A4 at 30% (about 4 passes over U) is still -0.55 vs A1; A4 at 10% (1.1 passes) is null; dev 3x check at 1% (A4 77.3 vs A1 78.9).

### E5 (optional). Two more JEPA pretraining seeds

- Question: is "MLM beats JEPA by 1.75 points" robust to the JEPA pretraining seed (the claim currently rests on one JEPA encoder against three MLM encoders)? Partly confounded by the warm MLM head vs cold predictor; that cannot be removed without changing the objective, so the text must say so whatever the result.
- Setup: `ssl_pretrain --objective jepa --config '{"seed": 1}'` and `{"seed": 2}`, in parallel on two GPUs; A22 at 1% and 10%, seeds 0-4, from each (20 runs).
- Reuses: plan modelled on `pass10_seeds`. Cost: 2 x 43.7 min + 21 min of runs = 1.8 GPU-h.
- Low information: cut first among Tier C.

---

## C. Paper edits (no GPU, `paper/v2/` after the owner agrees; confirmed items only)

**Title / Abstract / Introduction**
- Title: add scope (Q1), e.g. "... for Low-Label Headline Sarcasm Detection", or keep the title and bind the scope in the first abstract sentence.
- Abstract: "on News Headlines with bert-base-uncased" next to "wins clearly"; replace "absorbs"/"nothing" wording with "on this dataset"; spell out FGM and say what iSarcasmEval is; replace "predictions fixed before each experiment" with "directional hypotheses written before each pass"; add the E1 result as one sentence.
- Introduction: seed model's 93.88% used 70% of labels (and QPAIN 2025 origin, third person); why Headlines (recent systems claimed gains on it; label = outlet; this is an audit of those claims, not a claim about sarcasm in general); that GAN and diffusion act on [CLS] vectors while MLM adapts all layers; contribution (ii) "absorbs" -> "absorbs, on this dataset and pool"; contribution (iii) gains the in-domain result from E1.

**Related Work**
- Add Li et al. 2021 (TAPT + self-training complementary) and say why it differs here (closed pool, two-outlet data, teacher accuracy 92-96%); UDA is text-level augmentation, ours is latent diffusion; one sentence on missing baselines (MixText, FixMatch-NLP, PET, SetFit) with the reason; Zihan sentence softened (item 43); add only the Consensus references that carry a claim (item 37).

**III-B / III-C**
- Define A31 and arm-code scheme; A3 (helpers on L + U) with its numbers; A5 on L + U; why A2 on L only (it is the variant that helps; A3 removes the gain at 1%, A3 - A2 -1.76).
- Exposure: state 200 steps x 32 U rows = 28.4% of a pass (and the A6 student cap) vs about 28 passes for MLM/JEPA; reframe as "a fixed-step single-stage route vs a two-stage route"; add E2 (and E3 if run); A1 at 3x steps does not improve (dev).
- A10: 32 L + 32 pair rows per step; the three LLMs; prompt in one clause; item 24b replacement sentence.
- Pretraining reads L + U texts (all 22,800 training headlines, no labels); MLM keeps BERT's MLM head, JEPA predictor is cold, JEPA one seed (or three if E5).
- JEPA, diffusion, helper-teacher details in one line each (values in item 23).
- Holm: per pass, family names and sizes (Passes 1-3: post hoc over 38 contrasts, uncorrected p primary; Pass 10: 16/14/6/7/14/10; Pass 8/9 sizes to be read from their `analysis` tables); bootstrap conditions on five training splits, plus E0b results; rounding note ("differences computed from unrounded means").

**IV**
- Table I: add A3; consider a compact 5-budget table (A1, A3-A5, A20) and say A4/A5 at 100% were not run; Table II: s.d., base checkpoint = seed-0 encoder; Holm-adjusted p for the surviving combinations (H+S, H+F+S).
- IV-B: log-linear interpolation, with the A1/A20 numbers at 5/30/100%.
- IV-D: finish the polarity paragraph from E0a (report either outcome); note frozen-probe numbers 76.2 -> 86.4 (MLM), 78.5 (JEPA) as the feature-quality evidence; single annotator, 200 headlines sampled with a fixed seed.
- IV-E: fix "200 words per outlet" to 100 per outlet; method sentence (item 25); why AUC (14% vs 47.5%); A1's transfer AUC next to the pretrained range; what 65.8 is; genre and label-source confounds; E1 result and size-matched control.
- Figure 2 axis spelling.

**V / VI**
- Limitations: one encoder family (or two with E4), one confounded main testbed plus one in-domain check (E1), single JEPA seed, no external-pool DAPT, no text-augmentation SSL baselines, test-set bootstrap (plus E0b).
- Conclusion: "simplest route is best" only with the scope words from the abstract.

**References**: `Marasovi{\'c}`; [3] and [16] initials and full author lists (owner confirms on the publisher pages); duplicate year in [5]; page numbers; add Li et al. 2021 with correct authors (Li, Yavuz, Chen, Yan); verify each new DOI/venue on the publisher or ACL Anthology page before it enters `refs.bib` (done this session: Li et al. 2021 and ULMFiT only; the rest of Consensus's list is UNVERIFIED).

Page budget: 6 pages. Compress diffusion and LLM-pair paragraphs (item 41) and the GAN check (item 42) to fund E1/E2, Holm families and the 5-budget table.

---

## D. Session schedule

### D.1 Run queue (priority order)

| Order | Block | Contents | GPU-h | Wall | Cum. wall |
|---|---|---|---|---|---|
| 0 | CPU, in parallel from the start | E0a, E0b (owner runs locally or on a CPU Kaggle notebook) | 0 | | |
| 1 | Notebook 22 part 1 | Pre-flight: tests; re-run check G1 (A1_p1_s0, A20_p1_s0 must equal Pass 1/8a); E2 dose curve | 0.8 | 0.4 h | 0.4 h |
| 2 | Notebook 22 part 2 (same commit if the code is ready, else a second commit) | E1 pretrainings, then runs | 2.1 | 1.2 h | 1.6 h |
| 3 | Notebook 23 | E4 RoBERTa (tune, pretrain, runs) | 1.35 | 1.0 h | 2.6 h |
| 4 | Notebook 24 (optional) | E3 GAN-long + A1-long | 2.5 | 1.3 h | 3.9 h |
| 5 | Notebook 24 (optional) | E5 JEPA seeds | 1.8 | 1.0 h | 4.9 h |

E2 goes first because it needs almost no new code and its result decides whether E3 is promoted. E1 needs the data file and about half a day of code, so the first notebook can start while E1 code is written (code is reviewed before the commit that E1 uses).

### D.2 Blocks, start conditions, commands, criteria

All GPU runs are Kaggle notebooks run by the owner (rule 5); sessions only explain and write code. Structure follows notebook 20: clone the public repo at a pushed commit, run tests, check the data manifest, run the plan, analyse on CPU, zip, download.

| Block | Start condition | Command (owner) | Duration | Success | Failure |
|---|---|---|---|---|---|
| Predictions freeze | before any run | commit `vault/Pass 11 Predictions.md` (status FROZEN, hypotheses final); push | 5 min | commit hash recorded in the notebook | do not start a run notebook |
| E0a/E0b | `results/` unpacked locally | new CPU notebook or `python -m gainsplit.analysis_pass11 --cpu` (to be written) | 10 min | CSVs in `results/analysis_pass11/`, bootstrap re-run of a known contrast (A20-A1 at 1% = +7.42) reproduces | contrast does not reproduce: stop, fix before reporting |
| Notebook 22 | predictions committed; code pushed; tests pass; `gainsplit-processed` updated with `budgets_isarc.json` (for E1 only) | Kaggle: Accelerator GPU T4 x2, Internet on, inputs: processed data, notebook 15 output (seed-0 encoder). Save Version -> Save & Run All | 1.6 h | all runs have `result.json`; G1 identical; encoder hash check PASS; pretrain receipts present | tests fail: stop, paste output. Runs are resumable: re-running the plan cell skips finished runs |
| Notebook 23 | notebook 22 done or GPU free | same; no extra input | 1.0 h | tuning table, lr frozen by rule, 20 runs, receipts | pretraining diverges (dev MLM loss not below its start): report as failure, do not retune |
| Notebook 24 | tier A and B done and quota left | same | up to 2.3 h | as above | cut (D.5) |

### D.3 Checkpointing, resume, naming, determinism, seeds

- Resume: `train_run` skips a run whose `result.json` exists (written last, atomically); `pretrain` skips if `pretrain_info.json` exists; `runner.run_all` re-queues the rest. If a session dies, re-run the plan cell. Encoders stay in `/kaggle/working/ckpt/...` (about 440 MB each, not zipped, git-ignored); downloads happen only for the zip.
- Naming: plans `pass11_dose`, `pass11_isarc`, `pass11_roberta`, `pass11_ganlong`, `pass11_jepa_seeds`; run ids from `RunConfig.run_id` with tags `dose<steps>`, `isc28`, `isc100`, `hs`, `hsub`, `rob`, `long35`, `jt1`, `jt2`; folders `results/pass11_core/`, `pass11_roberta/`, `pass11_optional/`; logs `notebook_22.log` etc., filed with the zips as receipts (standing rule: every zip and log filed under `results/<plan>/` and committed; `.gitignore` already keeps `results/**/*.log`).
- Determinism: same seeds 0-4 (E1: 0-9), `set_seed`, deterministic data order from the seed; every `result.json` records the commit, data manifest hashes, torch/transformers versions and device. G1 re-runs (A1 and A20 at 1%, seed 0) must reproduce Pass 1/8a predictions exactly after the code change; if not, stop.
- Guards so no run touches dev/test in tuning: (1) tuning plans (E4 stage 1) are built with `eval_test=False`, and a test asserts it; (2) pretraining texts are train minus dev only (existing test: changing labels leaves weights unchanged; new test: no test or dev id appears in any pretraining text for iSarcasm or Headlines subset); (3) E4's lr is chosen by the dev rule written above, before any test score exists; (4) no setting is changed after the predictions commit (a change needs a new dated entry and a re-freeze); (5) the notebook prints the commit and refuses to start if `vault/Pass 11 Predictions.md` in the cloned repo is not marked FROZEN.

### D.4 Pre-registration procedure (rule: predictions before runs)

1. `vault/Pass 11 Predictions.md` is the skeleton (created now, status DRAFT). The owner edits and sets status FROZEN with the final hypotheses, then commits and pushes before the run commit; `result.json` records the commit, so order is auditable.
2. Predictions follow the existing convention (IDEA labels, "any outcome is reportable", supported only if direction and p_holm < 0.05).
3. Add a pointer to the block in `vault/Experiment Plan.md` ("Pass 11 plan") at freeze time (not done yet, see Q7).

### D.5 Fallbacks and cuts

Cut in this order when time or quota is short: E5, E3, E4, E1's A20-iS100 sensitivity, E1's second budget (keep N = 228), E2's 350-step point. Never cut: G1 re-run, E1 main arms at N = 228 (and 780), the size-matched control, E2's 100- and 3,000-step points, E0a/E0b.
- A run fails: re-run the plan cell once; if it fails twice, read the run log, record the failure in the Session Log, continue with the rest.
- E1 code not ready: start notebook 22 with E2 only (separate commit), E1 as notebook 22b.
- Quota hit: stop at a notebook boundary; the paper reports what exists (E2 and E0 alone already justify the exposure rewrite).
- Do not run anything on dev/test "to see" while waiting.

### D.6 Parallel text work while the GPU is busy

Items 1-4, 13-15, 17-18, 20-23, 24a/b, 25-27, 29-36, 41-45 (Section C) need no result; do them first. Wait for results: abstract result sentence, E1 paragraph, exposure paragraph (E2/E3), RoBERTa sentence, JEPA seed sentence, conclusion wording, polarity paragraph (E0a, available early).

### D.7 Decision points

| Result | Meaning | What changes |
|---|---|---|
| E1: A20-iS gain in AUC at N = 228 and 780 is at least +3, robust, and the size-matched control keeps at least half the Headlines gain | gain is not an outlet artefact | Keep the title scope; rewrite contribution (iii): label efficiency holds on author-labeled data, shortcut reliance and zero-shot failure remain; Discussion recommends the TAPT baseline more widely |
| E1: gain between +1 and +3 or only at one budget | partial | Say "smaller and noisier"; keep scope words in title and abstract; headline +7.4 stays labeled "on Headlines" |
| E1: gain below +1 or interval includes 0 at 228 and 780, control keeps the gain | gain is dataset-specific (outlet-driven) | Rewrite: "TAPT's low-label advantage on News Headlines does not replicate on author-labeled sarcasm"; title must say Headline; Conclusion and Discussion lead with the scope; contribution (iii) becomes a main finding |
| E1: gain below +1 and the size-matched control also fails | pool size, not outlet, may explain it | State both; do not claim an outlet explanation; add a limitation |
| E2: 100-step gain at most +1 | exposure matters for MLM too; GAN's null is partly exposure | Promote E3 to must-run; reframe III-C as "fixed-step single-stage vs two-stage"; weaken "generation does not help" to "not at 0.28 passes" |
| E2: 100-step gain at least +4 | the objective, not exposure, separates them | Strengthen the paper: even at GAN-matched exposure MLM wins; E3 demoted |
| E3: A4 - A1 at 35x steps is about 0 | GAN null survives long training | Add one sentence; keep claim |
| E3: A4 - A1 clearly positive | GAN needs many passes | Rewrite Q1 GAN paragraph; GAN is no longer a null |
| E4: A20-R gain at 1% below +2 or n.s. | gain is encoder-specific | Scope the abstract to bert-base; discuss RoBERTa's news-heavy pretraining |
| E4: gain similar to BERT's | not encoder-specific (for two encoders) | Add one sentence |
| E0b: a single-route +1.6 interval includes 0 | those gains are weaker than reported | Report with the wider interval; main +7.4 stays |
| E0a: DiD interval excludes 0 (positive) | helper pull on polarity flips | Keep the section as a mechanism candidate; otherwise report null |
| E5: JEPA below MLM on all three encoders | comparison robust | Remove the single-seed caveat; keep the warm-start caveat |

---

## E. Risks and open questions

### Contrarian reading and how the plan detects it

The strongest case against "Just keep pretraining": on News Headlines the label is the outlet, so the task is outlet classification, and in-domain MLM on 22.8k headlines is the ideal way to learn outlet style (it has outlet cue-word reliance identical to BERT's and no transfer). The advantage may then be a property of a style-labeled corpus, not of semi-supervised sarcasm detection; the ten-route ranking, and "absorbs the others", could differ on author-labeled data (Li et al. 2021 report complementarity elsewhere). A second reading: single-stage baselines starve on U. Detection: E1 (with its size-matched control) for the first; E2 and E3 for the second; E4 for encoder dependence; E0b for training-split variance. The in-domain reference AUC on iSarcasmEval is only 65.8, below word count (68.7), so E1 has low ceiling and noisy tests; that is why it uses 10 seeds and AUC as the primary metric.

### Other risks
1. **Power in E1:** 200 positives in the test set; an AUC difference under about 2 points will not be detectable even with 10 seeds. A null is then "no detectable gain", not "no gain".
2. **E1 TAPT length is a free parameter.** Fixed above (28.7 passes) with 100 passes as sensitivity; no further variants.
3. **Page limit:** the added content competes with six pages; compress as in Section C.
4. **Kaggle quota: 8 h available (owner, 2026-11-19).** Whether a T4 x2 session bills wall time or per-GPU time is still UNVERIFIED. Budget both ways: if wall time, the whole plan (4.9 h wall, about 5.2 with overhead) fits; if per-GPU, 8.5 GPU-h does not. Rule: run A, B, then E3 (6.7 GPU-h cumulative) and cut E5 first; keep at least 1 h in reserve for re-runs (G1 check, one failed block). Check the quota page after notebook 22: if it consumed about 1.6 h, billing is wall time; if about 3.2 h, per-GPU, and E3 is dropped unless E2 promotes it.
5. **E4 masked evals not available for RoBERTa**; the cue-mask comparison stays BERT-only.
6. **Consensus claims not checked** beyond those in the table (Gururangan gain ranges, Shi et al. 3.6 points on IMDB, SarcasmBench conclusions, Du et al. design): UNVERIFIED; check before any enters the text.
7. **Deadline:** QPAIN 2027 deadline 20 Dec 2026 (verified 2026-11-11); owner planning date 10 Dec. About three weeks.

### Questions (answer together)

- **Q1.** Title: (a) keep it and bind scope in the abstract, (b) append "Headline Sarcasm Detection", (c) Consensus's alternative wording? Recommended (b).
- **Q2.** E1 design: OK with N = 228 / 780 / 3,120, 10 seeds, AUC primary, passes-matched TAPT (1,400 steps), 100-pass sensitivity, Headlines size-matched control?
- **Q3.** Is the tier plan OK: A (E2, E1) today; B (E4) if time; C (E3, E5) only afterwards? Or drop E4 in favour of E3?
- **Q4.** ANSWERED: 8 h quota. Still open: does a T4 x2 session count as one or two hours per hour (see risk 4 for how to find out after notebook 22)?
- **Q5.** E4 learning rate: let the notebook pick by the dev rule (Section B), or fix 2e-5 now?
- **Q6.** May a coding session (repo rule 7: written on a cheaper model, reviewed before commit) write the E1/E2/E4 code and tests? And do you want to generate `budgets_isarc.json` yourself with a guided notebook cell?
- **Q7.** Repo bookkeeping (open): the root instructions file asks for a Session Log entry and a hub update at the end of a session; this planning step created only `PLAN.md` and the predictions note. To do: Session Log entry, pointer in `vault/Experiment Plan.md`, link from the hub to the predictions note (the note links to the hub but is not linked back).
- **Q8.** Author lists for references [3] and [16] (names appear in the reviews, unverified): can you confirm from the publisher pages?
- **Q9.** Are you willing to change the abstract wording before results (scope words, "on this dataset"), or wait for E1?

---

## F. Agreed lane (2026-11-19, supersedes the tiers in B.1 and D.1 where they differ)

**One question: where does the seed recipe's gain come from?** The seed recipe is GAN-BERT plus sentiment/emotion helper heads. The paper takes it apart, then shows what does carry a gain, then bounds the claim. Diffusion, LLM pairs, JEPA and the full factorial leave the main text (one sentence or one appendix line each).

**One sentence (for the owner to word later; no title, no paper text yet):** neither the GAN nor the helper heads carry the seed recipe's gain; continued pretraining (plus FGM and self-training) beats the recipe by a wide margin at low labels, and the seed recipe's reported accuracy is matched at high labels without the GAN or helpers.

**What the paper claims, and what it must not**
- Claims: (1) GAN null, helper-head null for the seed design (A3: -0.13 at 1%; full recipe A5: -2.3 at 1%; only the L-only variant A2 gains, +1.6, and shuffled labels keep 72%); (2) pretraining + FGM + helpers + self-training (A36) vs the seed recipe re-implemented (A5), same split and metric: 88.8 vs 78.1 at 1%, 91.6 vs 88.4 at 10% (paired bootstrap A36 - A5 still to compute, CPU); (3) at 70% labels (the seed paper's regime) everything ties around 93.4-93.9, so 93.88 needs neither GAN nor helpers; (4) bound: outlet reliance and no transfer, plus E1.
- Must not claim: that we exceed 93.88 (MLM + FGM at 100% = 93.64 < FGM alone 93.90; split and metric differ); that helper heads carry a gain; headline A34 (88.9, picked after seeing test), use A36 (pre-registered H10.3).

**Seed paper facts that shape the wording (VERIFIED 2026-11-19 from the PDF, `D:\Cloud Extension\QPAIN\Working paper\MimicGuard\Enhancing_Sarcasm_Detection_Using_GAN-BERT_with_Multi-Task_Learning.pdf`; not copied into the repo because it is a licensed IEEE copy and the repo is public):**
- Table II: accuracy 93.88 (GAN-BERT + MTL), 92.04 (BERT), 87.16 (LSTM), 85.18 (CNN), and the support column is 28,619 for every model, i.e. the whole dataset, while the text says the 20% test set is 5,724 items. So the reported numbers appear to be computed over all headlines, training ones included.
- Their own BERT baseline scores 92.04; our plain BERT with all labels scores 93.35 on a held-out test. Their reported gain of +1.84 over BERT is smaller than the gap between their BERT and ours (different split and metric, so INFERRED).
- GAN-BERT + MTL: 70% of the 80% training split labeled (about 16,026) and 30% unlabeled (about 6,868); this is our 70% budget. BERT, LSTM, CNN were trained on the full 80%. GAN-BERT ran about 25 epochs (Fig. 3), BERT about 9-10 (Fig. 4).
- No ablation: there is no GAN-BERT-without-MTL row and no MTL-without-GAN row, only BERT, LSTM, CNN. The sentence that both the auxiliary supervision and the adversarial loss contribute is not backed by a comparison.
- Architecture differences from our A5 (state them as re-implementation limits): Fig. 2 draws three generator-discriminator pairs (sarcasm, sentiment, emotion), ours has one GAN on the sarcasm head plus shared linear helper heads; they eliminate the neutral sentiment class (binary sentiment), ours keeps three; emotion mapped to joy/anger/disgust/surprise/sadness style classes from GoEmotions or pysentimiento (Table I also shows "Frustration"), ours uses five classes from one GoEmotions model. Their lambda values are not given.
- Claims in earlier lines of this section that were marked UNVERIFIED are now settled by the above. Whether the instructor accepts a paper that says her recipe adds nothing is a question for her, not answerable from the PDF. Wording rule from the earlier note: frame it as "which part of the recipe helps when", describe our model as a re-implementation (original code unavailable), and state the comparison facts, not conclusions about her paper.

**Revised experiment list (GPU quota is not a constraint: other accounts available; estimates below are INFERRED run-minutes)**
| Prio | Experiment | Why | GPU-h |
|---|---|---|---|
| CPU | E0a, E0b, plus A36 - A5 at 1% and 10%, A36 - A1, paired bootstrap on existing predictions | headline contrast; training-split variance; polarity placeholder | 0 |
| Must | E2 MLM dose curve | exposure fairness, cheap | 0.7 |
| Must | **E6 curve**: A36 at 5% and 30% (A5, A1, A20 exist at 5/30%); 70% budget (owner generates nested subsets): A1, A5, A7, A20 at 70%, seeds 0-4. A36 is skipped at 70% (self-training needs U and adds nothing at high labels; its cost is about 3x A7) | one figure, two curves, across budgets; tests "seed number needs no GAN/helpers" | about 1.6 (A36 5/30%) + 3.1 (70%) |
| Must | E3 GAN, about 10 passes over U, paired with A1 at the same steps | first objection to "GAN does not help" | 2.5 |
| Should | E1 in-domain iSarcasmEval with size-matched control | decides how strongly "dataset-specific" can be said | 2.1 |
| Optional | A5 (GAN + helpers) on top of the MLM encoder | shows the seed recipe adds nothing once pretrained; needs GAN arms to accept `init_dir` (small code change, test) | about 0.5 |
| Cut from this lane | E4 RoBERTa, E5 JEPA seeds | not needed for the lane; mention as limitations | |

Must-run total about 8 GPU-h (about 4-5 h wall on T4 x2); with E1 and the optional arm about 10.6. 70% budget file: owner generates (nested subsets per seed, same procedure as `budgets.json`) and adds it to `gainsplit-processed` with a manifest entry.

**Cuts in the main text:** diffusion arms, diffusion consistency, LLM pairs, JEPA, the 2x2x2 table (keep one sentence: pretraining absorbs the rest), the GAN dev-check paragraph, most of the reproducibility detail for those arms. Section C's text-edit list stays, limited to the items on the retained arms.

**Decision points added**
- A36 - A5 not robust at 1% or 10% (unexpected): the headline contrast fails; stop and re-plan before writing.
- 70% curve: any of A1/A7/A20 below A5 by more than 0.5 point: the "93.88 needs no GAN/helpers" sentence is dropped; report as is.
- E3 shows A4 - A1 clearly positive at about 10 passes: the first claim changes to "the GAN helps only with long training".

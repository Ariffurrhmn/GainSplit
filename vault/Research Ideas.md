---
tags: [ideas]
updated: 2026-11-13
---

# Research Ideas

Hub: [[00 - Project Context]]. Parked: [[00 - Backlog]].

- **IDEA** RQ3 shortcut check: mutual information between teacher labels (sentiment/emotion) and the sarcasm label. CPU only, one afternoon. Tests whether helper labels are outlet proxies ([[Headlines Dataset]]).
- **IDEA** Source-masked headline test set.
- **IDEA** Cheap per-sample weighting: head-gradient SLGrad-lite or teacher-agreement weights ([[SLGrad]]). The hold-out batch must be charged to the label budget.
- **IDEA** Self-training as a cheap SSL control next to the GAN.
- **IDEA** Supervised adversarial control (FGM), motivated by [[Zihan 2026]].

## Pass 4 candidates after instructor feedback (2026-11-11)
Instructor: an ablation that only shows the seed model is weak has no impact; the paper needs something new that gives more gain. Directions built on what our own results show works:
- **IDEA** Combine the three mechanisms that each gave about +1.6 at 1% (FGM, self-training, helper heads on L) into one recipe; test whether they stack. Never run together so far ([[Bootstrap - All Passes]]).
- **IDEA** Replace the GAN with perturbation-based semi-supervised adversarial training (VAT on U, FGM on L; Miyato 2017). Story: the adversarial idea works, the GAN form of it does not. Novelty for sarcasm UNVERIFIED.
- **IDEA** Cue-dropout training: randomly mask outlet-cue words during training to cut the ~3-point shortcut gap and test transfer to iSarcasmEval ([[Pass 3 Results]]). Risk: outlet and label are fully confounded in Headlines; transfer may stay near chance.
- Impact framing (INFERRED from the summary table): A1 needs 30% labels for 90.85; A7 alone reaches 89.76 at 10%. A recipe at about 91 with 10% labels would be "same F1 with 3x fewer labels".

## Generative AI and generalisation (instructor request, 2026-11-11)
Instructor asked for a generative-AI variant and for sarcasm detection that generalises across datasets. Options (all IDEA). Novelty checked 2026-11-11: [[Novelty Check - LLM Counterfactuals]] (technique not new; narrow gap remains).
- **IDEA** LLM as the generator (replaces the GAN generator): an open LLM writes real fake text (sarcastic rewrites of plain headlines and the reverse) instead of fake vectors.
- **IDEA** LLM counterfactual augmentation against the outlet shortcut: rewrite each headline into the opposite class keeping topic and outlet-style words, so cue words no longer predict the label ([[Pass 3 Results]] cue gap ~3 points).
- **IDEA** LLM style transfer to new domains: rewrite headlines as tweets / Reddit comments to add domain variety before testing on iSarcasmEval.
- **IDEA** LLM as teacher: LLM sarcasm pseudo-labels on unlabelled target-domain text, or rationale (incongruity) labels as helper task; current sentiment teacher is weak (gold kappa 0.24, [[Gold Audit Results]]).
- **Baseline needed** LLM zero/few-shot on every test set. Risk: Headlines may be in LLM pretraining data (contamination).
- **Evaluation protocol** train on one dataset, test on the others (Headlines, iSarcasmEval, SemEval-2018 Task 3, SARC subset); leave-one-dataset-out. Extra datasets not downloaded; licences in [[Datasets Index]].

## Paired-generation LLM choice and guardrails (2026-11-11, IDEA, not decided)
- Candidate generators: an open-weights 7-8B instruct model run on Kaggle (e.g. Qwen2.5-7B-Instruct, Apache 2.0; Mistral-7B-Instruct, Apache 2.0, same family Horvitz 2024 tested). Licences and Kaggle availability UNVERIFIED; check model cards. Newer versions may exist.
- Guardrail risk: refusals or softened sarcasm on sensitive topics (Horvitz 2024 notes GPT-4 refused offensive jokes but edited them away freely). Effect is asymmetric (sarcastic side fails more).
- Handling: drop the whole pair if either side fails (keeps classes balanced); log refusal rate by topic; compare tone of generated sarcasm with real Onion headlines; report results on sensitive vs other test items; pilot ~200 seeds and read ~50 pairs before the full run. Avoid "uncensored" community fine-tunes (hard to defend in review).
- Who judges the pairs (IDEA): (1) automatic rules (refusal text, empty, near-copy of seed, length); (2) an LLM judge from a different family than the generator, blind to the intended side; (3) the owner labels a random ~200-item sample blind (gold-audit app pattern, `experiments/gold_audit.html`), ideally with a second person for kappa, which also measures judge accuracy; (4) final test is downstream: gains on human-written test sets only (Headlines test, Unfun pairs, iSarcasmEval).
- Mixture of generators (owner idea, 2026-11-11, IDEA): 3 open model families (e.g. Qwen, Gemma, Llama/Mistral) to widen humour styles and blur any single-model fingerprint. Rules: both sides of a pair from the same model; models assigned to seeds at random in equal shares; judge from a family not used for generation (Phi-3.5-mini-instruct in the pilot; Claude Haiku dropped, no API key). Gives a built-in ablation: one model vs mixture at the same pair count. Mixing does not fix weak humour by itself; filtering does. Gemma and Llama are gated (accept licence on Hugging Face/Kaggle).
- Pipeline order (2026-11-11, IDEA): generate first, freeze, then train. Pilot ~200 seeds -> check -> full generation -> automatic filter + judge -> owner blind audit -> freeze as a new Kaggle dataset version with hashes -> training runs reuse the frozen file. Seeds from the train split only (never dev/test); each run uses pairs whose seed is in its own U. Check overlap between Unfun Onion headlines and Headlines train/test before using Unfun as a test set. Generator set: Qwen, Gemma, Mistral.

## Explainability (2026-11-12; adopted into the Pass 4 plan, see [[Experiment Plan]])
- Owner asked whether the models can be made explainable. Proposal: token attributions with Integrated Gradients (Captum) on a fixed test sample, for A1 and the Pass 4 pair-trained model, run in Kaggle after training.
- Main use is a number, not pictures: share of attribution falling on the top-100 outlet cue words (`cue_words.json`). Prediction (IDEA): the pair-trained model puts less weight on cue words than A1, matching the cue-mask gap from [[Pass 3 Results]]. Plus 2-3 example heatmaps.
- Not attention weights (contested as explanations). LIME/SHAP possible but slower and noisier on short text.

## After Pass 4 (2026-11-13, IDEA, not decided)
Pairs as plain extra labeled rows cost 15-16 F1 points in-domain but lift iSarcasmEval AUC by about 21-25 points ([[Pass 4 Results]]). Ways to keep the transfer without the in-domain loss:
- Lower pair weight (e.g. 0.1 or 0.3 on the pair rows' loss), or fewer pair rows per step.
- Pair-ranking loss: ask only that the sarcastic side scores higher than its serious side, instead of giving both sides absolute labels.
- Two stages: train on pairs (A15-style), then fine-tune on L only.
- Length-matched pairs: drop or rebalance pairs whose sides differ a lot in length (length-only AUC 0.785 inside the pairs).
- More generally: an explicit "avoid these words" list for the generation prompt, or a filter on pairs that rely heavily on them, built from the full shortcut-pattern search in [[Pass 4 Results]] "Shortcut-pattern hunting in the generated pairs" (2026-11-14) — not just length, every word/phrase the AI leans on (cv_auc 0.951, well above length or cue words alone).
- Reframe: zero-label cross-domain sarcasm from LLM pairs (A15 AUC 76.2 vs in-domain 65.75), only if the iSarcasmEval length check comes out clean.


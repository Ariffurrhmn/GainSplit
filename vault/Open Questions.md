---
tags: [questions]
updated: 2026-11-09
---

# Open Questions

Hub: [[00 - Project Context]]. Risks: [[Risks]].

## Do first
- [x] Get full text of [[Chen 2022]]. Done 2026-11-08: lexicon-BoW soft target, LSTM, supervised only.
- [x] Get full text of [[Zihan 2026]]. Done 2026-11-09: GAN-BERT numbers copied from Lora 2023; fractions 3/30/40/50/100%; no MTL arm; no FGM-off ablation.
- [x] Supervisor meant the seed paper ([[Seed Paper - Irfan 2025]]) for the unavailable setup (owner, 2026-11-09). Zihan 2026 authorship still unknown.
- [ ] Owner: approve the proposed A7 spec, Bengali arm out of scope, optional VAT arm ([[Decisions]]).
- [ ] Confirm the QPAIN 2027 deadline against the live site ([[Audit - Venues]]). Planning date: 10 Dec 2026 (owner's choice; repo says 20 Dec). Page limit 6 (owner); format undecided.

## Instructor status (2026-11-09)
- Proposal shown to the instructor and accepted; she asked to prepare code and setup as soon as possible so runs can start. Paper draft will get her feedback before submission.
- All papers the earlier sessions asked for have been supplied. Nothing more is awaited. RQ2 is not paused (the earlier pause was a misreading: "the instructor's paper" is the seed paper, already in hand).

## Instructor status (2026-11-08)
- Owner asked the teacher about the repo. Reply: the conference was over 2 years ago; they do not know if they have access to the code, but will ask around. Q1 stays open.
- Owner has no answers for Q2-Q5 and will not ask further questions about the teacher's setup. Treat Q2-Q5 as permanently UNKNOWN: our runs are a documented re-implementation, compared with the seed numbers only qualitatively. Q6 and Q7 can be checked from the PDF and the dataset ourselves.

## For the instructor (see [[Seed Paper - Irfan 2025]])
1. Code or repo available?
2. Hyperparameters: l1, l2, lr, batch, max length, generator/discriminator size, noise dim, optimizer, seed.
3. Which emotion model, and the exact 28-to-5 class mapping (Table I shows "Frustration", outside the stated set).
4. What happened to neutral headlines?
5. Were the helper heads trained on the "unlabeled" 30%?
6. What were Table II metrics computed on? (supports sum to 28,619 = whole dataset, not the 5,724 test set; VERIFIED in PDF)
7. Dataset version and class counts (text says 14,000/14,619; Table II says 14,470/14,149; Misra abstract says ~13K sarcastic of ~28K).
8. Model selection and stopping rule.
9. Is she comfortable with a re-run compared to her numbers? Authorship expectations?

## Unverified facts to check
- Kaggle free GPU: weekly quota and session length (owner uses free Kaggle, 2026-11-08). Read it off the account page before Pass 1 ([[Experiment Plan]]).
- iSarcasmEval English train size; SARC and Kaggle headlines licences; Kaggle GPU quota; 2027 venue deadlines.
- Informal claim of URL/section leakage in Headlines (single GitHub source). See [[Headlines Dataset]].

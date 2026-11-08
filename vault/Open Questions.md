---
tags: [questions]
updated: 2026-11-08
---

# Open Questions

Hub: [[00 - Project Context]]. Risks: [[Risks]].

## Do first
- [ ] Get full text of [[Chen 2022]] (IEEE Access, open access; try the university library). Read its ablation and soft-label construction.
- [ ] Get full text of [[Zihan 2026]] (ICECTE, paywalled): which GAN-BERT variant, label fractions, any MTL arm.
- [ ] Check the IEEE QPAIN 2027 call: deadline, page limit, format ([[Audit - Venues]]).

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
- iSarcasmEval English train size; SARC and Kaggle headlines licences; Kaggle GPU quota; 2027 venue deadlines.
- Informal claim of URL/section leakage in Headlines (single GitHub source). See [[Headlines Dataset]].

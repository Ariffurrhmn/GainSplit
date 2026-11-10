---
tags: [risks]
updated: 2026-11-11
---

# Risks

Hub: [[00 - Project Context]]. Source: [[Deep Dive - Paper A]] section (g), [[Proposal A]].

1. **GAN effect null at ~16k labels** (INFERRED; GAN-BERT's own curves show gains vanishing at moderate label counts, see [[GAN-BERT]]). Mitigation: frame as "when does each component help"; RQ2/RQ3 and a second dataset.
2. **Ceiling effect:** headlines accuracy ~92-94%, differences 0.2-0.5 pt. Need 5 seeds and paired tests ([[Experiment Plan]]).
3. **Novelty threat:** [[Chen 2022]] (now read in full) is only a partial overlap: lexicon-BoW soft target, LSTM, fully supervised, no SSL or label budgets. Cite and contrast. Strongest verified overlap remains [[Dai 2021]].
4. **Label = outlet**, not perceived sarcasm ([[Headlines Dataset]]). RQ3 and the audit address it.
5. **Framing vs instructor:** a GAN-null result may read as criticism of her model. Frame constructively ([[Decisions]]).
6. **Paper B shares the QPAIN target:** keep question, data and claims distinct.
7. Reproduction is unfaithful without her hyperparameters ([[Open Questions]]).
8. **Limitation: original setup unavailable (2026-11-09).** Supervisor advised to expect no access to the original setup. Code, splits, hyperparameters and helper-label mapping of [[Seed Paper - Irfan 2025]] are unknown, and [[Zihan 2026]] omits FGM settings and split details. All arms are re-implemented under one protocol; absolute scores may differ from published ones; published numbers are context only. Claims are about differences between arms inside our setup. Draft wording in [[Decisions]].
9. **Zihan 2026 comparison is not controlled:** it must be cited as a cross-paper observation, not as evidence that a GAN is unnecessary ([[Zihan 2026]]).
10. **Public repo vs double-blind review (UNVERIFIED):** the repo is public (owner, 2026-11-09). If QPAIN 2027 review is double-blind, the submitted paper must not link to it, and the repo should not name the venue as "submitted". Check the anonymity rule in [[Audit - Venues]].
11. **Emotion helper labels are mostly one class (2026-11-10, VERIFIED counts):** 83% of headlines get `neutral_other` and fear has only 339. The emotion head may carry little signal, which would weaken any "emotion helps" result. Report the distribution in the paper; soft labels and the raw 28 scores (RQ2) partly address it. Checked 2026-11-11 (VERIFIED, [[Pass 3 Results]]): helper labels barely track sarcasm (NMI <= 0.004, probe AUC 0.546); the emotion-only row is not worse than both heads at 10%, and dropping emotion costs 0.47 at 1% (not robust).
12. **Single-annotator gold audit (2026-11-10):** the owner, who knows the hypotheses, is the only labeler of the 200 audit headlines, so there is no agreement score. Paper wording: one annotator, blind to sarcasm and teacher labels, fixed written guideline, seeded random sample; teacher accuracy is reported with this caveat.
13. **Unequal tuning across arms (2026-11-10):** A7 (FGM epsilon) and A3/A5 (helper weight w) each had one hyperparameter picked from 3 values on dev; the GAN arms (A4/A5 GAN part) and A6 used fixed literature settings. A reviewer may call the GAN under-tuned next to the winning FGM arm ([[Pass 2b Results]]). Answer: healthy GAN pilot, GAN null against A1 at every budget; state it in the limitations. Done 2026-11-11 (VERIFIED, [[GAN Check Results]]): generator lr 1e-5 / 2e-4 at 10% and 3x steps at 1% did not improve A4 on dev under the pre-set rule. Limitations still say the sweep was small.
14. **Rounded prediction files (2026-11-11, VERIFIED):** probabilities are saved to 6 decimals, so confident models have many tied scores and AUC recomputed from the files differs from the in-training AUC by up to 1.3 points per seed. Conclusions unchanged; the paper takes AUCs from one source ([[Decisions]]).
15. **Outlet shortcut in every model (2026-11-11, INFERRED from VERIFIED numbers):** masking outlet-cue words costs about 3 points more than random masking for every Headlines-trained arm ([[Pass 3 Results]]). A reviewer may say the in-domain gains measure outlet detection. Answer: report it as a finding (RQ3), it applies equally to all arms, so the arm comparisons stand.

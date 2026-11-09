---
tags: [risks]
updated: 2026-11-08
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

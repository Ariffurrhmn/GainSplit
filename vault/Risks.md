---
tags: [risks]
updated: 2026-11-08
---

# Risks

Hub: [[00 - Project Context]]. Source: [[Deep Dive - Paper A]] section (g), [[Proposal A]].

1. **GAN effect null at ~16k labels** (INFERRED; GAN-BERT's own curves show gains vanishing at moderate label counts, see [[GAN-BERT]]). Mitigation: frame as "when does each component help"; RQ2/RQ3 and a second dataset.
2. **Ceiling effect:** headlines accuracy ~92-94%, differences 0.2-0.5 pt. Need 5 seeds and paired tests ([[Experiment Plan]]).
3. **Novelty threat:** [[Chen 2022]] may already do soft machine labels plus an ablation (unread). Verified fallback threat: [[Dai 2021]].
4. **Label = outlet**, not perceived sarcasm ([[Headlines Dataset]]). RQ3 and the audit address it.
5. **Framing vs instructor:** a GAN-null result may read as criticism of her model. Frame constructively ([[Decisions]]).
6. **Paper B shares the QPAIN target:** keep question, data and claims distinct.
7. Reproduction is unfaithful without her hyperparameters ([[Open Questions]]).

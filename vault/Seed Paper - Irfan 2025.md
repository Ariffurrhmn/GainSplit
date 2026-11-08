---
tags: [literature, seed]
updated: 2026-11-08
---

# Seed Paper - Irfan 2025

Instructor's paper, IEEE QPAIN 2025. Hub: [[00 - Project Context]]. Index: [[Literature Index]].

- GAN-BERT + multi-task learning; sentiment and emotion auxiliary tasks with labels made by pretrained models.
- Reports 93.88% accuracy on [[Headlines Dataset]].
- Three entangled ingredients: adversarial SSL, helper heads, helper labels. Our study separates them ([[Proposal A]]).
- Key observation (INFERRED): helper labels exist for every headline, so helper heads can learn from the unlabeled pool, making MTL a semi-supervised mechanism that competes with the GAN.
- Reproduction details known and unknown: [[Experiment Plan]], [[Open Questions]].
- Base method: [[GAN-BERT]].

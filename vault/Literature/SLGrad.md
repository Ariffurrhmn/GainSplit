---
tags: [literature]
updated: 2026-11-08
read_depth: FULL
---

# SLGrad

Hub: [[00 - Project Context]] · Index: [[Literature Index]] · Source table: [[Deep Dive - Paper A]]

Gregoire et al., Applied Intelligence 2024, arXiv 2306.04519. https://arxiv.org/pdf/2306.04519 (VERIFIED)

- Per-sample, per-task weights = clamped cosine between the sample gradient and a main-task validation gradient.
- Needs a labeled hold-out batch each step, per-sample gradients (Opacus), 1 forward + 2 backward passes. Tested only on toy, CIFAR-10 and Multi-MNIST with LeNet-5; no NLP, no teacher-made labels. Not best on clean CIFAR-10.
- Template for the cheap weighting arm in [[Research Ideas]]; restrict gradients to heads or last layer. Compute in [[Experiment Plan]].

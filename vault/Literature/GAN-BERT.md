---
tags: [literature]
updated: 2026-11-08
read_depth: FULL
---

# GAN-BERT

Hub: [[00 - Project Context]] · Index: [[Literature Index]] · Source table: [[Deep Dive - Paper A]]

Croce, Castellucci, Basili, ACL 2020. https://aclanthology.org/2020.acl-main.191.pdf (VERIFIED)

- G = MLP on 100-d noise; D = MLP with k+1 outputs. Unlabeled pool up to 100x labeled; labeled copies replicated by log(|U|/|L|). lr 2e-5, 3 epochs, 5 shuffles averaged.
- Own curves: gains shrink as labels grow (SST-5 ~1,700 labels, MNLI ~2,000, QC-coarse ~1,000; still ahead until 40-50% on 20N and QC-fine). Headlines at ~16k labels is far past that, so a null GAN effect is likely (INFERRED). See [[Risks]].
- Base of [[Seed Paper - Irfan 2025]].

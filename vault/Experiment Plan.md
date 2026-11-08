---
tags: [experiments, plan]
updated: 2026-11-08
status: not started
---

# Experiment Plan

Hub: [[00 - Project Context]]. Pitch: [[Proposal A]]. Prior work: [[Deep Dive - Paper A]].

## Conditions
Reference arm = faithful re-implementation of [[Seed Paper - Irfan 2025]]. Then: supervised BERT; + helper heads (hard, soft, weighted, lexicon); GAN only ([[GAN-BERT]]); GAN + helper heads; self-training; FGM supervised adversarial control; TF-IDF + LR.

## Protocol
5 budgets (1/5/10/30/100%) x 5 seeds; mean +- std; paired significance tests; ROC-AUC and calibration. Confirm the main finding with RoBERTa. Transfer to [[iSarcasmEval]]. Gold audit of ~200 headlines. MI check ([[Research Ideas]]).

## Data
[[Headlines Dataset]] (main), [[iSarcasmEval]] (transfer). See [[Datasets Index]].

## Compute (INFERRED, not benchmarked)
BERT-base, max_len ~48, fp16, T4: ~1-1.5 min/epoch at 100% labels. ~175 runs ~ 9 GPU-h; with a gradient-weighting arm ~15-25 GPU-h total. Fits a free Kaggle GPU or one RTX card.

## Known reproduction details (VERIFIED from the seed PDF)
bert-base-uncased; 80/20 split; 70% of train labelled, 30% unlabelled; sentiment from cardiffnlp/twitter-roberta-base-sentiment (neutral eliminated); emotion from a GoEmotions model or pysentimiento mapped to 5 classes; L = L_s + l1 L_sent + l2 L_emo + L_adv; per-head G/D pairs; punctuation removed; ~25 epochs GAN-BERT vs ~10 BERT. Unknowns: [[Open Questions]].

## Status
Nothing implemented. Code in `src/`, configs and run scripts in `experiments/`, outputs in `results/`. Log runs in [[Session Log]].

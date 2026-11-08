---
tags: [proposal, sarcasm, semi-supervised, multi-task]
created: 2026-11-02
updated: 2026-11-08
target: IEEE QPAIN 2027
status: draft proposal, not yet shown to the instructor
---

# Proposal A: Where Does the Gain Come From? Separating Adversarial Training, Multi-Task Helper Heads and Helper-Label Quality in Semi-Supervised Sarcasm Detection

Hub: [[00 - Project Context]] · Parked ideas: [[00 - Backlog]]

## Motivation

Irfan et al. (IEEE QPAIN 2025) combine GAN-BERT with multi-task learning, using sentiment and emotion as auxiliary tasks, and report 93.88% accuracy on the News Headlines dataset. Their framework has three ingredients at once: adversarial semi-supervised training, auxiliary heads, and auxiliary labels produced by pretrained models. This proposal asks how much each ingredient contributes, and when.

One observation motivates the study. The auxiliary labels are machine-made, so they exist for every headline, including the ones whose sarcasm label is withheld. The auxiliary heads can therefore learn from the whole pool while the sarcasm head sees only the labelled part. Multi-task learning then acts as a semi-supervised mechanism in its own right, competing with the GAN. A controlled comparison of the two has not, to our knowledge, been reported for sarcasm.

## Research questions

1. **RQ1.** At 1%, 5%, 10%, 30% and 100% labels, how much of the gain over a plain BERT comes from (i) the GAN, (ii) the auxiliary heads trained on the unlabeled pool, and (iii) both together?
2. **RQ2.** How much does auxiliary-label quality matter? Compare hard model-made labels (as in the original setup), soft or confidence-weighted labels, lexicon-based sentiment labels, teacher-model swaps, and a controlled synthetic noise level, checked against a hand-labelled audit of about 200 headlines.
3. **RQ3.** Do the findings hold on a second sarcasm dataset (iSarcasmEval), and when helper-label information is only an outlet or style proxy rather than sentiment?

## Approach

- **Reference arm:** a faithful re-implementation of the original configuration, so every comparison is against her setup.
- **Conditions:** supervised BERT; + auxiliary heads (hard, soft, weighted, lexicon-based); GAN only; GAN + auxiliary heads; self-training; a supervised adversarial control (FGM fine-tuning); TF-IDF + logistic regression.
- **Protocol:** 5 label budgets x 5 seeds, mean and standard deviation, paired significance tests, ROC-AUC and calibration. Confirm the main finding with one stronger encoder (RoBERTa).
- **Data:** News Headlines (Misra and Arora); iSarcasmEval (English) for transfer. Dataset licences to be checked before use.
- **Compute (rough estimate, not benchmarked):** about 15 to 25 GPU-hours, which fits a free Kaggle GPU or one RTX card.

## Why this is new, and what it builds on

Closest work we read in full: GAN-BERT (Croce et al., ACL 2020), whose own curves show the gain shrinking as labelled data grows; Savini and Caragea (Mathematics 2022), who vary dataset size and domain but use sequential transfer without a GAN; Chen et al. (IEEE Access 2022), who use lexicon-based soft sentiment labels at full labels only; and Dai et al. (arXiv 2021) on weakly labelled auxiliary tasks for multimodal affect. We found no study that separates the adversarial and multi-task effects for sarcasm across label budgets. This is a negative-search result from partial coverage and should be treated as provisional.

## Expected outcomes and risks

- With about 16,000 labels the GAN effect may be small. That is still a useful, publishable finding when framed as "when does each component help", which also tells readers when the original method is most worth using.
- Gains on this dataset are near ceiling (about 92% to 94%), so differences may be 0.2 to 0.5 points. Seeds and paired tests are therefore essential.
- The headlines label reflects the news outlet, not necessarily perceived sarcasm, so RQ3 and the audit are part of the design, not extras.

## Target venue

**IEEE QPAIN 2027** (stated by the user, 2026-11-08). The call for papers, deadline, page limit and format are not yet checked. The instructor's seed paper appeared at QPAIN 2025, which suits Paper A. Fallbacks if QPAIN falls through are in [[Audit - Venues]] (WASSA 2027, EACL 2027 SRW and others; dates unverified). Paper B shares the target with Paper A, so the two papers must be clearly distinct in question, data and claims.

## What we would like from the instructor

1. Whether code, hyperparameters (the loss weights, learning rate, batch size, generator and discriminator sizes) and the emotion-model mapping can be shared, so the reproduction is faithful.
2. Which data the Table II metrics were computed on, and the class counts used, so our evaluation protocol matches hers.
3. Whether she is comfortable with a re-run being compared against her reported numbers, and her expectations on authorship.

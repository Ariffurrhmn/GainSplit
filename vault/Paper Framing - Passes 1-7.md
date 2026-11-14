---
tags: [paper, plan]
created: 2026-11-14
updated: 2026-11-14
---

# Paper Framing - Passes 1-7

Hub: [[00 - Project Context]]. Current draft: [[Paper Draft]] (Passes 1-3 only, 6 pages). Results: [[Bootstrap - All Passes]], [[Pass 3 Results]], [[Pass 4 Results]], [[Pass 5 Results]], [[Pass 6 Results]], [[Pass 7 Results]]. Venue: QPAIN 2027, 4-6 pages, double-blind, deadline 20 Dec 2026 ([[Audit - Venues]]); planning deadline 10 Dec (owner).

Status: IDEA. A proposal to discuss with the instructor; nothing decided. Pass 5-7 numbers are INFERRED until notebook 14 (bootstrap) has run.

## The problem to solve with the instructor
Her feedback (2026-11-11): the ablation alone lacks impact; it needs a new method with more gain. The method tried for that (LLM-generated sarcastic/serious pairs, Passes 4-6) gives no in-domain gain, and its transfer gain is mostly a length shortcut. The one ingredient that helps at every label budget is FGM, which is not new. So the paper cannot honestly claim a new method with more gain. It can claim a controlled answer to "what actually helps", with two negative findings explained by their mechanism.

## What the evidence supports (one line each)
1. The GAN never helps; a sweep and longer training do not change that (Passes 1, GAN check). Robust.
2. FGM helps at every budget: +1.6 at 1% and 10%, +0.55 at 100% (5/5 seeds; 93.90% accuracy, level with the seed paper's reported 93.88% on a different split, which is quoted as a reference only, [[Decisions]] 2026-11-11).
3. Helper heads help at 1% mostly as a regulariser: most of the gain survives shuffled labels (Pass 3).
4. Every model leans on outlet words (cue-mask gap about 3 points) and none transfers to iSarcasmEval (Pass 3).
5. LLM pairs as extra training rows cost 10-16 F1 points (Passes 4-5); down-weighting and cleaning recover only part of it.
6. The pairs' transfer gain is mostly length (word count alone AUC 68.7), and the generated text carries AI register words absent from real headlines (word model cv AUC 0.95) (Pass 4).
7. Two-stage training (pairs first) restores in-domain parity at 10% and 100%, but the transfer falls below word count (Passes 6-7). No setting gives both.

## Options
| | Framing | Fits the 6 pages? | Risk |
|---|---|---|---|
| **A (recommended)** | **One diagnostic paper:** "what actually helps low-resource sarcasm detection", testing two popular routes to extra gain (GAN-BERT with helper tasks; LLM-generated counterfactual pairs) against simple baselines. Positive message: FGM alone is the reliable gain, up to the seed paper's level. Negative message with mechanism: both complex routes fail, and the pairs fail through shortcuts that a cheap audit (length, AI word model) catches before training. | Yes, if Passes 1-3 shrink to about 2 pages | Instructor may still want a "method"; answer: the audit is a reusable method for checking generated augmentation data |
| B | Keep the current Passes 1-3 paper; pairs go to the ERX journal version | Yes, as is | Ignores the instructor's request; ERX scope risk ([[Audit - Venues]]) |
| C | Pairs-first paper: "LLM-generated counterfactuals for sarcasm: a shortcut audit"; Passes 1-3 as background | Yes | Drops the ablation she accepted; a purely negative headline |
| D | A or C plus one more GPU pass for gain: the stacked FGM + self-training + helpers-on-L recipe ([[Research Ideas]] "After Pass 6", item 2) | Only if it fits before about 1 Dec | Roughly 30-40 runs; may also give nothing; adds a contrast family |

## Proposed outline for A (6 pages incl. references)
| Section | Content | Pages |
|---|---|---|
| Abstract, I Introduction | Two routes to extra gain, one controlled testbed, three findings | 0.75 |
| II Related work | GAN-BERT + MTL (seed paper), adversarial training, LLM data augmentation and counterfactuals, shortcut learning | 0.5 |
| III Setup | Data, shared encoder and step budget, arms A0-A19 in one table, paired bootstrap + Holm | 1.0 |
| IV Results 1: GAN, helpers, FGM | Main grid as differences from BERT; FGM across 1/10/100%; shuffled-label result in two sentences; outlet-word masking | 1.5 |
| V Results 2: LLM pairs | Generation in brief; in-domain cost; length and AI-word audit; transfer net of word count (`auc_vs_length`); two-stage trade-off figure (in-domain vs transfer, one point per arm) | 1.5 |
| VI Discussion, limitations, conclusion | Re-implementation, one dataset in-domain, no human check of generated text | 0.5 |
| References | | 0.25 |

Cuts to make room: the GAN-check details and the gold audit become one sentence each; the RQ2 forest plot goes; the label-quality table becomes a two-line summary.

## Questions for the instructor
1. Is a diagnostic paper with a clear positive (FGM) and an explained negative (LLM pairs) acceptable for QPAIN, instead of a new method?
2. Should the pair results share the paper with the ablation (A), or go to the journal version (B)?
3. Is one more GPU pass for a stacked recipe (D) worth the time before the 20 Dec deadline?
4. May the paper put A7's 93.90% next to the seed paper's 93.88% as a reference (different split), or should it leave the seed paper's number out?

## Timeline (proposal)
- By 17 Nov: owner runs notebook 14 (bootstrap); framing questions sent to the instructor.
- By 1 Dec: redraft `paper/main.tex` to the agreed framing (and option D's runs, if chosen, finished).
- By 10 Dec: framing and draft agreed with the instructor.
- By 18 Dec: final checks (references, similarity check, anonymity); submit by 20 Dec.

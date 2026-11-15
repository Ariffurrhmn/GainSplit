---
tags: [literature, novelty]
created: 2026-11-15
---

# Novelty Check - MLM Pretraining

Hub: [[00 - Project Context]]. Question (owner, 2026-11-15): has task-adaptive MLM pretraining for low-label sarcasm detection, or our controlled comparison, already been published? If so, Pass 10 is moot. Related: [[Pass 8 Results]], [[Experiment Plan]] "Pass 10 plan", [[Literature Index]].

Method: a delegated web and Consensus search (about 12 queries, mostly abstracts), then the key leads checked directly in this session (two extra searches, three pages read). Absence of evidence is not proof of absence; a Google Scholar pass is still worth doing before submission.

## Verdict
- **The method is not new** (VERIFIED, abstract): task-adaptive pretraining is Gururangan et al. 2020 ("Don't Stop Pretraining", ACL 2020).
- **TAPT beats self-training at low labels is not new** (VERIFIED, abstract): Shi, Tonolini, Aletras, Yilmaz, Kazai, Jiao, "Rethinking Semi-supervised Learning with Language Models", Findings of ACL 2023 (arXiv 2305.13002). TAPT vs five self-training methods on general text classification sets; TAPT is the more robust semi-supervised learner, and self-training is risky with few labels. Their appendix reportedly also runs self-training from TAPT checkpoints (UNVERIFIED, from a secondary summary): that is the general version of our planned A32.
- **TAPT on sarcasm, on the Headlines dataset, in a label sweep: not found** (INFERRED from the searches). No paper applies TAPT / continued MLM pretraining to the News Headlines sarcasm dataset, and no low-label (1-30%) study of that dataset was found.
- **Controlled comparison not found** (INFERRED): no paper puts TAPT beside GAN-BERT, FGM, multi-task helper heads, self-training, diffusion, LLM-generated pairs and JEPA in one testbed.
- **JEPA-style vs MLM pretraining for text classification: not found as a direct comparison** (UNVERIFIED, abstracts only); text JEPA variants exist (BERT-JEPA, LLM-JEPA arXiv 2509.14252) but they add a latent loss rather than test it against MLM on a classification task.

## Closest work
| Paper | Overlap | Difference | Read |
|---|---|---|---|
| Gururangan et al. 2020, ACL | TAPT, largest gains in low-resource settings | No sarcasm, no GAN/FGM/helper/self-training comparison | abstract |
| Shi et al. 2023, Findings of ACL | TAPT vs self-training at low labels; TAPT then self-training | No sarcasm; no GAN-BERT, FGM, helper heads, JEPA | abstract (VERIFIED), appendix via summary (UNVERIFIED) |
| Irfan et al. 2025, IEEE QPAIN (seed paper, [[Seed Paper - Irfan 2025]]) | Headlines, GAN-BERT + sentiment/emotion heads | 70% labeled; no label sweep, no TAPT | abstract |
| Savini & Caragea 2022, Mathematics 10(5):844 | Transfer before sarcasm fine-tuning (sentiment/emotion first) | Supervised intermediate tasks, not unlabeled MLM; not Headlines | abstract |
| Sosea, Li, Caragea 2023, arXiv 2308.08156 (HurricaneSARC) | Semi-supervised sarcasm; a search summary says MLM on an unlabeled sarcasm Reddit corpus helped | Disaster tweets; the abstract only confirms intermediate-task transfer (UNVERIFIED for the MLM part) | abstract |
| Croce et al. 2020, ACL (GAN-BERT) | Low-label semi-supervised baseline we use | Compared only with BERT | summary |
| Almutairi et al., ICNLSP 2026 (ResPL) | Low-label sarcasm with pseudo-labels | Arabic dialects, LLM chain-of-thought pseudo-labels; no TAPT, no Headlines | first page, full text extracted |

## Claims the paper must not make
- "First to use (domain- or task-adaptive) pretraining for sarcasm" (risky: HurricaneSARC, BERTweet-style domain models, Savini & Caragea).
- "TAPT beats self-training" or "TAPT then self-training" as new findings (Shi et al. 2023).
- Any comparison with the seed paper's 93.88% without saying it used 70% labels.

## What still looks new (INFERRED)
- The controlled, same-testbed comparison of every route to using unlabeled headlines (GAN, helpers, FGM, self-training, diffusion, LLM pairs, MLM, JEPA), with pre-registered predictions and paired bootstraps.
- The label-efficiency result on Headlines (MLM with 1% labels about equals BERT with 10%).
- Sarcasm-specific checks: cue-word shortcut gap and transfer to iSarcasmEval net of word count.
- JEPA vs MLM under equal compute (weak novelty, one pretraining seed).
- Whether helper heads, FGM and self-training add on top of TAPT for sarcasm (Pass 10 Q1); the self-training part has a general precedent in Shi et al.

## Reading (INFERRED)
Pass 10 is not moot: nobody has published this on sarcasm or this dataset, and the paper's contribution is the controlled comparison, not the method. The novelty claim must be framed as "a controlled study shows that the simplest, already-known use of unlabeled text (TAPT) beats every specialised semi-supervised route for low-label sarcasm detection", citing Gururangan 2020 and Shi 2023 as the general precedent our result agrees with. The +7.4 at 1% is in line with those papers' trend (gains largest at low labels), so it is not suspicious in itself.

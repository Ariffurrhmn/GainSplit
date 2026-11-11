---
tags: [literature, novelty]
created: 2026-11-11
updated: 2026-11-11
---

# Novelty Check - LLM Counterfactuals

Hub: [[00 - Project Context]]. Idea checked: [[Research Ideas]] "Generative AI and generalisation". Search 2026-11-11 (Consensus, 3 queries; web search for the Unfun data). Labels are from abstracts and search excerpts, except Horvitz 2024 (full PDF read). Second search the same day: section below.

## Closest prior work
| Paper | What it does | Overlap | Label |
|---|---|---|---|
| Jia et al. 2024, AAAI, "Debiasing Multimodal Sarcasm Detection with Contrastive Learning" | ChatGPT counterfactual rewrites against spurious text words; OOD test | High in idea; multimodal (image + text), not headlines | VERIFIED (abstract + excerpt) |
| Hong et al. 2025, Findings ACL, "Rhetorical Device-Aware Sarcasm Detection with Counterfactual Data Augmentation" (RedSD) | GPT-4o counterfactuals both ways ("duplex"), human-revised, sitcom dialogues | High in method; dialogue domain, no shortcut or cross-dataset focus seen | VERIFIED (abstract + excerpt) |
| Horvitz et al. 2024, ACL short, "Getting Serious about Humor" | GPT-4/3.5, Mistral unfun Onion headlines (Unfun corpus); classifiers trained on synthetic unfuns + original satire reach 76.5 (Mistral) vs 80.3 with human unfuns vs 66.3 with unrelated real news headlines as negatives; humor direction (making serious text funny) clearly worse | **Pre-empts** the 'counterfactual test for Headlines' idea: the 'real news as negatives' row is the News Headlines setup. Full data only; no low-label, no SSL, no sarcasm datasets, no cross-dataset transfer | VERIFIED (full PDF read 2026-11-11) |
| West and Horvitz 2019, AAAI, "Reverse-Engineering Satire" (Unfun.me) | Human minimal edits of Onion headlines into serious ones (~2,800 edits of ~1,200 headlines) | Human counterfactuals for our exact source; usable as a counterfactual test set. Data on GitHub `epfl-dlab/unfun` (availability not opened) | VERIFIED (abstract); data UNVERIFIED |
| Veselovsky et al. 2023, arXiv, "Generating Faithful Synthetic Data with LLMs" | Synthetic sarcasm data; grounding works best | Medium | VERIFIED (abstract) |
| Majer et al. 2026, WASSA, multi-aspect evaluation of synthetic irony/sarcasm data | LLM augmentation helps, synthetic items easy and low-diversity | Medium | VERIFIED (abstract) |
| Mahato et al. 2026, ACL, "LLMs in Sarcasm Detection? It's elementary! (Or is it?)" | Models trained on synthetic sarcasm learn generated-text shortcuts, collapse on human text | Risk for our plan | VERIFIED (abstract) |
| Jang and Frassinelli 2024, NAACL | Cross-dataset sarcasm: most models fail to generalise | Motivation, already cited | VERIFIED (abstract) |
| Haroon 2026, arXiv | Shortcut learning and cross-dataset collapse incl. sarcasm (TweetEval Irony vs Sarcasm Corpus V2) | Motivation; does not use Headlines | VERIFIED (abstract) |

## Verdict (INFERRED)
- "LLM counterfactual augmentation for sarcasm" is **not new** (Jia 2024, Hong 2025); "LLM unfunning of headlines" is **not new** (Horvitz 2024).
- Not seen in this search: counterfactuals aimed at a **measured** outlet shortcut in News Headlines, judged by the cue-masked test, the Unfun human counterfactual test, and cross-dataset transfer, in a **low-label** setting with a GAN-BERT baseline. That is a narrower, defensible gap, not a new technique.
- Must control for Mahato 2026: synthetic text can create its own shortcut. Needs a "generated vs real" check (e.g. a classifier that separates LLM text from real headlines) and evaluation only on human-written text.
- Not checked: Google Scholar / ACL Anthology beyond Consensus; full texts of Jia, Hong, Horvitz.

## Second search, 2026-11-11 (looking for an unblocked contribution)
Checked (abstracts unless stated):
- VAT/FGM for sarcasm: adversarial training for low-label sarcasm exists (Zhang et al. 2018, LNCS); FGM vs GAN-BERT done for Bengali (Zihan 2026). VAT for sarcasm not seen, but incremental. Not a headline contribution.
- LLM inside GAN-BERT: Forged-GAN-BERT (Silva et al. 2024, EACL SRW) adds ChatGPT text plus the GAN generator for authorship attribution. LLM-GAN (Wang et al. 2025, ICASSP) for fake news. Partial overlap.
- Outlet confound: McHardy et al. 2019 (NAACL) adversarially remove publication source in German satire. Not usable on Headlines (source = label).
- LLM pseudo-labels for SSL: LG-CoTrain (Rahman and Caragea 2025, EMNLP), LLKD (Li et al. 2025, Findings NAACL). Strong baseline to include, not sarcasm-specific.
- Unfun data: GitHub `epfl-dlab/unfun`, SQL dumps + `pairs_with_ratings.tsv` (original, edit, player seriousness rating). No licence stated; README asks for citation (VERIFIED, repo README). Usable as a human counterfactual test set.
- Using Headlines-trained models on Unfun pairs: largely shown by Horvitz 2024 (row "News Headlines").

## Candidate that survived (INFERRED, not seen in either search)
**Label-free paired generation from the unlabeled pool.** For each unlabeled headline, an open LLM writes *both* a sarcastic and a serious version on the same topic. Both sides are machine-written, so neither topic, outlet words nor "LLM style" can predict the label, and no pseudo-label is needed. Used as the semi-supervised signal in place of the GAN generator, inside the existing controlled study (1/5/10%, 5 seeds, paired bootstrap).
- Differs from Horvitz 2024 (one synthetic side, full data, humor), Hong 2025 (duplex on labeled dialogues, human revised), Jia 2024 (multimodal), LG-CoTrain (labels, not pairs), Mahato 2026 (motivates the both-sides design).
- Baselines: A1, A4 GAN-BERT, A6 self-training, A7 FGM, LLM pseudo-labels on U, one-sided pairs (Horvitz-style). Tests: Headlines test, cue-masked test, Unfun human pairs, iSarcasmEval.
- Risks: making serious text sarcastic is harder for LLMs (Horvitz Table 1/2); LLM pseudo-labels may beat pairs; generation time on Kaggle not measured; Unfun licence unstated.

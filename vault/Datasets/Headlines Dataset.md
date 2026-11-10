---
tags: [dataset]
updated: 2026-11-10
---

# Headlines Dataset

News Headlines Sarcasm (Misra and Arora). arXiv 1908.07414 (early version of AI Open 2023). Hub: [[00 - Project Context]] · Index: [[Datasets Index]].

- **Label = outlet** (VERIFIED): sarcastic items come only from The Onion's "News in Brief" and "News in Photos"; HuffPost is the non-sarcastic side. v1 counts 26,709 = 11,725 + 14,984. Misra abstract (arXiv 2212.06035): ~13K sarcastic of ~28K.
- Informal GitHub claim of URL/section leakage (single source, UNVERIFIED). No peer-reviewed headlines shortcut audit found.
- Cross-dataset failure evidence: Helal et al. (Sci. Rep. 2024, abstract only) 49% F1 on Reddit; Jang and Frassinelli (NAACL 2024).
- Correction: arXiv 2607.14131 uses TweetEval Irony / SC V2, not Headlines.
- Count mismatch in the seed paper: [[Open Questions]]. Ceiling ~92-94% accuracy ([[Risks]]).

## Local copy (VERIFIED 2026-11-08)
- Downloaded `Sarcasm_Headlines_Dataset.json` from the author's GitHub repo (rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection, master) to `data/headlines/` (git-ignored). SHA-256 `ab0faf86c2dbbef47a95080bc7601981f3ea9dc48ccca73382f90b9a56e02b49`, 6,057,046 bytes.
- JSON lines, fields `is_sarcastic`, `headline`, `article_link`. **28,619 rows, 13,634 sarcastic** (47.6%). That matches the later (v2) size, not the 26,709 of v1 quoted above; the file name has no `_v2`. No `_v2` file exists in that repo path (404).
- Contains `article_link` (outlet URLs): drop it before training; it is a direct label leak (see [[Risks]]).
- Licence still not checked from the repo.
- 2026-11-09: raw file re-checked unedited (SHA-256 matches above) and kept. Derived split files in `data/processed/` removed (moved to the Recycle Bin). The owner will clean and split the data personally ([[Repo Rules]] rule 5).

## Processed split (VERIFIED from `manifest.json`, run 2026-11-10)
- Made by `experiments/01_preprocess.ipynb` (interactive Kaggle session, owner ran it). Settings: punctuation removed, duplicates dropped, split seed 42, `bert-base-uncased`, MAX_LEN 48. Kaggle versions: Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, pandas 2.3.3, numpy 2.1.3, sklearn 1.6.1.
- Raw 28,619 / 13,634 sarcastic, SHA-256 matches the local copy. Cleaning changed 16,614 headlines (58%; punctuation replaced by a space, so "can't" becomes "can t", the same word pieces BERT gets from "can't" minus the apostrophe). Empty removed 0; extra duplicate copies removed 118; conflicting-label duplicates 0. **Kept 28,501, 13,550 sarcastic (47.5%).**
- Token lengths (bert-base-uncased, incl. [CLS]/[SEP]): median 13, p90 18, p99 24, max 177. One headline exceeds 48 and is truncated.
- **Split:** train 22,800 (10,840 sarc), dev 2,850 (1,355), test 2,851 (1,355); sarcastic share 47.5% in all three. No id or text overlap between parts (asserted in the notebook).
- **Label budgets** (same sizes for seeds 0-4, contents differ, nested): 1% = 228, 5% = 1,140, 10% = 2,280, 30% = 6,840, 100% = 22,800.
- **Helper labels (all 28,501):** sentiment neutral 16,046 (56%), negative 9,316 (33%), positive 3,139 (11%). Emotion (5-class mapping in [[Decisions]]) neutral_other 23,576 (83%), joy 2,411, sadness 1,357, anger 818, fear 339. See [[Risks]] item 11.
- Sentiment spot-check (top 5 per class, read by the owner): label order correct. Most-negative are profane Onion-style headlines; most-positive include a sarcastic one ("excited cia director ..."), i.e. the teacher reads surface sentiment, as expected.
- Output files and SHA-256 hashes: `data/processed/manifest.json` (local copy, sizes match). Uploaded to Kaggle as the private dataset `gainsplit-processed` by the owner (2026-11-10). The raw dataset mounted at `/kaggle/input/datasets/arifurrhmn/task-a-en/`, so training code should locate `manifest.json` under `/kaggle/input` rather than hard-code the path.

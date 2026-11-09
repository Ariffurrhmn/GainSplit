---
tags: [dataset]
updated: 2026-11-09
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

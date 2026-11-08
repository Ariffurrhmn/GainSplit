---
tags: [dataset]
updated: 2026-11-08
---

# Headlines Dataset

News Headlines Sarcasm (Misra and Arora). arXiv 1908.07414 (early version of AI Open 2023). Hub: [[00 - Project Context]] · Index: [[Datasets Index]].

- **Label = outlet** (VERIFIED): sarcastic items come only from The Onion's "News in Brief" and "News in Photos"; HuffPost is the non-sarcastic side. v1 counts 26,709 = 11,725 + 14,984. Misra abstract (arXiv 2212.06035): ~13K sarcastic of ~28K.
- Informal GitHub claim of URL/section leakage (single source, UNVERIFIED). No peer-reviewed headlines shortcut audit found.
- Cross-dataset failure evidence: Helal et al. (Sci. Rep. 2024, abstract only) 49% F1 on Reddit; Jang and Frassinelli (NAACL 2024).
- Correction: arXiv 2607.14131 uses TweetEval Irony / SC V2, not Headlines.
- Count mismatch in the seed paper: [[Open Questions]]. Ceiling ~92-94% accuracy ([[Risks]]).

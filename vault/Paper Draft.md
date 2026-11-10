---
tags: [paper]
created: 2026-11-11
updated: 2026-11-11
---

# Paper Draft

Hub: [[00 - Project Context]]. Venue rules: [[Audit - Venues]]. Results it reports: [[Bootstrap - All Passes]], [[Pass 3 Results]], [[GAN Check Results]], [[Gold Audit Results]].

First full draft written 2026-11-11 for the instructor. Files in `paper/`:
- `main.tex` (IEEEtran conference, anonymous author block), `refs.bib`.
- `figures/arch.tex` (TikZ model diagram, Fig. 1); `figures/fig_rq1.pdf`, `fig_rq2.pdf`, `fig_rq3.pdf` (Figs. 2-4), made by `figures/make_figures.py` from `results/analysis/*.csv`.
- `tables/main_results.tex`, `tables/gold.tex`, made by `tables/make_tables.py` from the same CSVs.
- `gainsplit_overleaf.zip`: the eight files Overleaf needs (upload as a new project; compiler pdfLaTeX).

Status: compiles (checked locally with tectonic; local fonts differ from Overleaf's Times) to **6 pages including references**, at the 4-6 page limit. Every number in the text was checked against the analysis CSVs or the vault notes on 2026-11-11.

Structure: abstract; I Introduction (3 RQs, contributions); II Related work; III Design (data, arms A0-A9, training and statistics); IV Results (RQ1 with the GAN check, RQ2 with the gold audit, RQ3 masking and transfer); V Discussion with limitations; VI Conclusion. Figures: model diagram, RQ1 two-panel (budget curve + all arms at 1/10%), RQ2 forest plot of label variants, RQ3 two-panel (masking dumbbells + transfer AUC). Tables: main macro-F1 grid, gold agreement.

Open items before submission:
- **Seed-paper reference** (`irfan2025seed` in `refs.bib`): full title and author list are missing (not in the vault, not found online 2026-11-11). Must be filled before the draft goes out; the PDF shows "CHECK: ..." until then.
- Bibliography details not read from the sources (UNVERIFIED): first names for Chen 2022 and Zihan 2026, page numbers for Jang and Frassinelli 2024; standard references (BERT, GAN-BERT, FGM, VADER, Holm, etc.) entered from common knowledge, not re-checked. Marked "CHECK" in comments in `refs.bib`.
- iSarcasmEval is cited as the SemEval-2022 task paper; [[iSarcasmEval]] says "Findings EMNLP 2022" (conflict, UNVERIFIED).
- Similarity check (<= 20%) not run.
- Instructor feedback.

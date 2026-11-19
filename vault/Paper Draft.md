---
tags: [paper]
created: 2026-11-11
updated: 2026-11-16
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

## Version 2 (2026-11-15): "Just Keep Pretraining"
Owner chose the title "Just Keep Pretraining: Unlabeled Headlines Beat GANs, Helper Tasks and LLM Data for Low-Label Sarcasm Detection" and asked for a new draft beside the first one (the first draft in `paper/` is unchanged).

Files in `paper/v2/`: `main.tex` (IEEEtran, anonymous), `refs.bib` (copy of the first draft's, plus Gururangan 2020, Shi 2023, I-JEPA, DDPM, UDA; the seed paper now has title and DOI), `figures/make_figures.py` (reads only committed result tables), `figures/fig_methods.pdf` (gain over BERT for ten routes at 1% and 10%), `figures/fig_curve.pdf` (label curve, BERT vs MLM, with the two extra pretraining seeds), `draft_v2_preview.pdf`, `gainsplit_v2_overleaf.zip` (four files; upload as a new Overleaf project, pdfLaTeX).

Status: compiles with tectonic to **5 pages including references**, no overfull boxes, no BibTeX errors. Numbers checked against the committed tables 2026-11-15 ([[Pass 10 Results]], [[Pass 8 Results]], [[Pass 9 Results]], [[Pass 4 Results]], [[Bootstrap - All Passes]], [[Pass 3 Results]]).

Structure: abstract; I Introduction (Q1-Q3, three contributions); II Related work (semi-supervised, TAPT and Shi 2023, affect and sarcasm, shortcuts); III Design (data, ten routes, training and statistics, pre-registered predictions); IV Results: Q1 routes (Table I, Fig. 1), Q2 stability and label efficiency (Fig. 2), stacking on the pretrained encoder (Table II), why helper tasks barely help (incl. the polarity-flip check), Q3 outlet words and transfer; V Discussion with limitations; VI Conclusion.

Open items:
- **Polarity-flip check not run**: the paragraph shows a red PENDING marker (helper-head arms vs A1/A20 on the 107 sarcastic test headlines with positive teacher sentiment; CPU analysis on existing predictions, exploratory). From [[Proposal B]] / PolarityTax, see Session Log 2026-11-15.
- Bibliography items marked CHECK in `refs.bib` (entered from memory or secondary sources), incl. the seed paper's authors and its 70% labeled share (comment in `main.tex`).
- The 5-page length leaves room for a model figure or a JEPA/MLM pretraining table if the instructor wants one.
- Similarity check not run; instructor feedback pending.

### Deep Review Received (2026-11-16)
Full review text: [[Review - Draft v2]] (and `paper/v2/review.md`).
- **Verdict / Summary**: Validates paper rigor, controlled ablation of 10 routes, mechanistic deconstruction of affect helper heads, and exemplary claim calibration / diagnostic bounding. Confirmed fully double-blind compliant; all 25 citations verified (0 hallucinated).
- **Key action items identified**:
  1. *Title & claim calibration*: Consider narrowing the title slightly (e.g. *"...for Low-Label Headline Sarcasm Detection"*) or explicitly stating in Intro/Limitations that *News Headlines* was chosen specifically to audit published semi-supervised claims.
  2. *Data exposure & step budget asymmetry*: Note that single-stage semi-supervised baselines (GAN-BERT, consistency) see <28.4% of 1 pass over $U$ at 1% labels under the 200-step budget, whereas MLM/JEPA receive 10k dedicated pretraining steps (~28 passes); frame as two-stage pretraining beating fixed-step joint fine-tuning.
  3. *Helper heads on $L$ vs $U$*: Clarify whether A5 applies helper losses to $U$ as well as $L$, and explain why A2 was restricted to $L$ rather than $L \cup U$ when teacher pseudo-labels are cost-free.
  4. *TAPT + Self-Training positioning*: Contrast the absorption finding with Li et al. (EMNLP 2021) complementarity, noting that on this closed pool TAPT already pushes teacher accuracy to 92-96%. Distinguish latent diffusion consistency from text-level UDA.
  5. *Statistical protocol details*: State family sizes $m$ for Holm's correction; report s.d. in Table II; clarify warm-start vs cold-start head initialization and seed parity (3 seeds for MLM vs 1 for JEPA).
  6. *Bibliography polish*: Fix author initials for Irfan [3] and Zihan [16], LaTeX diacritic in Marasović [4], duplicate year in [5], missing page numbers.
  7. *Polarity-flip analysis guidance*: When resolving the PENDING placeholder in IV-D, use a difference-in-differences interaction test to separate polarity sensitivity from overall recall differences.

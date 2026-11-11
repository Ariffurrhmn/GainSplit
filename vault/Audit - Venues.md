---
tags: [venues]
updated: 2026-11-11
---

# Audit - Venues

Hub: [[00 - Project Context]]. All dates and formats UNVERIFIED unless marked.

**Primary:** IEEE QPAIN 2027 (stated by the owner, 2026-11-08). The seed paper appeared at QPAIN 2025 ([[Seed Paper - Irfan 2025]]).

### QPAIN 2027: checked on the live site (2026-11-11, VERIFIED)
Read from https://qpain.org/, /call-for-papers and /submission-guidelines on 2026-11-11:
- 2027 IEEE 3rd Intl. Conf. on Quantum Photonics, Artificial Intelligence & Networking, Chattogram, Bangladesh, 31 Mar - 3 Apr 2027.
- **Paper deadline 20 Dec 2026**; notification 5 Feb 2027; camera-ready and registration 20 Feb 2027. (The owner's planning date of 10 Dec stays as a buffer.)
- **4-6 pages**, PDF, English, IEEE conference template; "extra pages may incur additional fees per IEEE policy". Whether references count is not stated.
- **Double-blind:** "remove author names, affiliations, and any identifying information in the initial submission". Self-citation, acknowledgements and code links are not addressed, so the draft has no repo link.
- Similarity <= 20% overall (references included), <= 4% from a single source.
- No AI/LLM-tool policy on the guidelines page. Portal: PROCONF (myproconf.com, event qpain-2027).

### Engineering Research Express (IOP), checked 2026-11-11 (VERIFIED)
Proposed by the instructor as a journal target, with a 9-page instruction (the instructor's rule, not the journal's). Read on 2026-11-11 from publishingsupport.iopscience.iop.org: the ERX author guidelines and "About" pages, the IOP ethical policy and the research data policy. The iopscience.iop.org "About" page sat behind a bot check and was not read; the publishing-support copy covers the same ground.
- **Scope risk (main issue):** the scope note says AI/ML and data-driven submissions "must demonstrate a clear and substantive contribution to engineering" by addressing a well-defined engineering problem, system, process or application; those without a strong engineering focus "will not be considered". Listed areas are electrical, mechanical, civil, chemical, autonomous/intelligent systems, manufacturing. A sarcasm-detection NLP study has no obvious engineering application, so desk rejection on scope is likely unless the paper is reframed (INFERRED).
- In our favour: review judges rigour, not novelty or impact; null and negative results and replication studies are explicitly in scope.
- Article type Research Paper, **unlimited length** (no page limit). Any layout is accepted for submission (single PDF); IOP LaTeX/Word templates optional; at least 12-pt font and reasonable line spacing; figures and tables inside the text; line numbers added automatically.
- Peer review: single- or double-anonymous, **author's choice**. If double: anonymise, keep the ethics statement and conflicts out of the PDF (they go in the submission form / cover letter).
- Abstract at most 300 words; no undefined acronyms, references, figure or table numbers in it. Structure: Introduction, Methods, Results, Discussion, Conclusion, Acknowledgements. Methods must allow reproduction.
- Tables must not use colour (bold/italics instead). Figures: do not rely on colour alone; EPS or PDF preferred; captions self-contained without acronyms; text 8-12 pt at final size.
- References: Vancouver numbered or Harvard; prefer DOIs. **Every reference must be checked against the original source; references to non-existent work usually lead to rejection.**
- **Generative-AI policy:** where such tools materially contributed (research process, code, analysis, text), a separate AI disclosure statement in the Acknowledgements is required (tool, version, use, how the outputs were checked); research use also described in Methods. Routine grammar editing needs no statement. Unacceptable: generating or modifying reference lists without checking each source, generating responses to reviewers.
- **Data availability statement required** (chosen at submission). Public sharing encouraged, not required; if not shared, a reason is published and data must be available to editors and reviewers on request. Code on GitHub: deposit a DOI-issuing copy (e.g. Zenodo) and cite it.
- No parallel submission: the paper may not be under consideration elsewhere. A journal version of conference work needs a new contribution, cited prior version, and a prior word with the journal staff.
- Authorship is frozen at submission (no changes during review or proofs); all authors must meet the four ICMJE criteria. ORCID recommended; contribution statement (CRediT) encouraged.
- Ethics statement needed for research with human participants (the gold audit has one annotator, the owner; whether that counts as human-participant research is for the owner and instructor, INFERRED probably not).
- Costs: subscription publication free; gold open access APC GBP 1970 / USD 2645 (discounts for many lower and middle income countries, applied automatically). No submission fee.
- Indexed in Scopus, Web of Science (ESCI), EI Compendex, Inspec.

### QPAIN 2027 facts (2026-11-08, superseded by the check above)
Source: the owner's venue-instructions notes (not included in this repo). Their facts were second-hand as of 2026-11-05 and never checked against the live site (https://qpain.org/). All **UNVERIFIED** until someone reads the live pages.
- Name: 2027 IEEE 3rd Intl. Conf. on Quantum Photonics, AI and Networking, Chattogram, Bangladesh. Dates 31 Mar - 3 Apr 2027 (one page says 8-10 Apr: conflict).
- **Deadline conflict:** owner says 10 Dec 2026; the repo says 20 Dec 2026 (notification 5 Feb, camera-ready 20 Feb 2027). **Decision 2026-11-08: owner plans to 10 Dec 2026.** Still check the live site; if 20 Dec is right it only adds slack.
- Length: owner says 6 pages max; repo says 4-6 pages, extra pages may cost fees (INFERRED: references count). PDF only, English, double-blind (both sources agree).
- Similarity <= 20% overall, <= 4% single source (includes references). Original, not under review elsewhere.
- Submission via PROCONF/MyProConf (portal possibly not open yet). Template: IEEEtran conference zip, anonymised copy in the repo. Format of our paper: not decided yet (owner, 2026-11-08).
- Track fit: Track 03 AI and ML lists "Natural Language Processing and Understanding". Venue is photonics-led, so NLP-only benchmark work is a weaker fit (INFERRED).
- Unknowns: timezone, LLM policy, checklists, arXiv/code-link rules, review criteria, awards. Resolve via https://qpain.org/contact.
- Double-blind consequence: nothing in `paper/` may name the owner, institution, repo or accounts; no code link until acceptance.

Fallbacks (from [[Deep Dive - Paper A]] (e)):
| Venue | Verified | Fit |
|---|---|---|
| Insights from Negative Results in NLP workshop | 2025: 23 + 2 ARR submissions, 16 accepted; scope includes reproducibility and failure analysis. 2026/27 call not found | Best for null-result framing |
| WASSA | EACL 2026 edition was the 15th; 8pp long / 4pp short; 2027 call unverified | Good |
| ACL Student Research Workshop | 2026: first author must be a student; 2027 call not posted | Good for an MSc first author |
| Journals: Sci. Reports, IEEE Access, Inf. Sciences, Heliyon, ESWA | Published comparable work; reviewer demands unverified | Realistic for a full 2-dataset study |

Reviewer expectations (INFERRED): RoBERTa/DeBERTa baselines, 5+ seeds mean +- std, McNemar/bootstrap, full ablation, second dataset, released code. See [[Experiment Plan]].

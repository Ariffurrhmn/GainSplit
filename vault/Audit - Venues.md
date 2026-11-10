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

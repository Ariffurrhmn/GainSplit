---
tags: [results, rq2]
created: 2026-11-10
updated: 2026-11-10
---

# Gold Audit Results

Hub: [[00 - Project Context]]. Protocol: [[Decisions]] (gold audit, 2026-11-10). Limitation: [[Risks]] item 12.

The owner labeled the 200 sampled training headlines for surface sentiment in `experiments/gold_audit.html` (seed 20261110, sample fingerprint 6139c5bf). Export: `results/gold_audit/gold_sentiment_audit.csv` (kept local, not in git: it holds headline text; [[Decisions]]). Teacher labels: `sent_label` in `data/processed/helper_labels.csv`.

**Status: PROVISIONAL.** The numbers are VERIFIED as computed from the two files. Whether they measure teacher error is UNVERIFIED until the owner confirms how the "unsure" flag was used (see below).

## Checks (VERIFIED)
- 200 rows, 200 distinct ids, all in the training split, no missing labels; the text in the export matches `headlines_train.csv` for every id; one seed and one fingerprint.

## Teacher vs owner (surface sentiment, 3 classes)
| Subset | n | Accuracy | Macro-F1 | Cohen's kappa |
|---|---|---|---|---|
| All | 200 | 0.530 (95% bootstrap 0.46-0.60) | 0.491 | 0.237 |
| Not unsure | 104 | 0.481 | 0.463 | 0.254 |
| Non-sarcastic headlines | 97 | 0.557 | 0.541 | 0.295 |
| Sarcastic headlines | 103 | 0.505 | 0.437 | 0.180 |

Per-class F1 (all 200): negative 0.58, neutral 0.56, positive 0.33.

Confusion (rows = owner, columns = teacher):
| | negative | neutral | positive |
|---|---|---|---|
| negative | 38 | 25 | 0 |
| neutral | 24 | 57 | 10 |
| positive | 6 | 29 | 11 |

The teacher almost never says "positive" (21 of 200, owner 46) and calls many of the owner's positives neutral (29/46).

## Problem with the "unsure" flag
- 96 of 200 items are flagged unsure (48%). The rate climbs through the sample: 22%, 34%, 68%, 66%, **100%** in consecutive blocks of 50. The app stores the flag per item (checked in the code), so it is not a stuck button.
- Unsure items are mostly "neutral" (70/96); sure items are mostly negative or positive (83/104).
- Agreement is *lower* on sure items (0.48) than overall (0.53), the opposite of what a reliable unsure flag gives.
- Reading (INFERRED): the flag changed meaning during labeling (fatigue, or "neutral" used as a fallback for hard items). Until clarified, the "sure" subset cannot be used as the clean gold set.

## What it would mean (INFERRED, if the labels stand)
The sentiment teacher agrees with a human on surface sentiment only about half the time (kappa 0.24, "fair"), and less on sarcastic headlines. So the helper labels in A3 are noisy, yet the label content still matters at 10% (A3 - A8 = +0.57, [[Pass 2a Results]]). This motivates the RQ2 rows (soft labels, teacher swap, synthetic noise) in Pass 3.

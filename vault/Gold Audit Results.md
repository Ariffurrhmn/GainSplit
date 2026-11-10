---
tags: [results, rq2]
created: 2026-11-10
updated: 2026-11-10
---

# Gold Audit Results

Hub: [[00 - Project Context]]. Protocol: [[Decisions]] (gold audit, 2026-11-10). Limitation: [[Risks]] item 12.

The owner labeled the 200 sampled training headlines for surface sentiment in `experiments/gold_audit.html` (seed 20261110, sample fingerprint 6139c5bf). Export: `results/gold_audit/gold_sentiment_audit.csv` (kept local, not in git: it holds headline text; [[Decisions]]). Teacher labels: `sent_label` in `data/processed/helper_labels.csv`.

**Status:** numbers VERIFIED as computed from the two files. Labels accepted as they are (owner confirmed the unsure flag is literal). Main figure for the paper: accuracy 0.53, kappa 0.24 on all 200; the unsure rate and the first-50 warm-up are reported with it.

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

## The "unsure" flag (owner's answer, 2026-11-10)
- The owner used "unsure" literally: not sure of the surface sentiment. So the flag is honest and every label stands. 96 of 200 items (48%) are flagged.
- The unsure rate climbs through the sample (blocks of 50: 22%, 34%, 70%, 66%), and the first block is an outlier in agreement. VERIFIED from the export:

| Items | Teacher accuracy | Unsure | Owner "neutral" |
|---|---|---|---|
| 1-50 | 0.30 | 0.22 | 0.38 |
| 51-100 | 0.64 | 0.34 | 0.40 |
| 101-150 | 0.62 | 0.70 | 0.40 |
| 151-200 | 0.56 | 0.66 | 0.64 |

- Agreement is 0.58 on unsure items and 0.48 on sure items.
- Reading (INFERRED): a warm-up effect. The owner's reading of "surface sentiment" settled after about 50 items, and grew more cautious after that (more unsure, more neutral). Surface sentiment of satirical headlines is hard for a human too: about half the items felt uncertain.

## What it means (INFERRED)
The sentiment teacher agrees with a human on surface sentiment only about half the time (kappa 0.24, "fair"), and less on sarcastic headlines. So the helper labels in A3 are noisy, yet the label content still matters at 10% (A3 - A8 = +0.57, [[Pass 2a Results]]). This motivates the RQ2 rows (soft labels, teacher swap, synthetic noise) in Pass 3.
- Second point for the paper: the human labeler was unsure on 48% of items, so a single hard sentiment label per headline is a weak target in itself. That supports testing soft labels in Pass 3 (RQ2).
- Re-label of items 1-50: declined by the owner (2026-11-10, [[Decisions]]). The warm-up block is reported as a limitation.

## Other sentiment sources vs the owner (2026-11-10, VERIFIED)
From `helper_sent_alt.csv` (made by `01b_pass3_data`, commit 654de15).

| Source | All 200: acc / macro-F1 / kappa | Sure (104): acc / macro-F1 / kappa |
|---|---|---|
| Teacher 1 (cardiffnlp, used in training) | 0.53 / 0.49 / 0.24 | 0.48 / 0.46 / 0.25 |
| VADER lexicon | 0.51 / 0.50 / 0.26 | 0.59 / 0.57 / 0.37 |
| Teacher 2 (j-hartmann RoBERTa-large) | 0.48 / 0.43 / 0.14 | 0.38 / 0.37 / 0.13 |

Over all 28,501 headlines, teacher 2 agrees with teacher 1 on 76% and VADER on 55%. Teacher 2 is mostly neutral (66%, positive only 6%); VADER is near-balanced. Reading (INFERRED): the three sources give the RQ2 rows a spread in label quality (VADER best on the sure items, teacher 2 worst), which is what the label-quality figure needs.

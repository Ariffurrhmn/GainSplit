| hypothesis | check | value | verdict |
|---|---|---|---|
| H11.24 | A5@70: accuracy on all rows minus test accuracy >= 2.0 | 3.97 | SUPPORTED |
| H11.25 | A5@70: accuracy on all rows >= 93.88 (the seed paper's figure) | 96.48 | SUPPORTED |
| H11.26 | A1@100 (their BERT setting): accuracy on all rows (their BERT: 92.04), descriptive | 98.62 | descriptive |
| H11.28 | A5@70: U accuracy within 0.7 of test, L accuracy at least 2.0 above test | U 91.76, test 92.51, L 100.00 | not supported |
| H11.28 | A1@70: U accuracy within 0.7 of test, L accuracy at least 2.0 above test | U 92.39, test 92.76, L 100.00 | SUPPORTED |
| H11.27 | all 16 re-runs reproduce the originals' test probabilities exactly | 16 of 16 PASS | SUPPORTED |

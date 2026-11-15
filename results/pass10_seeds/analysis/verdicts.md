| hypothesis | contrast | budget | mean_diff | ci_low | ci_high | p_holm | verdict | claim |
|---|---|---|---|---|---|---|---|---|
| H10.6 | A20[pt1]-A1 (pretraining seed 1) | 1 | 7.720 | 6.820 | 8.630 | 0.000 | SUPPORTED | pretraining seed 1 helps at 1% |
| H10.6 | A20[pt1]-A1 (pretraining seed 1) | 10 | 2.680 | 1.970 | 3.420 | 0.000 | SUPPORTED | ... and at 10% |
| H10.6 | A20[pt2]-A1 (pretraining seed 2) | 1 | 6.890 | 5.970 | 7.820 | 0.000 | SUPPORTED | pretraining seed 2 helps at 1% |
| H10.6 | A20[pt2]-A1 (pretraining seed 2) | 10 | 2.260 | 1.550 | 2.970 | 0.000 | SUPPORTED | ... and at 10% |
| H10.7 | A20-A1 (MLM vs BERT) | 5 | 3.490 | 2.760 | 4.220 | 0.000 | SUPPORTED | MLM helps at 5% |
| H10.7 | A20-A1 (MLM vs BERT) | 30 | 1.040 | 0.500 | 1.610 | 0.000 | SUPPORTED | MLM helps at 30% |
| H10.7 | A20-A1 (MLM vs BERT) | 100 | 0.090 | -0.420 | 0.600 | 1.000 | SUPPORTED (no significant difference) | no difference with every label |
| H10.8 | A20@1% - A1@10% | 1 | -0.380 | -1.150 | 0.410 | 1.000 | SUPPORTED (no significant difference) | MLM with 1% labels matches BERT with 10% |
| H10.8 | A20@10% - A1@30% | 10 | 0.170 | -0.440 | 0.800 | 1.000 | SUPPORTED (no significant difference) | MLM with 10% labels matches BERT with 30% |

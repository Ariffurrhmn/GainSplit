| hypothesis | contrast | budget | mean_diff | ci_low | ci_high | p_holm | verdict | claim |
|---|---|---|---|---|---|---|---|---|
| H11.18 | A36-A5 (our recipe vs the seed recipe) | 1 | 10.680 | 9.580 | 11.840 | 0.000 | SUPPORTED | our recipe beats the seed recipe at 1% |
| H11.18 | A36-A5 (our recipe vs the seed recipe) | 10 | 3.150 | 2.370 | 3.930 | 0.000 | SUPPORTED | our recipe beats the seed recipe at 10% |
| H11.19 | A36-A5 (our recipe vs the seed recipe) | 5 | 3.990 | 3.140 | 4.820 | 0.000 | SUPPORTED | ... and at 5% |
| H11.19 | A36-A5 (our recipe vs the seed recipe) | 30 | 1.640 | 1.010 | 2.260 | 0.000 | SUPPORTED | ... and at 30% |
| H11.20 | A5-A1 (at 70% labels) | 70 | -0.250 | -0.610 | 0.110 | 0.357 | SUPPORTED (not better) | the seed recipe is not better than A1 at 70% |
| H11.20 | A5-A7 (at 70% labels) | 70 | -0.890 | -1.380 | -0.420 | 0.002 | SUPPORTED (not better) | the seed recipe is not better than A7 at 70% |
| H11.20 | A5-A20 (at 70% labels) | 70 | -0.660 | -1.180 | -0.140 | 0.054 | SUPPORTED (not better) | the seed recipe is not better than A20 at 70% |
| H11.21 | A5-A1 (seed recipe vs BERT) | 1 | -2.300 | -2.990 | -1.630 | 0.000 | SUPPORTED (not better) | the seed recipe does not beat BERT at 1% |
| H11.21 | A5-A1 (seed recipe vs BERT) | 5 | 0.000 | -0.450 | 0.460 | 1.000 | SUPPORTED (not better) | the seed recipe does not beat BERT at 5% |
| H11.21 | A5-A1 (seed recipe vs BERT) | 10 | 0.230 | -0.190 | 0.650 | 1.000 | SUPPORTED (not better) | the seed recipe does not beat BERT at 10% |
| H11.21 | A5-A1 (seed recipe vs BERT) | 30 | -0.080 | -0.460 | 0.300 | 1.000 | SUPPORTED (not better) | the seed recipe does not beat BERT at 30% |
| H11.21 | A5-A1 (seed recipe vs BERT) | 70 | -0.250 | -0.610 | 0.110 | 1.000 | SUPPORTED (not better) | the seed recipe does not beat BERT at 70% |
| H11.16 | A4[long35]-A1[long35] (GAN vs BERT at 35x steps) | 1 | -7.920 | -8.920 | -6.940 | 0.000 | not supported (significant difference) | the GAN does not beat BERT at 35x steps (about 10 passes over U) |
| H11.22 | A37-A20 (seed recipe on the MLM encoder vs MLM alone) | 1 | -1.200 | -1.700 | -0.690 | 0.000 | SUPPORTED (not better) | the seed recipe adds nothing on the MLM encoder at 1% |
| H11.22 | A37-A20 (seed recipe on the MLM encoder vs MLM alone) | 10 | -0.400 | -0.820 | 0.010 | 0.059 | SUPPORTED (not better) | the seed recipe adds nothing on the MLM encoder at 10% |

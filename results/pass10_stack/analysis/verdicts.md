| hypothesis | contrast | budget | mean_diff | ci_low | ci_high | p_holm | verdict | claim |
|---|---|---|---|---|---|---|---|---|
| H10.1 | A32-A20 (self-training on top of MLM) | 1 | 0.350 | -0.120 | 0.820 | 1.000 | not supported (no significant difference) | self-training adds on top of MLM at 1% |
| H10.1 | A32-A20 (self-training on top of MLM) | 10 | 0.030 | -0.340 | 0.410 | 1.000 | no significant difference | ... at 10% (no direction) |
| H10.2 | A31-A20 (helpers on top of MLM) | 1 | 0.480 | 0.120 | 0.850 | 0.089 | SUPPORTED (no significant difference) | helpers add nothing on top of MLM at 1% |
| H10.2 | A31-A20 (helpers on top of MLM) | 10 | 0.070 | -0.280 | 0.420 | 1.000 | SUPPORTED (no significant difference) | ... nor at 10% |
| H10.2 | A21-A20 (FGM on top of MLM) | 1 | -0.090 | -0.530 | 0.340 | 1.000 | SUPPORTED (no significant difference) | FGM adds nothing on top of MLM at 1% |
| H10.2 | A21-A20 (FGM on top of MLM) | 10 | 0.140 | -0.290 | 0.550 | 1.000 | SUPPORTED (no significant difference) | ... nor at 10% |
| H10.3 | A36-A20 (all three on top of MLM) | 1 | 0.970 | 0.390 | 1.540 | 0.023 | SUPPORTED | the full stack beats MLM alone at 1% |
| H10.3 | A36-A20 (all three on top of MLM) | 10 | 0.580 | 0.090 | 1.060 | 0.202 | no significant difference | ... at 10% (no direction) |
| H10.4 | pretraining x helpers | 1 | -1.150 | -1.740 | -0.540 | 0.000 | SUPPORTED | pretraining shrinks the helpers' gain at 1% |
| H10.4 | pretraining x FGM | 1 | -1.670 | -2.310 | -1.020 | 0.000 | SUPPORTED | pretraining shrinks FGM's gain at 1% |
| H10.4 | pretraining x self-training | 1 | -1.260 | -1.880 | -0.650 | 0.002 | significant difference | pretraining and self-training (no direction) |
| H10.5 | A36-A30 (MLM under the full stack) | 1 | 4.210 | 3.450 | 4.990 | 0.000 | SUPPORTED | MLM still adds under the full stack at 1% |
| H10.5 | A36-A30 (MLM under the full stack) | 10 | 1.710 | 1.120 | 2.320 | 0.000 | SUPPORTED | ... and at 10% |

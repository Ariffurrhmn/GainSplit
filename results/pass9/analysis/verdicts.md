| hypothesis | contrast | budget | mean_diff | ci_low | ci_high | p_holm | verdict | claim |
|---|---|---|---|---|---|---|---|---|
| H9.1 | A30-A1 (whole stack vs BERT) | 1 | 4.170 | 3.440 | 4.890 | 0.000 | SUPPORTED | the stack beats plain BERT at 1% |
| H9.1 | A30-A1 (whole stack vs BERT) | 10 | 1.670 | 1.070 | 2.250 | 0.000 | SUPPORTED | ... and at 10% |
| H9.2 | A30-A7 (whole stack vs FGM alone) | 1 | 2.590 | 2.070 | 3.110 | 0.000 | SUPPORTED | the stack beats the best single ingredient at 1% (the 'more gain' claim) |
| H9.2 | A30-A7 (whole stack vs FGM alone) | 10 | 0.130 | -0.290 | 0.540 | 0.548 | SUPPORTED (no significant difference) | no difference at 10% |
| H9.3 | helpers x FGM | 1 | -0.660 | -1.290 | -0.040 | 0.269 | not supported (no significant difference) | helpers and FGM overlap at 1% (both regularisers) |
| H9.3 | all three vs sum of single gains | 1 | -0.650 | -1.560 | 0.230 | 0.632 | not supported (no significant difference) | the stack is below the sum of the three single gains at 1% |
| H9.4 | helpers x self-training | 1 | -0.610 | -1.150 | -0.050 | 0.269 | no significant difference | helpers and self-training (no direction) |
| H9.4 | FGM x self-training | 1 | 0.240 | -0.330 | 0.830 | 1.000 | no significant difference | FGM and self-training (no direction) |
| H9.5 | A27-A7 (helpers on top of FGM) | 100 | n/a | n/a | n/a | n/a | no runs (skipped) | no difference with every label |

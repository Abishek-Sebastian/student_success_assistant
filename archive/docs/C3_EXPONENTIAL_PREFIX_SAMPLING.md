# C3: Exponentially weighted prefix sampling

## Executive summary

- Fixed evaluation: the same penultimate-only C1 cohort and the same locked 20% student test split (outer seed 42) are used for every condition.
- Fixed model: `xgb_d2_group_weight` with the existing compact + per-lesson feature set (`compact_plus_module`); this isolates the training-prefix strategy.
- Baseline trains on each development student's penultimate prefix. C3 samples exactly one earlier-or-penultimate prefix per training student with probability proportional to exp(-lambda * distance from penultimate).
- Validation and test always use penultimate-prefix rows. All splits are by student; no student contributes to both training and validation/test.
- The supplied eventual dropout label is used for every sampled prefix. This is not a next-attempt target, and no dropout timestamp exists.
- Decision: the penultimate-only baseline had OOF AP 0.5077; the best C3 variant (c3_exp_lambda_0.5, lambda=0.5) had OOF AP 0.5064 (delta -0.0013).
- On the locked test, baseline AP was 0.9160 and best-C3 AP was 0.9113 (delta -0.0048). C3 does not improve the ranking metric in this run; do not replace the baseline based on these results.

## Development out-of-fold comparison

| condition            |   lambda |   development_oof_ap |   development_threshold | development_precision_floor_feasible   |   development_precision |   development_recall |   development_f1 |   development_flagged |   development_tp |   development_fp |   development_fn |   development_tn |   development_n |   development_positives |
|:---------------------|---------:|---------------------:|------------------------:|:---------------------------------------|------------------------:|---------------------:|-----------------:|----------------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|------------------------:|
| penultimate_baseline |   nan    |             0.507717 |                0.554945 | True                                   |                0.407407 |             0.511628 |         0.453608 |                    54 |               22 |               32 |               21 |              224 |             299 |                      43 |
| c3_exp_lambda_0.5    |     0.5  |             0.506435 |                0.532566 | True                                   |                0.411765 |             0.488372 |         0.446809 |                    51 |               21 |               30 |               22 |              226 |             299 |                      43 |
| c3_exp_lambda_0.25   |     0.25 |             0.505679 |                0.499946 | True                                   |                0.403509 |             0.534884 |         0.46     |                    57 |               23 |               34 |               20 |              222 |             299 |                      43 |
| c3_exp_lambda_1      |     1    |             0.50235  |                0.493078 | True                                   |                0.4      |             0.511628 |         0.44898  |                    55 |               22 |               33 |               21 |              223 |             299 |                      43 |
| c3_exp_lambda_0      |     0    |             0.50064  |                0.523962 | True                                   |                0.416667 |             0.465116 |         0.43956  |                    48 |               20 |               28 |               23 |              228 |             299 |                      43 |

## Locked test comparison

| condition            |   lambda | best_c3_variant_by_development_ap   |   development_oof_ap |   test_average_precision |   test_threshold |   test_precision |   test_recall |   test_f1 |   test_flagged |   test_tp |   test_fp |   test_fn |   test_tn |   test_n |   test_positives |
|:---------------------|---------:|:------------------------------------|---------------------:|-------------------------:|-----------------:|-----------------:|--------------:|----------:|---------------:|----------:|----------:|----------:|----------:|---------:|-----------------:|
| penultimate_baseline |   nan    | False                               |             0.507717 |                 0.916029 |         0.554945 |         0.692308 |      0.818182 |      0.75 |             13 |         9 |         4 |         2 |        60 |       75 |               11 |
| c3_exp_lambda_0      |     0    | False                               |             0.50064  |                 0.913648 |         0.523962 |         0.692308 |      0.818182 |      0.75 |             13 |         9 |         4 |         2 |        60 |       75 |               11 |
| c3_exp_lambda_0.25   |     0.25 | False                               |             0.505679 |                 0.913648 |         0.499946 |         0.692308 |      0.818182 |      0.75 |             13 |         9 |         4 |         2 |        60 |       75 |               11 |
| c3_exp_lambda_0.5    |     0.5  | True                                |             0.506435 |                 0.911255 |         0.532566 |         0.692308 |      0.818182 |      0.75 |             13 |         9 |         4 |         2 |        60 |       75 |               11 |
| c3_exp_lambda_1      |     1    | False                               |             0.50235  |                 0.906926 |         0.493078 |         0.692308 |      0.818182 |      0.75 |             13 |         9 |         4 |         2 |        60 |       75 |               11 |

## Interpretation rules

- Prefer a C3 condition only if its development OOF AP improves over baseline and the locked test result is directionally consistent. One test split with a small positive count is noisy.
- A higher AP measures ranking, not calibrated probability. Precision/recall thresholds are selected from development OOF scores only.
- Repeated sampled-prefix fits reduce dependence on a single random draw; they do not create additional independent students.
- Since validation/test remain penultimate-only, this experiment tests whether earlier-prefix training improves penultimate prediction. It does not directly establish performance at earlier live checkpoints.
- Existing files, datasets, scripts and outputs were not modified or deleted. This script and its outputs are additive.

## Candidate prefix construction

- For each training student, candidate prefix lengths are k=4 through that student's penultimate prefix (k=N-1).
- If a student's penultimate prefix is k=3 (four total attempts), the only valid row is the penultimate k=3 row and it is used as a fallback.
- Sampling probability is proportional to `exp(-lambda * (penultimate_k - k))`; therefore the penultimate prefix is most likely, and earlier prefixes become exponentially less likely.
- Each student contributes exactly one row to each fitted training dataset, avoiding extra row-level weighting for students with longer histories.

## Files

- `development_oof_comparison.csv`: OOF AP and development-selected operating point for each condition.
- `locked_test_comparison.csv`: locked test metrics for each condition, on identical test students.
- `test_predictions_*.csv`: per-student held-out scores and flags for each condition.
- `candidate_prefix_audit.csv`: number of eligible prefixes per student and fallback indicator.
- `oof_fold_sampling_diagnostics.csv`: fold/repeat diagnostics.

## Limitations

- This is a single locked outer split to preserve comparability with the existing seed-42 experiment; uncertainty across new cohorts is not estimated.
- Sampling a shorter prefix while retaining eventual outcome labels is valid for eventual-dropout prediction, but the feature distribution differs from the penultimate test distribution.
- The project has 500 students overall and 374 eligible C-stage students; results should be treated as exploratory.

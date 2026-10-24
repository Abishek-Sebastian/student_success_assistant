# Final A/B/C cascade evaluation

## Fixed configuration
- A: class-weighted logistic regression; demographics/signup-calendar features.
- B: class-weighted logistic regression; compact + per-lesson features.
- C: depth-2 XGBoost with `scale_pos_weight` calculated from unique training students; compact + per-lesson features.
- C uses XGBoost rather than the degree-3 polynomial SVM because the existing development OOF comparisons were approximately 0.529 AP for weighted depth-2 XGBoost, 0.521 for bagged degree-3 SVM, and 0.493 for class-weighted degree-3 SVM. This is a fixed final candidate, not a claim that XGBoost wins on every split.
- Existing eventual-dropout target, snapshot builder, feature construction, and A/B/C cascade are unchanged.

## Evaluation protocol
- Outer student-level 80/20 stratified holdouts: seeds [17, 42, 73]. The splits may overlap across seeds.
- Thresholds are selected only from 2-fold grouped OOF scores on each outer development set.
- Joint threshold selection maximizes recall subject to a 40% development precision floor; if infeasible, the script falls back to the best F1 threshold combination and records this.
- Stage B/C snapshot scores are aggregated by maximum per student, matching the existing cascade policy.
- All binary policy metrics are evaluated on the full outer test cohort. Stage-specific AP/AUC are computed among students with a score available at that stage; `scored_student_coverage` reports this coverage.
- End-to-end AP/AUC use the maximum of development-OOF-referenced percentile ranks across available stages, since raw scores from different models are not directly comparable. This composite rank is not a calibrated probability.
- Brier score and log loss are reported only for stage-level raw model scores in [0,1]. Class weighting can make these scores poorly calibrated; interpret them cautiously.

## Stepwise metrics (mean/std across outer splits)
| scope                   |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |    f1_std |   accuracy_mean |   accuracy_std |   specificity_mean |   specificity_std |   balanced_accuracy_mean |   balanced_accuracy_std |   mcc_mean |   mcc_std |   average_precision_mean |   average_precision_std |   roc_auc_mean |   roc_auc_std |   brier_score_mean |   brier_score_std |   log_loss_mean |   log_loss_std |   flagged_mean |   flagged_std |   flag_rate_mean |   flag_rate_std |   tp_mean |   tp_std |   fp_mean |   fp_std |   fn_mean |   fn_std |   tn_mean |   tn_std |   n_scored_mean |   n_scored_std |   scored_student_coverage_mean |   scored_student_coverage_std |
|:------------------------|-----------------:|----------------:|--------------:|-------------:|----------:|----------:|----------------:|---------------:|-------------------:|------------------:|-------------------------:|------------------------:|-----------:|----------:|-------------------------:|------------------------:|---------------:|--------------:|-------------------:|------------------:|----------------:|---------------:|---------------:|--------------:|-----------------:|----------------:|----------:|---------:|----------:|---------:|----------:|---------:|----------:|---------:|----------------:|---------------:|-------------------------------:|------------------------------:|
| cascade_A_only          |         0.047619 |       0.0824786 |     0.0133333 |     0.023094 | 0.0208333 | 0.0360844 |        0.73     |      0.0264575 |           0.968889 |         0.0428607 |                 0.491111 |               0.0101835 | -0.0419701 | 0.0366799 |                 0.345739 |               0.0652522 |       0.613956 |     0.0718232 |           0.280348 |        0.0282612  |        0.868276 |     0.0712152  |        2.66667 |       3.78594 |        0.0266667 |       0.0378594 |  0.333333 |  0.57735 |   2.33333 |  3.21455 |  24.6667  |  0.57735 |   72.6667 |  3.21455 |        100      |        0       |                       100      |                       0       |
| cascade_A_plus_B        |         0.470204 |       0.101786  |     0.68      |     0.08     | 0.550059  | 0.0696717 |        0.716667 |      0.0763763 |           0.728889 |         0.113399  |                 0.704444 |               0.0491408 |  0.374137  | 0.106247  |                 0.466448 |               0.0896002 |       0.716978 |     0.0252058 |           0.311275 |        0.0128484  |        0.942199 |     0.0256872  |       37.3333  |       9.71253 |        0.373333  |       0.0971253 | 17        |  2       |  20.3333  |  8.5049  |   8       |  2       |   54.6667 |  8.5049  |        100      |        0       |                       100      |                       0       |
| cascade_A_plus_B_plus_C |         0.420924 |       0.0358977 |     0.773333  |     0.061101 | 0.543159  | 0.0216814 |        0.673333 |      0.0472582 |           0.64     |         0.0811035 |                 0.706667 |               0.0176383 |  0.361734  | 0.0319849 |                 0.549559 |               0.0589812 |       0.757689 |     0.0166967 |           0.366492 |        0.00956891 |        1.09001  |     0.0237309  |       46.3333  |       7.50555 |        0.463333  |       0.0750555 | 19.3333   |  1.52753 |  27       |  6.08276 |   5.66667 |  1.52753 |   48      |  6.08276 |        100      |        0       |                       100      |                       0       |
| stage_A_standalone      |         0.047619 |       0.0824786 |     0.0133333 |     0.023094 | 0.0208333 | 0.0360844 |        0.73     |      0.0264575 |           0.968889 |         0.0428607 |                 0.491111 |               0.0101835 | -0.0419701 | 0.0366799 |                 0.348118 |               0.0669133 |       0.613422 |     0.0712765 |           0.238494 |        0.016746   |        0.670577 |     0.0372744  |        2.66667 |       3.78594 |        0.0266667 |       0.0378594 |  0.333333 |  0.57735 |   2.33333 |  3.21455 |  24.6667  |  0.57735 |   72.6667 |  3.21455 |        100      |        0       |                       100      |                       0       |
| stage_B_standalone      |         0.48888  |       0.123983  |     0.68      |     0.08     | 0.560956  | 0.0806712 |        0.726667 |      0.0862168 |           0.742222 |         0.126725  |                 0.711111 |               0.0542969 |  0.391216  | 0.122997  |                 0.576739 |               0.0708194 |       0.768689 |     0.0331783 |           0.21683  |        0.00638622 |        0.62246  |     0.00959484 |       36.3333  |      10.6927  |        0.363333  |       0.106927  | 17        |  2       |  19.3333  |  9.50438 |   8       |  2       |   55.6667 |  9.50438 |         99.6667 |        0.57735 |                        99.6667 |                       0.57735 |
| stage_C_standalone      |         0.476409 |       0.112201  |     0.4       |     0.10583  | 0.426436  | 0.0698741 |        0.733333 |      0.0416333 |           0.844444 |         0.0684214 |                 0.622222 |               0.0428607 |  0.262705  | 0.0954043 |                 0.692664 |               0.038565  |       0.87132  |     0.0435497 |           0.183662 |        0.0240676  |        0.53627  |     0.0550253  |       21.6667  |       7.02377 |        0.216667  |       0.0702377 | 10        |  2.64575 |  11.6667  |  5.1316  |  15       |  2.64575 |   63.3333 |  5.1316  |         85.3333 |        2.51661 |                        85.3333 |                       2.51661 |

## End-to-end results by split
|   seed |   precision |   recall |       f1 |   accuracy |   specificity |   balanced_accuracy |      mcc |   average_precision |   roc_auc |   flagged |   tp |   fp |   fn |   tn |
|-------:|------------:|---------:|---------:|-----------:|--------------:|--------------------:|---------:|--------------------:|----------:|----------:|-----:|-----:|-----:|-----:|
|     17 |    0.452381 |     0.76 | 0.567164 |       0.71 |      0.693333 |            0.726667 | 0.397722 |            0.484468 |  0.7416   |        42 |   19 |   23 |    6 |   52 |
|     42 |    0.428571 |     0.72 | 0.537313 |       0.69 |      0.68     |            0.7      | 0.350931 |            0.564752 |  0.756533 |        42 |   18 |   24 |    7 |   51 |
|     73 |    0.381818 |     0.84 | 0.525    |       0.62 |      0.546667 |            0.693333 | 0.33655  |            0.599458 |  0.774933 |        55 |   21 |   34 |    4 |   41 |

## Mean end-to-end summary
```json
{
  "outer_seeds": [
    17,
    42,
    73
  ],
  "precision_floor_selected_on_development": 0.4,
  "mean_precision": 0.4209235209235209,
  "mean_recall": 0.7733333333333333,
  "mean_f1": 0.5431592039800995,
  "mean_accuracy": 0.6733333333333333,
  "mean_specificity": 0.64,
  "mean_balanced_accuracy": 0.7066666666666667,
  "mean_mcc": 0.3617344560936173,
  "mean_average_precision": 0.5495594269723616,
  "mean_roc_auc": 0.7576888888888887,
  "mean_flagged": 46.333333333333336,
  "mean_tp": 19.333333333333332,
  "mean_fp": 27.0,
  "mean_fn": 5.666666666666667,
  "mean_tn": 48.0
}
```

## Interpretation cautions
- The dataset has 500 students and 125 eventual-dropout positives; three overlapping holdouts are internal robustness checks, not independent external validation.
- There is no true dropout timestamp. Results predict the supplied eventual-dropout label and do not establish warning lead time or intervention impact.
- Because B/C use a maximum over available snapshots, students with more snapshots have more opportunities to cross a threshold.
- The development precision floor is a threshold-selection constraint, not a guarantee on test or future data.
- Class-weighted logistic/XGBoost scores are not guaranteed to be calibrated probabilities. Ranking metrics and thresholded metrics answer different questions.

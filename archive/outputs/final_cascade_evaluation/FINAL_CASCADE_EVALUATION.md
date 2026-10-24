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
| cascade_A_only          |         0.047619 |       0.0824786 |     0.0266667 |     0.046188 |  0.034188 | 0.0592154 |        0.713333 |      0.0550757 |           0.942222 |         0.0887777 |                 0.484444 |               0.0214303 | -0.0526199 | 0.0501359 |                 0.343758 |               0.0637635 |       0.612356 |     0.0707433 |           0.28636  |        0.0307793  |        0.878744 |     0.101451   |         5      |       7.81025 |         0.05     |       0.0781025 |  0.666667 |   1.1547 |   4.33333 |  6.65833 |   24.3333 |   1.1547 |   70.6667 |  6.65833 |        100      |        0       |                       100      |                       0       |
| cascade_A_plus_B        |         0.490278 |       0.0486508 |     0.64      |     0.08     |  0.554403 | 0.0548703 |        0.743333 |      0.0305505 |           0.777778 |         0.030792  |                 0.708889 |               0.0428607 |  0.386139  | 0.0771933 |                 0.478085 |               0.124345  |       0.716356 |     0.0364138 |           0.31574  |        0.00858342 |        0.94814  |     0.0561318  |        32.6667 |       3.05505 |         0.326667 |       0.0305505 | 16        |   2      |  16.6667  |  2.3094  |    9      |   2      |   58.3333 |  2.3094  |        100      |        0       |                       100      |                       0       |
| cascade_A_plus_B_plus_C |         0.430422 |       0.0662382 |     0.72      |     0.04     |  0.537647 | 0.0622989 |        0.686667 |      0.0665833 |           0.675556 |         0.0781262 |                 0.697778 |               0.05591   |  0.3488    | 0.10196   |                 0.56881  |               0.0642891 |       0.762222 |     0.0283738 |           0.368689 |        0.00319203 |        1.09107  |     0.0618267  |        42.3333 |       5.1316  |         0.423333 |       0.051316  | 18        |   1      |  24.3333  |  5.85947 |    7      |   1      |   50.6667 |  5.85947 |        100      |        0       |                       100      |                       0       |
| stage_A_standalone      |         0.047619 |       0.0824786 |     0.0266667 |     0.046188 |  0.034188 | 0.0592154 |        0.713333 |      0.0550757 |           0.942222 |         0.0887777 |                 0.484444 |               0.0214303 | -0.0526199 | 0.0501359 |                 0.348118 |               0.0669133 |       0.613422 |     0.0712765 |           0.238494 |        0.016746   |        0.670577 |     0.0372744  |         5      |       7.81025 |         0.05     |       0.0781025 |  0.666667 |   1.1547 |   4.33333 |  6.65833 |   24.3333 |   1.1547 |   70.6667 |  6.65833 |        100      |        0       |                       100      |                       0       |
| stage_B_standalone      |         0.515815 |       0.0574817 |     0.64      |     0.08     |  0.571132 | 0.0664371 |        0.76     |      0.0360555 |           0.8      |         0.023094  |                 0.72     |               0.0503322 |  0.41184   | 0.0926058 |                 0.576739 |               0.0708194 |       0.768689 |     0.0331783 |           0.21683  |        0.00638622 |        0.62246  |     0.00959484 |        31      |       1       |         0.31     |       0.01      | 16        |   2      |  15       |  1.73205 |    9      |   2      |   60      |  1.73205 |         99.6667 |        0.57735 |                        99.6667 |                       0.57735 |
| stage_C_standalone      |         0.412698 |       0.0765369 |     0.373333  |     0.046188 |  0.389876 | 0.0495106 |        0.706667 |      0.0404145 |           0.817778 |         0.053886  |                 0.595556 |               0.0328859 |  0.199634  | 0.0771164 |                 0.692664 |               0.038565  |       0.87132  |     0.0435497 |           0.183662 |        0.0240676  |        0.53627  |     0.0550253  |        23      |       4.3589  |         0.23     |       0.043589  |  9.33333  |   1.1547 |  13.6667  |  4.04145 |   15.6667 |   1.1547 |   61.3333 |  4.04145 |         85.3333 |        2.51661 |                        85.3333 |                       2.51661 |

## End-to-end results by split
|   seed |   precision |   recall |       f1 |   accuracy |   specificity |   balanced_accuracy |      mcc |   average_precision |   roc_auc |   flagged |   tp |   fp |   fn |   tn |
|-------:|------------:|---------:|---------:|-----------:|--------------:|--------------------:|---------:|--------------------:|----------:|----------:|-----:|-----:|-----:|-----:|
|     17 |    0.463415 |     0.76 | 0.575758 |       0.72 |      0.706667 |            0.733333 | 0.410856 |            0.504764 |  0.732533 |        41 |   19 |   22 |    6 |   53 |
|     42 |    0.354167 |     0.68 | 0.465753 |       0.61 |      0.586667 |            0.633333 | 0.231125 |            0.568326 |  0.765067 |        48 |   17 |   31 |    8 |   44 |
|     73 |    0.473684 |     0.72 | 0.571429 |       0.73 |      0.733333 |            0.726667 | 0.404418 |            0.63334  |  0.789067 |        38 |   18 |   20 |    7 |   55 |

## Mean end-to-end summary
```json
{
  "outer_seeds": [
    17,
    42,
    73
  ],
  "precision_floor_selected_on_development": 0.4,
  "mean_precision": 0.43042183711310794,
  "mean_recall": 0.7200000000000001,
  "mean_f1": 0.5376465239478938,
  "mean_accuracy": 0.6866666666666666,
  "mean_specificity": 0.6755555555555556,
  "mean_balanced_accuracy": 0.6977777777777777,
  "mean_mcc": 0.3487996989754179,
  "mean_average_precision": 0.5688098392067528,
  "mean_roc_auc": 0.7622222222222222,
  "mean_flagged": 42.333333333333336,
  "mean_tp": 18.0,
  "mean_fp": 24.333333333333332,
  "mean_fn": 7.0,
  "mean_tn": 50.666666666666664
}
```

## Interpretation cautions
- The dataset has 500 students and 125 eventual-dropout positives; three overlapping holdouts are internal robustness checks, not independent external validation.
- There is no true dropout timestamp. Results predict the supplied eventual-dropout label and do not establish warning lead time or intervention impact.
- Because B/C use a maximum over available snapshots, students with more snapshots have more opportunities to cross a threshold.
- The development precision floor is a threshold-selection constraint, not a guarantee on test or future data.
- Class-weighted logistic/XGBoost scores are not guaranteed to be calibrated probabilities. Ranking metrics and thresholded metrics answer different questions.

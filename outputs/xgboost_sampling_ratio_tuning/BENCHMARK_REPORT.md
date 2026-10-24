# XGBoost and oversampling tuning: grouped student-level benchmark

## Design
- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.
- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.
- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.
- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.
- Focused imbalance follow-up: A uses demographics/signup-calendar only; B/C are fixed to the previously strongest compact + per-lesson feature family.
- Compact-only mode: False.
- Families: logistic regression, GradientBoosting, and XGBoost. XGBoost variants compare shallow trees, regularization, learning rates, group-level random oversampling, and student-level class weighting.
- The top 2 stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.
- Outer split count in this run: 1. Repeated outer splits can overlap in students and are not independent external validation.

## Repeated outer cascade summary
|   precision_floor |   test_precision_mean |   test_precision_std |   test_precision_min |   test_precision_max |   test_recall_mean |   test_recall_std |   test_recall_min |   test_recall_max |   test_flagged_mean |   test_flagged_std |   test_flagged_min |   test_flagged_max |   test_tp_mean |   test_tp_std |   test_tp_min |   test_tp_max |   test_fp_mean |   test_fp_std |   test_fp_min |   test_fp_max |   test_fn_mean |   test_fn_std |   test_fn_min |   test_fn_max |   precision_floor_met_on_test_mean |   development_feasible_mean |
|------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|-----------------------------------:|----------------------------:|
|               0.3 |              0.26506  |                  nan |             0.26506  |             0.26506  |               0.88 |               nan |              0.88 |              0.88 |                  83 |                nan |                 83 |                 83 |             22 |           nan |            22 |            22 |             61 |           nan |            61 |            61 |              3 |           nan |             3 |             3 |                                  0 |                           1 |
|               0.4 |              0.409091 |                  nan |             0.409091 |             0.409091 |               0.72 |               nan |              0.72 |              0.72 |                  44 |                nan |                 44 |                 44 |             18 |           nan |            18 |            18 |             26 |           nan |            26 |            26 |              7 |           nan |             7 |             7 |                                  1 |                           1 |
|               0.5 |              0.619048 |                  nan |             0.619048 |             0.619048 |               0.52 |               nan |              0.52 |              0.52 |                  21 |                nan |                 21 |                 21 |             13 |           nan |            13 |            13 |              8 |           nan |             8 |             8 |             12 |           nan |            12 |            12 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                     | feature_set         |   selection_count |
|------------------:|:--------|:-------------------------------------------|:--------------------|------------------:|
|               0.3 | A       | demographics::xgb_d2_os50                  | demographics        |                 1 |
|               0.3 | B       | compact_plus_module::logistic              | compact_plus_module |                 1 |
|               0.3 | C       | compact_plus_module::xgb_d1_group_weight   | compact_plus_module |                 1 |
|               0.4 | A       | demographics::xgb_d2_os50                  | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic              | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::xgb_d2_group_weight   | compact_plus_module |                 1 |
|               0.5 | A       | demographics::xgb_d2_os50                  | demographics        |                 1 |
|               0.5 | B       | compact_plus_module::logistic_class_weight | compact_plus_module |                 1 |
|               0.5 | C       | compact_plus_module::xgb_d1_group_weight   | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                         |     mean |   std |      min |      max |
|:--------|:-----------------------------------------------|---------:|------:|---------:|---------:|
| A       | demographics::xgb_d2_os50                      | 0.286075 |     0 | 0.286075 | 0.286075 |
| A       | demographics::xgb_d1_os50                      | 0.271642 |     0 | 0.271642 | 0.271642 |
| A       | demographics::xgb_d2_group_weight              | 0.270076 |     0 | 0.270076 | 0.270076 |
| A       | demographics::xgb_d2_slow                      | 0.268697 |     0 | 0.268697 | 0.268697 |
| A       | demographics::xgb_d2_strong_reg                | 0.268376 |     0 | 0.268376 | 0.268376 |
| A       | demographics::xgb_d2_spw1_5                    | 0.267263 |     0 | 0.267263 | 0.267263 |
| A       | demographics::xgb_d1_slow                      | 0.266571 |     0 | 0.266571 | 0.266571 |
| A       | demographics::xgb_d2_spw2                      | 0.266463 |     0 | 0.266463 | 0.266463 |
| A       | demographics::xgb_d1_group_weight              | 0.261251 |     0 | 0.261251 | 0.261251 |
| A       | demographics::xgb_d2_medium_lr                 | 0.261161 |     0 | 0.261161 | 0.261161 |
| A       | demographics::xgb_d1_spw1_5                    | 0.259869 |     0 | 0.259869 | 0.259869 |
| A       | demographics::logistic                         | 0.259325 |     0 | 0.259325 | 0.259325 |
| B       | compact_plus_module::logistic                  | 0.443869 |     0 | 0.443869 | 0.443869 |
| B       | compact_plus_module::logistic_class_weight     | 0.440548 |     0 | 0.440548 | 0.440548 |
| B       | compact_plus_module::logistic_group_oversample | 0.430726 |     0 | 0.430726 | 0.430726 |
| B       | compact_plus_module::xgb_d1_group_oversample   | 0.422073 |     0 | 0.422073 | 0.422073 |
| B       | compact_plus_module::xgb_d1_more_trees         | 0.4201   |     0 | 0.4201   | 0.4201   |
| B       | compact_plus_module::xgb_d1_spw1_5             | 0.418696 |     0 | 0.418696 | 0.418696 |
| B       | compact_plus_module::xgb_d1_spw2               | 0.416467 |     0 | 0.416467 | 0.416467 |
| B       | compact_plus_module::xgb_d1_slow               | 0.416346 |     0 | 0.416346 | 0.416346 |
| B       | compact_plus_module::xgb_d1_os75               | 0.415884 |     0 | 0.415884 | 0.415884 |
| B       | compact_plus_module::xgb_d1_group_weight       | 0.41583  |     0 | 0.41583  | 0.41583  |
| B       | compact_plus_module::xgb_d1_os50               | 0.415397 |     0 | 0.415397 | 0.415397 |
| B       | compact_plus_module::gradient_boosting         | 0.40032  |     0 | 0.40032  | 0.40032  |
| C       | compact_plus_module::xgb_d2_group_weight       | 0.587138 |     0 | 0.587138 | 0.587138 |
| C       | compact_plus_module::xgb_d1_group_weight       | 0.580094 |     0 | 0.580094 | 0.580094 |
| C       | compact_plus_module::xgb_d2_os50               | 0.568541 |     0 | 0.568541 | 0.568541 |
| C       | compact_plus_module::xgb_d2_medium_lr          | 0.564038 |     0 | 0.564038 | 0.564038 |
| C       | compact_plus_module::xgb_d2_spw2               | 0.557732 |     0 | 0.557732 | 0.557732 |
| C       | compact_plus_module::xgb_d2_group_oversample   | 0.552851 |     0 | 0.552851 | 0.552851 |
| C       | compact_plus_module::xgb_d3_group_oversample   | 0.551401 |     0 | 0.551401 | 0.551401 |
| C       | compact_plus_module::xgb_d2_os75               | 0.546557 |     0 | 0.546557 | 0.546557 |
| C       | compact_plus_module::xgb_d2_spw1_5             | 0.542868 |     0 | 0.542868 | 0.542868 |
| C       | compact_plus_module::xgb_d1_spw2               | 0.537496 |     0 | 0.537496 | 0.537496 |
| C       | compact_plus_module::xgb_d1_more_trees         | 0.535345 |     0 | 0.535345 | 0.535345 |
| C       | compact_plus_module::xgb_d1_group_oversample   | 0.530525 |     0 | 0.530525 | 0.530525 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum probability across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a focused XGBoost/imbalance search, not an exhaustive Bayesian or large compute hyperparameter search. XGBoost candidates emphasize shallow trees and regularization because the cohort is small.

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
|               0.3 |              0.277108 |                  nan |             0.277108 |             0.277108 |               0.92 |               nan |              0.92 |              0.92 |                  83 |                nan |                 83 |                 83 |             23 |           nan |            23 |            23 |             60 |           nan |            60 |            60 |              2 |           nan |             2 |             2 |                                  0 |                           1 |
|               0.4 |              0.352941 |                  nan |             0.352941 |             0.352941 |               0.72 |               nan |              0.72 |              0.72 |                  51 |                nan |                 51 |                 51 |             18 |           nan |            18 |            18 |             33 |           nan |            33 |            33 |              7 |           nan |             7 |             7 |                                  0 |                           1 |
|               0.5 |              0.517241 |                  nan |             0.517241 |             0.517241 |               0.6  |               nan |              0.6  |              0.6  |                  29 |                nan |                 29 |                 29 |             15 |           nan |            15 |            15 |             14 |           nan |            14 |            14 |             10 |           nan |            10 |            10 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                       | feature_set         |   selection_count |
|------------------:|:--------|:---------------------------------------------|:--------------------|------------------:|
|               0.3 | A       | demographics::logistic                       | demographics        |                 1 |
|               0.3 | B       | compact_plus_module::logistic_class_weight   | compact_plus_module |                 1 |
|               0.3 | C       | compact_plus_module::xgb_d2_group_weight     | compact_plus_module |                 1 |
|               0.4 | A       | demographics::logistic                       | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic                | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::xgb_d2_group_oversample | compact_plus_module |                 1 |
|               0.5 | A       | demographics::logistic                       | demographics        |                 1 |
|               0.5 | B       | compact_plus_module::logistic_class_weight   | compact_plus_module |                 1 |
|               0.5 | C       | compact_plus_module::xgb_d2_group_oversample | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                         |     mean |   std |      min |      max |
|:--------|:-----------------------------------------------|---------:|------:|---------:|---------:|
| A       | demographics::logistic                         | 0.279809 |     0 | 0.279809 | 0.279809 |
| A       | demographics::xgb_d2_medium_lr                 | 0.279677 |     0 | 0.279677 | 0.279677 |
| A       | demographics::logistic_class_weight            | 0.275362 |     0 | 0.275362 | 0.275362 |
| A       | demographics::xgb_d2_group_weight              | 0.274162 |     0 | 0.274162 | 0.274162 |
| A       | demographics::gradient_boosting                | 0.271956 |     0 | 0.271956 | 0.271956 |
| A       | demographics::logistic_group_oversample        | 0.271424 |     0 | 0.271424 | 0.271424 |
| A       | demographics::xgb_d2_group_oversample          | 0.271277 |     0 | 0.271277 | 0.271277 |
| A       | demographics::xgb_d2_spw2                      | 0.270675 |     0 | 0.270675 | 0.270675 |
| A       | demographics::xgb_d2_os75                      | 0.267248 |     0 | 0.267248 | 0.267248 |
| A       | demographics::xgb_d1_group_weight              | 0.266388 |     0 | 0.266388 | 0.266388 |
| A       | demographics::xgb_d2_spw1_5                    | 0.26575  |     0 | 0.26575  | 0.26575  |
| A       | demographics::xgb_d1_more_trees                | 0.265426 |     0 | 0.265426 | 0.265426 |
| B       | compact_plus_module::logistic_class_weight     | 0.488306 |     0 | 0.488306 | 0.488306 |
| B       | compact_plus_module::logistic                  | 0.485874 |     0 | 0.485874 | 0.485874 |
| B       | compact_plus_module::logistic_group_oversample | 0.484649 |     0 | 0.484649 | 0.484649 |
| B       | compact_plus_module::xgb_d1_os50               | 0.448528 |     0 | 0.448528 | 0.448528 |
| B       | compact_plus_module::xgb_d1_group_weight       | 0.442475 |     0 | 0.442475 | 0.442475 |
| B       | compact_plus_module::xgb_d1_more_trees         | 0.435648 |     0 | 0.435648 | 0.435648 |
| B       | compact_plus_module::xgb_d1_os75               | 0.434761 |     0 | 0.434761 | 0.434761 |
| B       | compact_plus_module::xgb_d1_spw2               | 0.43062  |     0 | 0.43062  | 0.43062  |
| B       | compact_plus_module::xgb_d1_spw1_5             | 0.427764 |     0 | 0.427764 | 0.427764 |
| B       | compact_plus_module::gradient_boosting         | 0.427586 |     0 | 0.427586 | 0.427586 |
| B       | compact_plus_module::xgb_d2_slow               | 0.427334 |     0 | 0.427334 | 0.427334 |
| B       | compact_plus_module::xgb_d2_os50               | 0.426636 |     0 | 0.426636 | 0.426636 |
| C       | compact_plus_module::xgb_d2_group_oversample   | 0.570813 |     0 | 0.570813 | 0.570813 |
| C       | compact_plus_module::xgb_d2_group_weight       | 0.565716 |     0 | 0.565716 | 0.565716 |
| C       | compact_plus_module::xgb_d3_group_oversample   | 0.564507 |     0 | 0.564507 | 0.564507 |
| C       | compact_plus_module::xgb_d2_os75               | 0.559779 |     0 | 0.559779 | 0.559779 |
| C       | compact_plus_module::xgb_d1_group_weight       | 0.552899 |     0 | 0.552899 | 0.552899 |
| C       | compact_plus_module::xgb_d2_os50               | 0.550013 |     0 | 0.550013 | 0.550013 |
| C       | compact_plus_module::xgb_d2_spw2               | 0.545208 |     0 | 0.545208 | 0.545208 |
| C       | compact_plus_module::xgb_d2_medium_lr          | 0.539188 |     0 | 0.539188 | 0.539188 |
| C       | compact_plus_module::logistic_group_oversample | 0.53881  |     0 | 0.53881  | 0.53881  |
| C       | compact_plus_module::gradient_boosting         | 0.537823 |     0 | 0.537823 | 0.537823 |
| C       | compact_plus_module::xgb_d1_group_oversample   | 0.536384 |     0 | 0.536384 | 0.536384 |
| C       | compact_plus_module::logistic_class_weight     | 0.536036 |     0 | 0.536036 | 0.536036 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum probability across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a focused XGBoost/imbalance search, not an exhaustive Bayesian or large compute hyperparameter search. XGBoost candidates emphasize shallow trees and regularization because the cohort is small.

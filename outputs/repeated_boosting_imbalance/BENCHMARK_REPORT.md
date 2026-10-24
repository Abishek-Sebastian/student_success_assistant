# Repeated grouped benchmark: boosted models, oversampling and feature ablations

## Design
- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.
- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.
- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.
- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.
- Feature ablations: compact levels/recency/engagement; + slopes; + per-lesson features (trends and module families are tested separately). A remains demographics/signup-calendar only.
- Compact-only mode: False.
- Families: logistic regression (unweighted and class-weighted), and GradientBoosting. Student-group oversampling is compared with no oversampling for logistic and GradientBoosting.
- The top 3 stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.
- Outer split count in this run: 1. Repeated outer splits can overlap in students and are not independent external validation.

## Repeated outer cascade summary
|   precision_floor |   test_precision_mean |   test_precision_std |   test_precision_min |   test_precision_max |   test_recall_mean |   test_recall_std |   test_recall_min |   test_recall_max |   test_flagged_mean |   test_flagged_std |   test_flagged_min |   test_flagged_max |   test_tp_mean |   test_tp_std |   test_tp_min |   test_tp_max |   test_fp_mean |   test_fp_std |   test_fp_min |   test_fp_max |   test_fn_mean |   test_fn_std |   test_fn_min |   test_fn_max |   precision_floor_met_on_test_mean |   development_feasible_mean |
|------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|-----------------------------------:|----------------------------:|
|               0.4 |              0.583333 |                  nan |             0.583333 |             0.583333 |               0.56 |               nan |              0.56 |              0.56 |                  24 |                nan |                 24 |                 24 |             14 |           nan |            14 |            14 |             10 |           nan |            10 |            10 |             11 |           nan |            11 |            11 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                         | feature_set         |   selection_count |
|------------------:|:--------|:-----------------------------------------------|:--------------------|------------------:|
|               0.4 | A       | demographics::logistic                         | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic                  | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::logistic_group_oversample | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                                  |     mean |   std |      min |      max |
|:--------|:--------------------------------------------------------|---------:|------:|---------:|---------:|
| A       | demographics::logistic                                  | 0.279809 |   nan | 0.279809 | 0.279809 |
| A       | demographics::logistic_class_weight                     | 0.275362 |   nan | 0.275362 | 0.275362 |
| A       | demographics::gradient_boosting                         | 0.271956 |   nan | 0.271956 | 0.271956 |
| A       | demographics::logistic_group_oversample                 | 0.271424 |   nan | 0.271424 | 0.271424 |
| A       | demographics::gradient_boosting_group_oversample        | 0.266492 |   nan | 0.266492 | 0.266492 |
| B       | compact_plus_module::logistic_class_weight              | 0.488306 |   nan | 0.488306 | 0.488306 |
| B       | compact_plus_module::logistic                           | 0.485874 |   nan | 0.485874 | 0.485874 |
| B       | compact_plus_module::logistic_group_oversample          | 0.484649 |   nan | 0.484649 | 0.484649 |
| B       | compact::logistic                                       | 0.471545 |   nan | 0.471545 | 0.471545 |
| B       | compact::logistic_class_weight                          | 0.462347 |   nan | 0.462347 | 0.462347 |
| B       | compact::logistic_group_oversample                      | 0.459726 |   nan | 0.459726 | 0.459726 |
| B       | compact_plus_trends::logistic_group_oversample          | 0.457691 |   nan | 0.457691 | 0.457691 |
| B       | compact_plus_trends::logistic                           | 0.457307 |   nan | 0.457307 | 0.457307 |
| B       | compact_plus_trends::logistic_class_weight              | 0.452503 |   nan | 0.452503 | 0.452503 |
| B       | compact_plus_module::gradient_boosting                  | 0.427586 |   nan | 0.427586 | 0.427586 |
| B       | compact_plus_trends::gradient_boosting                  | 0.422727 |   nan | 0.422727 | 0.422727 |
| B       | compact::gradient_boosting                              | 0.420411 |   nan | 0.420411 | 0.420411 |
| C       | compact_plus_module::logistic_group_oversample          | 0.53881  |   nan | 0.53881  | 0.53881  |
| C       | compact_plus_module::gradient_boosting                  | 0.537823 |   nan | 0.537823 | 0.537823 |
| C       | compact_plus_module::logistic_class_weight              | 0.536036 |   nan | 0.536036 | 0.536036 |
| C       | compact_plus_module::logistic                           | 0.532467 |   nan | 0.532467 | 0.532467 |
| C       | compact_plus_module::gradient_boosting_group_oversample | 0.515974 |   nan | 0.515974 | 0.515974 |
| C       | compact::logistic                                       | 0.400657 |   nan | 0.400657 | 0.400657 |
| C       | compact_plus_trends::logistic_class_weight              | 0.399273 |   nan | 0.399273 | 0.399273 |
| C       | compact::logistic_group_oversample                      | 0.39363  |   nan | 0.39363  | 0.39363  |
| C       | compact_plus_trends::logistic_group_oversample          | 0.392548 |   nan | 0.392548 | 0.392548 |
| C       | compact::logistic_class_weight                          | 0.391677 |   nan | 0.391677 | 0.391677 |
| C       | compact_plus_trends::gradient_boosting                  | 0.387157 |   nan | 0.387157 | 0.387157 |
| C       | compact_plus_trends::logistic                           | 0.385444 |   nan | 0.385444 | 0.385444 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum probability across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a small controlled benchmark, not a broad hyperparameter search. Model configurations are intentionally conservative.

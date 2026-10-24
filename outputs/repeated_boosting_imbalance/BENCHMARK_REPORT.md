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
|               0.4 |              0.435897 |                  nan |             0.435897 |             0.435897 |               0.68 |               nan |              0.68 |              0.68 |                  39 |                nan |                 39 |                 39 |             17 |           nan |            17 |            17 |             22 |           nan |            22 |            22 |              8 |           nan |             8 |             8 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                        | feature_set         |   selection_count |
|------------------:|:--------|:------------------------------|:--------------------|------------------:|
|               0.4 | A       | demographics::logistic        | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::logistic | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                                  |     mean |   std |      min |      max |
|:--------|:--------------------------------------------------------|---------:|------:|---------:|---------:|
| A       | demographics::logistic                                  | 0.259325 |   nan | 0.259325 | 0.259325 |
| A       | demographics::logistic_class_weight                     | 0.257308 |   nan | 0.257308 | 0.257308 |
| A       | demographics::gradient_boosting_group_oversample        | 0.255628 |   nan | 0.255628 | 0.255628 |
| A       | demographics::logistic_group_oversample                 | 0.248927 |   nan | 0.248927 | 0.248927 |
| A       | demographics::gradient_boosting                         | 0.24208  |   nan | 0.24208  | 0.24208  |
| B       | compact_plus_module::logistic                           | 0.443869 |   nan | 0.443869 | 0.443869 |
| B       | compact_plus_module::logistic_class_weight              | 0.440548 |   nan | 0.440548 | 0.440548 |
| B       | compact_plus_module::logistic_group_oversample          | 0.430726 |   nan | 0.430726 | 0.430726 |
| B       | compact_plus_trends::gradient_boosting                  | 0.411811 |   nan | 0.411811 | 0.411811 |
| B       | compact::gradient_boosting                              | 0.411393 |   nan | 0.411393 | 0.411393 |
| B       | compact_plus_trends::logistic                           | 0.405433 |   nan | 0.405433 | 0.405433 |
| B       | compact::logistic                                       | 0.404578 |   nan | 0.404578 | 0.404578 |
| B       | compact_plus_module::gradient_boosting                  | 0.40032  |   nan | 0.40032  | 0.40032  |
| B       | compact::logistic_class_weight                          | 0.396683 |   nan | 0.396683 | 0.396683 |
| B       | compact_plus_trends::logistic_class_weight              | 0.394896 |   nan | 0.394896 | 0.394896 |
| B       | compact_plus_module::gradient_boosting_group_oversample | 0.392938 |   nan | 0.392938 | 0.392938 |
| B       | compact_plus_trends::gradient_boosting_group_oversample | 0.386009 |   nan | 0.386009 | 0.386009 |
| C       | compact_plus_module::logistic                           | 0.522699 |   nan | 0.522699 | 0.522699 |
| C       | compact_plus_module::gradient_boosting_group_oversample | 0.519126 |   nan | 0.519126 | 0.519126 |
| C       | compact_plus_module::gradient_boosting                  | 0.516167 |   nan | 0.516167 | 0.516167 |
| C       | compact_plus_module::logistic_class_weight              | 0.513213 |   nan | 0.513213 | 0.513213 |
| C       | compact_plus_module::logistic_group_oversample          | 0.497495 |   nan | 0.497495 | 0.497495 |
| C       | compact::gradient_boosting                              | 0.457631 |   nan | 0.457631 | 0.457631 |
| C       | compact_plus_trends::gradient_boosting                  | 0.437782 |   nan | 0.437782 | 0.437782 |
| C       | compact::gradient_boosting_group_oversample             | 0.434225 |   nan | 0.434225 | 0.434225 |
| C       | compact_plus_trends::gradient_boosting_group_oversample | 0.408994 |   nan | 0.408994 | 0.408994 |
| C       | compact::logistic_group_oversample                      | 0.396563 |   nan | 0.396563 | 0.396563 |
| C       | compact::logistic_class_weight                          | 0.390433 |   nan | 0.390433 | 0.390433 |
| C       | compact::logistic                                       | 0.388377 |   nan | 0.388377 | 0.388377 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum probability across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a small controlled benchmark, not a broad hyperparameter search. Model configurations are intentionally conservative.

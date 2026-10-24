# Kernel SVM benchmark: nonlinear decision boundaries within the existing cascade

## Design
- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.
- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.
- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.
- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.
- Feature sets tested: compact, compact_plus_trends, compact_plus_module. A remains demographics/signup-calendar only; trends are omitted from this focused kernel run to keep runtime bounded.
- Compact-only mode: False.
- Families: existing logistic regression and GradientBoosting controls, plus RBF and linear SVMs. The RBF grid varies C, gamma and class weighting; a small group-oversampling sensitivity check is also included. A focused degree-2 polynomial SVM grid is included for Stage C only.
- SVMs use standardized numeric features and one-hot categorical features. SVM decision_function margins are used for AP ranking and threshold selection; they are not calibrated dropout probabilities.
- The top 3 stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.
- Outer split count in this run: 1. Repeated outer splits can overlap in students and are not independent external validation.

## Repeated outer cascade summary
|   precision_floor |   test_precision_mean |   test_precision_std |   test_precision_min |   test_precision_max |   test_recall_mean |   test_recall_std |   test_recall_min |   test_recall_max |   test_flagged_mean |   test_flagged_std |   test_flagged_min |   test_flagged_max |   test_tp_mean |   test_tp_std |   test_tp_min |   test_tp_max |   test_fp_mean |   test_fp_std |   test_fp_min |   test_fp_max |   test_fn_mean |   test_fn_std |   test_fn_min |   test_fn_max |   precision_floor_met_on_test_mean |   development_feasible_mean |
|------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|-----------------------------------:|----------------------------:|
|               0.4 |              0.608696 |                  nan |             0.608696 |             0.608696 |               0.56 |               nan |              0.56 |              0.56 |                  23 |                nan |                 23 |                 23 |             14 |           nan |            14 |            14 |              9 |           nan |             9 |             9 |             11 |           nan |            11 |            11 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                         | feature_set         |   selection_count |
|------------------:|:--------|:-----------------------------------------------|:--------------------|------------------:|
|               0.4 | A       | demographics::linear_svm_C1_cwnone             | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic                  | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::logistic_group_oversample | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                                  |     mean |   std |      min |      max |
|:--------|:--------------------------------------------------------|---------:|------:|---------:|---------:|
| A       | demographics::linear_svm_C1_cwnone                      | 0.332158 |   nan | 0.332158 | 0.332158 |
| A       | demographics::rbf_svm_C1_gscale_cwnone                  | 0.306822 |   nan | 0.306822 | 0.306822 |
| A       | demographics::rbf_svm_C0.1_gscale_cwnone                | 0.306453 |   nan | 0.306453 | 0.306453 |
| A       | demographics::rbf_svm_C10_gscale_cwnone                 | 0.303745 |   nan | 0.303745 | 0.303745 |
| A       | demographics::rbf_svm_C1_gscale_cwbalanced              | 0.295024 |   nan | 0.295024 | 0.295024 |
| A       | demographics::rbf_svm_C1_gscale_group_os                | 0.293305 |   nan | 0.293305 | 0.293305 |
| A       | demographics::rbf_svm_C0.1_g0.1_cwnone                  | 0.290612 |   nan | 0.290612 | 0.290612 |
| A       | demographics::rbf_svm_C10_g0.1_cwbalanced               | 0.290332 |   nan | 0.290332 | 0.290332 |
| A       | demographics::rbf_svm_C1_g0.1_cwnone                    | 0.289416 |   nan | 0.289416 | 0.289416 |
| A       | demographics::rbf_svm_C1_g0.1_cwbalanced                | 0.289219 |   nan | 0.289219 | 0.289219 |
| A       | demographics::rbf_svm_C10_gscale_cwbalanced             | 0.287336 |   nan | 0.287336 | 0.287336 |
| A       | demographics::rbf_svm_C10_g0.1_cwnone                   | 0.285888 |   nan | 0.285888 | 0.285888 |
| B       | compact_plus_module::logistic_class_weight              | 0.488306 |   nan | 0.488306 | 0.488306 |
| B       | compact_plus_module::logistic                           | 0.485874 |   nan | 0.485874 | 0.485874 |
| B       | compact_plus_module::logistic_group_oversample          | 0.484649 |   nan | 0.484649 | 0.484649 |
| B       | compact::logistic                                       | 0.471545 |   nan | 0.471545 | 0.471545 |
| B       | compact::logistic_class_weight                          | 0.462347 |   nan | 0.462347 | 0.462347 |
| B       | compact::logistic_group_oversample                      | 0.459726 |   nan | 0.459726 | 0.459726 |
| B       | compact_plus_trends::logistic_group_oversample          | 0.457691 |   nan | 0.457691 | 0.457691 |
| B       | compact_plus_trends::logistic                           | 0.457307 |   nan | 0.457307 | 0.457307 |
| B       | compact_plus_trends::rbf_svm_C0.1_gscale_cwbalanced     | 0.455321 |   nan | 0.455321 | 0.455321 |
| B       | compact_plus_trends::logistic_class_weight              | 0.452503 |   nan | 0.452503 | 0.452503 |
| B       | compact::rbf_svm_C0.1_gscale_cwbalanced                 | 0.450174 |   nan | 0.450174 | 0.450174 |
| B       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced     | 0.446164 |   nan | 0.446164 | 0.446164 |
| C       | compact_plus_module::logistic_group_oversample          | 0.53881  |   nan | 0.53881  | 0.53881  |
| C       | compact_plus_module::gradient_boosting                  | 0.537823 |   nan | 0.537823 | 0.537823 |
| C       | compact_plus_module::logistic_class_weight              | 0.536036 |   nan | 0.536036 | 0.536036 |
| C       | compact_plus_module::logistic                           | 0.532467 |   nan | 0.532467 | 0.532467 |
| C       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced     | 0.527824 |   nan | 0.527824 | 0.527824 |
| C       | compact_plus_module::linear_svm_C1_cwbalanced           | 0.521853 |   nan | 0.521853 | 0.521853 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwbalanced       | 0.518107 |   nan | 0.518107 | 0.518107 |
| C       | compact_plus_module::gradient_boosting_group_oversample | 0.515974 |   nan | 0.515974 | 0.515974 |
| C       | compact_plus_module::rbf_svm_C1_gscale_group_os         | 0.512945 |   nan | 0.512945 | 0.512945 |
| C       | compact_plus_module::poly2_svm_C0.1_g0.1_coef0_1        | 0.501179 |   nan | 0.501179 | 0.501179 |
| C       | compact_plus_module::linear_svm_C1_cwnone               | 0.497267 |   nan | 0.497267 | 0.497267 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwnone           | 0.489235 |   nan | 0.489235 | 0.489235 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum model score across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a small controlled benchmark, not a broad hyperparameter search. Model configurations are intentionally conservative.

## Interpretation
- Compare stage-level average precision before changing the cascade: a kernel can help one stage without helping the others.
- SVM decision margins are ranking scores, not calibrated probabilities. Do not interpret them as literal dropout risks.
- The precision floor is selected on development OOF predictions; held-out test precision is a diagnostic, not a guarantee.

# Kernel SVM benchmark: nonlinear decision boundaries within the existing cascade

## Design
- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.
- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.
- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.
- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.
- Feature sets tested: compact, compact_plus_module. A remains demographics/signup-calendar only; trends are omitted from this focused kernel run to keep runtime bounded.
- Compact-only mode: False.
- Families: existing logistic regression and GradientBoosting controls, plus RBF and linear SVMs. The RBF grid varies C, gamma and class weighting; a small group-oversampling sensitivity check is also included. A focused degree-2 polynomial SVM grid is included for Stage C only.
- SVMs use standardized numeric features and one-hot categorical features. SVM decision_function margins are used for AP ranking and threshold selection; they are not calibrated dropout probabilities.
- The top 3 stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.
- Outer split count in this run: 3. Repeated outer splits can overlap in students and are not independent external validation.

## Repeated outer cascade summary
|   precision_floor |   test_precision_mean |   test_precision_std |   test_precision_min |   test_precision_max |   test_recall_mean |   test_recall_std |   test_recall_min |   test_recall_max |   test_flagged_mean |   test_flagged_std |   test_flagged_min |   test_flagged_max |   test_tp_mean |   test_tp_std |   test_tp_min |   test_tp_max |   test_fp_mean |   test_fp_std |   test_fp_min |   test_fp_max |   test_fn_mean |   test_fn_std |   test_fn_min |   test_fn_max |   precision_floor_met_on_test_mean |   development_feasible_mean |
|------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|-----------------------------------:|----------------------------:|
|               0.4 |              0.505596 |             0.112943 |             0.387755 |             0.612903 |               0.72 |          0.069282 |              0.64 |              0.76 |                  37 |            10.3923 |                 31 |                 49 |             18 |       1.73205 |            16 |            19 |             19 |       9.64365 |            12 |            30 |              7 |       1.73205 |             6 |             9 |                           0.666667 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                           | feature_set         |   selection_count |
|------------------:|:--------|:-------------------------------------------------|:--------------------|------------------:|
|               0.4 | A       | demographics::gradient_boosting_group_oversample | demographics        |                 1 |
|               0.4 | A       | demographics::linear_svm_C1_cwnone               | demographics        |                 1 |
|               0.4 | A       | demographics::rbf_svm_C0.1_g0.1_cwbalanced       | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic                    | compact_plus_module |                 2 |
|               0.4 | B       | compact_plus_module::logistic_group_oversample   | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::poly2_svm_C0.1_g0.1_coef0_1 | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::poly2_svm_C1_gscale_coef0_1 | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::rbf_svm_C1_gscale_group_os  | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                              |     mean |        std |      min |      max |
|:--------|:----------------------------------------------------|---------:|-----------:|---------:|---------:|
| A       | demographics::linear_svm_C1_cwnone                  | 0.303851 | 0.0523335  | 0.261513 | 0.362362 |
| A       | demographics::logistic_class_weight                 | 0.300489 | 0.0423156  | 0.257308 | 0.341883 |
| A       | demographics::logistic                              | 0.297529 | 0.0373686  | 0.259325 | 0.334002 |
| A       | demographics::logistic_group_oversample             | 0.29545  | 0.0416722  | 0.248927 | 0.329354 |
| A       | demographics::rbf_svm_C0.1_g0.1_cwbalanced          | 0.291911 | 0.0255696  | 0.264668 | 0.315391 |
| A       | demographics::gradient_boosting                     | 0.287888 | 0.0421411  | 0.24208  | 0.325009 |
| A       | demographics::linear_svm_C1_cwbalanced              | 0.287357 | 0.0317969  | 0.255774 | 0.319364 |
| A       | demographics::rbf_svm_C1_g0.1_cwbalanced            | 0.285107 | 0.0283092  | 0.2601   | 0.315842 |
| A       | demographics::rbf_svm_C0.1_gscale_cwnone            | 0.284536 | 0.0313846  | 0.24872  | 0.307225 |
| A       | demographics::rbf_svm_C1_gscale_cwnone              | 0.284435 | 0.0304473  | 0.249809 | 0.307018 |
| A       | demographics::rbf_svm_C0.1_gscale_cwbalanced        | 0.282893 | 0.0183953  | 0.264547 | 0.301337 |
| A       | demographics::rbf_svm_C0.1_g0.1_cwnone              | 0.281089 | 0.0269783  | 0.250272 | 0.300442 |
| B       | compact_plus_module::logistic_class_weight          | 0.467013 | 0.0258924  | 0.440548 | 0.492292 |
| B       | compact_plus_module::logistic                       | 0.465615 | 0.0255689  | 0.443869 | 0.493783 |
| B       | compact_plus_module::logistic_group_oversample      | 0.458951 | 0.0246318  | 0.430726 | 0.476104 |
| B       | compact_plus_module::linear_svm_C1_cwnone           | 0.446629 | 0.030052   | 0.41239  | 0.468636 |
| B       | compact_plus_module::linear_svm_C1_cwbalanced       | 0.434303 | 0.0344468  | 0.39612  | 0.463044 |
| B       | compact::rbf_svm_C0.1_gscale_cwbalanced             | 0.434205 | 0.0154922  | 0.416707 | 0.446175 |
| B       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced | 0.432005 | 0.0192522  | 0.415623 | 0.45321  |
| B       | compact_plus_module::rbf_svm_C0.1_gscale_cwnone     | 0.428772 | 0.0421049  | 0.40034  | 0.477143 |
| B       | compact_plus_module::rbf_svm_C1_gscale_cwnone       | 0.4287   | 0.0412023  | 0.401417 | 0.476096 |
| B       | compact_plus_module::rbf_svm_C1_gscale_group_os     | 0.421336 | 0.0159274  | 0.403575 | 0.434351 |
| B       | compact::rbf_svm_C1_gscale_cwbalanced               | 0.418689 | 0.023558   | 0.403191 | 0.445799 |
| B       | compact::rbf_svm_C0.1_g0.1_cwbalanced               | 0.417184 | 0.0213715  | 0.398856 | 0.440659 |
| C       | compact_plus_module::poly2_svm_C1_gscale_coef0_1    | 0.527294 | 0.0256696  | 0.509281 | 0.556686 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwnone       | 0.526885 | 0.0165584  | 0.508847 | 0.541395 |
| C       | compact_plus_module::poly2_svm_C0.1_g0.1_coef0_1    | 0.514316 | 0.0398615  | 0.476938 | 0.556266 |
| C       | compact_plus_module::logistic                       | 0.506847 | 0.0443033  | 0.456799 | 0.541044 |
| C       | compact_plus_module::rbf_svm_C0.1_gscale_cwnone     | 0.500132 | 0.00938685 | 0.492133 | 0.510466 |
| C       | compact_plus_module::logistic_class_weight          | 0.499987 | 0.0488117  | 0.445925 | 0.540823 |
| C       | compact_plus_module::rbf_svm_C1_gscale_group_os     | 0.499608 | 0.0523006  | 0.445467 | 0.54985  |
| C       | compact_plus_module::rbf_svm_C10_gscale_cwnone      | 0.49394  | 0.0428463  | 0.450129 | 0.535751 |
| C       | compact_plus_module::poly2_svm_C0.1_gscale_coef0_1  | 0.491976 | 0.0226095  | 0.473146 | 0.517052 |
| C       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced | 0.491119 | 0.0314     | 0.459415 | 0.522206 |
| C       | compact_plus_module::linear_svm_C1_cwnone           | 0.483852 | 0.0178943  | 0.472979 | 0.504505 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwbalanced   | 0.483195 | 0.0221285  | 0.459645 | 0.503556 |

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

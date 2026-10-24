# Kernel SVM benchmark: nonlinear decision boundaries within the existing cascade

## Design
- Outer 80/20 student-level stratified splits (seeds supplied at runtime). A student's rows never cross a split.
- Within each outer development set, grouped StratifiedGroupKFold generates inner out-of-fold predictions for model and threshold selection.
- Each model/feature/sampling configuration is fitted within each inner training fold; preprocessing is fit only on the original training fold.
- Group-level random oversampling duplicates positive students as groups, retaining all their snapshots. Each selected student-copy has total sample weight one.
- Feature ablations: compact levels/recency/engagement; + slopes; + per-lesson features (trends and module families are tested separately). A remains demographics/signup-calendar only.
- Compact-only mode: False.
- Families: existing logistic regression and GradientBoosting controls, plus RBF and linear SVMs. The RBF grid varies C, gamma and class weighting; a small group-oversampling sensitivity check is also included.
- SVMs use standardized numeric features and one-hot categorical features. SVM decision_function margins are used for AP ranking and threshold selection; they are not calibrated dropout probabilities.
- The top 3 stage configurations by inner OOF AP are combined, then stage thresholds are jointly selected to maximize recall under each precision floor. This limits combinatorial search; it is not exhaustive over all configurations.
- Outer split count in this run: 3. Repeated outer splits can overlap in students and are not independent external validation.

## Repeated outer cascade summary
|   precision_floor |   test_precision_mean |   test_precision_std |   test_precision_min |   test_precision_max |   test_recall_mean |   test_recall_std |   test_recall_min |   test_recall_max |   test_flagged_mean |   test_flagged_std |   test_flagged_min |   test_flagged_max |   test_tp_mean |   test_tp_std |   test_tp_min |   test_tp_max |   test_fp_mean |   test_fp_std |   test_fp_min |   test_fp_max |   test_fn_mean |   test_fn_std |   test_fn_min |   test_fn_max |   precision_floor_met_on_test_mean |   development_feasible_mean |
|------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|---------------:|--------------:|--------------:|--------------:|-----------------------------------:|----------------------------:|
|               0.4 |              0.511265 |             0.104156 |             0.404762 |             0.612903 |           0.693333 |          0.061101 |              0.64 |              0.76 |             34.6667 |            6.35085 |                 31 |                 42 |        17.3333 |       1.52753 |            16 |            19 |        17.3333 |       6.80686 |            12 |            25 |        7.66667 |       1.52753 |             6 |             9 |                                  1 |                           1 |

## Selected configuration frequency
|   precision_floor | stage   | config                                           | feature_set         |   selection_count |
|------------------:|:--------|:-------------------------------------------------|:--------------------|------------------:|
|               0.4 | A       | demographics::gradient_boosting_group_oversample | demographics        |                 1 |
|               0.4 | A       | demographics::linear_svm_C1_cwnone               | demographics        |                 1 |
|               0.4 | A       | demographics::logistic_class_weight              | demographics        |                 1 |
|               0.4 | B       | compact_plus_module::logistic                    | compact_plus_module |                 2 |
|               0.4 | B       | compact_plus_module::logistic_group_oversample   | compact_plus_module |                 1 |
|               0.4 | C       | compact_plus_module::rbf_svm_C1_gscale_cwnone    | compact_plus_module |                 2 |
|               0.4 | C       | compact_plus_module::rbf_svm_C1_gscale_group_os  | compact_plus_module |                 1 |

## Inner OOF AP by stage/configuration (top rows)
| stage   | config                                                  |     mean |        std |      min |      max |
|:--------|:--------------------------------------------------------|---------:|-----------:|---------:|---------:|
| A       | demographics::linear_svm_C1_cwnone                      | 0.302481 | 0.0237831  | 0.287691 | 0.329916 |
| A       | demographics::logistic_class_weight                     | 0.300489 | 0.0423156  | 0.257308 | 0.341883 |
| A       | demographics::logistic                                  | 0.297529 | 0.0373686  | 0.259325 | 0.334002 |
| A       | demographics::logistic_group_oversample                 | 0.29545  | 0.0416722  | 0.248927 | 0.329354 |
| A       | demographics::rbf_svm_C0.1_g0.1_cwbalanced              | 0.291911 | 0.0255696  | 0.264668 | 0.315391 |
| A       | demographics::linear_svm_C1_cwbalanced                  | 0.289235 | 0.0402744  | 0.256993 | 0.334379 |
| A       | demographics::gradient_boosting                         | 0.287888 | 0.0421411  | 0.24208  | 0.325009 |
| A       | demographics::rbf_svm_C1_g0.1_cwbalanced                | 0.285107 | 0.0283103  | 0.2601   | 0.315844 |
| A       | demographics::rbf_svm_C0.1_gscale_cwnone                | 0.284536 | 0.0313846  | 0.24872  | 0.307225 |
| A       | demographics::rbf_svm_C1_gscale_cwnone                  | 0.284435 | 0.0304473  | 0.249809 | 0.307018 |
| A       | demographics::rbf_svm_C0.1_gscale_cwbalanced            | 0.282893 | 0.0183953  | 0.264547 | 0.301337 |
| A       | demographics::rbf_svm_C0.1_g0.1_cwnone                  | 0.281089 | 0.0269783  | 0.250272 | 0.300442 |
| B       | compact_plus_module::logistic_class_weight              | 0.467013 | 0.0258924  | 0.440548 | 0.492292 |
| B       | compact_plus_module::logistic                           | 0.465615 | 0.0255689  | 0.443869 | 0.493783 |
| B       | compact_plus_module::logistic_group_oversample          | 0.458951 | 0.0246318  | 0.430726 | 0.476104 |
| B       | compact_plus_module::linear_svm_C1_cwnone               | 0.446629 | 0.030052   | 0.41239  | 0.468636 |
| B       | compact_plus_module::linear_svm_C1_cwbalanced           | 0.434303 | 0.0344468  | 0.39612  | 0.463044 |
| B       | compact::rbf_svm_C0.1_gscale_cwbalanced                 | 0.434205 | 0.0154922  | 0.416707 | 0.446175 |
| B       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced     | 0.431525 | 0.0198728  | 0.414183 | 0.45321  |
| B       | compact_plus_module::rbf_svm_C0.1_gscale_cwnone         | 0.428772 | 0.0421049  | 0.40034  | 0.477143 |
| B       | compact_plus_module::rbf_svm_C1_gscale_cwnone           | 0.4287   | 0.0412023  | 0.401417 | 0.476096 |
| B       | compact_plus_module::rbf_svm_C1_gscale_group_os         | 0.421368 | 0.0159789  | 0.403575 | 0.434493 |
| B       | compact::rbf_svm_C1_gscale_cwbalanced                   | 0.418689 | 0.023558   | 0.403191 | 0.445799 |
| B       | compact::rbf_svm_C0.1_g0.1_cwbalanced                   | 0.417184 | 0.0213715  | 0.398856 | 0.440659 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwnone           | 0.526885 | 0.0165584  | 0.508847 | 0.541395 |
| C       | compact_plus_module::logistic                           | 0.506847 | 0.0443033  | 0.456799 | 0.541044 |
| C       | compact_plus_module::rbf_svm_C0.1_gscale_cwnone         | 0.500132 | 0.00938685 | 0.492133 | 0.510466 |
| C       | compact_plus_module::logistic_class_weight              | 0.499987 | 0.0488117  | 0.445925 | 0.540823 |
| C       | compact_plus_module::rbf_svm_C1_gscale_group_os         | 0.499608 | 0.0523006  | 0.445467 | 0.54985  |
| C       | compact_plus_module::rbf_svm_C10_gscale_cwnone          | 0.49394  | 0.0428463  | 0.450129 | 0.535751 |
| C       | compact_plus_module::rbf_svm_C0.1_gscale_cwbalanced     | 0.491119 | 0.0314     | 0.459415 | 0.522206 |
| C       | compact_plus_module::linear_svm_C1_cwnone               | 0.483852 | 0.0178943  | 0.472979 | 0.504505 |
| C       | compact_plus_module::rbf_svm_C1_gscale_cwbalanced       | 0.483195 | 0.0221285  | 0.459645 | 0.503556 |
| C       | compact_plus_module::logistic_group_oversample          | 0.482822 | 0.0589225  | 0.41795  | 0.533022 |
| C       | compact_plus_module::gradient_boosting_group_oversample | 0.477918 | 0.0359949  | 0.452617 | 0.519126 |
| C       | compact_plus_module::rbf_svm_C1_g0.1_cwnone             | 0.475481 | 0.0410386  | 0.450451 | 0.522843 |

## Important limitations
- Only 500 students and a limited number of positive outcomes are available; split-to-split variation is expected. A single split cannot estimate that variation.
- Precision floors are selected in inner OOF predictions, but can fail on outer test splits. Report pass rate across repeats, not only average precision/recall.
- B and C cascade scores use the maximum model score across available snapshots, so students with longer histories have more opportunities to be flagged. This is a deliberate baseline policy to compare later with first-alert/routing policies.
- The label has no event timestamp. This measures eventual labelled-dropout detection, not lead time or intervention effect.
- Oversampling is done at the student-group level inside inner training folds only; validation and outer test students are untouched.
- This is a small controlled benchmark, not a broad hyperparameter search. Model configurations are intentionally conservative.


## Interpretation
- The RBF SVM with per-lesson features was the best Stage C candidate by mean development OOF AP (0.527) versus logistic regression with the same feature family (0.507). The RBF configuration C=1, gamma=scale, without class weighting was consistently strong across the three splits.
- Stage B remained strongest with logistic regression and per-lesson features (mean AP about 0.467). RBF did not improve the early-history stage.
- Stage A's signal remains weak (best mean AP about 0.302), and selected configurations varied across splits.
- The 40% development precision-floor cascade averaged 51.1% test precision and 69.3% recall, flagging 34.7 students per 100. This is a small-sample robustness result, not a performance guarantee.
- Kernel SVM scores are decision margins, not calibrated probabilities.

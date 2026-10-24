# Stage C: Degree-3 polynomial SVM with bagging and boosting

## Objective
Compare a degree-3 polynomial SVM with student-cluster bagging and AdaBoost, while preserving the existing cascade structure, Stage C snapshot data, target, and feature set.

## Experiment design
- Outer split seeds: [17, 42, 73]; each split holds out 20% of students.
- Inner validation: 2-fold StratifiedGroupKFold over development students.
- Stage C rows are the existing C2 rolling plus censored snapshots, labelled with the same eventual-dropout target used by the current kernel benchmark.
- Feature set: compact + per-lesson/module features. Numeric scaling and imputation are fitted within each training fold.
- Student-level evaluation: row-level scores are aggregated by the maximum score per student, matching the existing Stage C evaluation policy.
- Thresholds are selected from 21 development OOF score quantiles only, with a 40% precision floor; test labels are not used for threshold selection. The more granular Stage-C-only threshold grid is used because this script selects one stage threshold rather than jointly optimizing the full A/B/C cascade.
- Bagging uses 21 student-cluster bootstrap models; each bootstrap samples students with replacement and keeps each sampled student's full set of C snapshots together.
- AdaBoost uses 10 SAMME rounds with a degree-3 polynomial SVC base estimator. This is exploratory: a strong kernel SVM is not always an ideal weak learner for boosting.
- SVM and ensemble decision margins are ranking scores, not calibrated probabilities.

## Results summary
| config                                 | family             |   oof_ap_mean |   oof_ap_std |   test_ap_mean |   test_precision_mean |   test_recall_mean |   test_flagged_mean |   test_scored_mean |   precision_floor_pass_rate |   repeats |
|:---------------------------------------|:-------------------|--------------:|-------------:|---------------:|----------------------:|-------------------:|--------------------:|-------------------:|----------------------------:|----------:|
| poly3_svm_bagged_21_student_bootstraps | bagged_poly3_svm   |      0.528164 |    0.0315805 |       0.683631 |              0.423903 |           0.426667 |             25.3333 |            85.3333 |                    0.666667 |         3 |
| logistic_class_weight                  | logistic           |      0.522683 |    0.0129974 |       0.592142 |              0.551282 |           0.346667 |             17.3333 |            85.3333 |                    0.666667 |         3 |
| poly3_svm_class_weighted               | poly3_svm          |      0.503653 |    0.0246023 |       0.697425 |              0.447392 |           0.426667 |             24.6667 |            85.3333 |                    0.333333 |         3 |
| gradient_boosting                      | gradient_boosting  |      0.495222 |    0.0371543 |       0.633431 |              0.403968 |           0.4      |             25      |            85.3333 |                    0.666667 |         3 |
| poly3_svm_unweighted                   | poly3_svm          |      0.493081 |    0.0298337 |       0.606467 |              0.504413 |           0.373333 |             19.6667 |            85.3333 |                    0.666667 |         3 |
| poly3_svm_adaboost_10_rounds           | adaboost_poly3_svm |    nan        |  nan         |     nan        |              0        |           0        |              0      |             0      |                    0        |         3 |

## Boosting diagnostic
AdaBoost was included as an experimental combination, not assumed to work. With a degree-3 SVC base estimator, the base learner was worse than random under the current fold weighting in this run, so scikit-learn correctly refused to continue boosting. This configuration is retained in the output as a failed experiment rather than silently removed. A polynomial SVM is a relatively strong learner and is not necessarily a suitable weak learner for AdaBoost.

## Interpretation notes
- Compare development OOF AP and test AP, but also inspect precision, recall, number flagged, and precision-floor pass rate.
- Outer splits may overlap in students and are robustness diagnostics rather than independent external validation.
- If bagging helps, it suggests variance reduction across student-level resamples. If AdaBoost fails or does not improve, that is useful evidence; do not force an ensemble merely because it is more complex.
- This script evaluates Stage C candidates only. It does not alter or reselect the Stage A/B models or change the cascade routing policy.
- The dataset does not contain a true dropout timestamp, so results concern eventual labelled-dropout detection, not warning lead time.


## Findings from this run
- Highest mean development OOF AP was **poly3_svm_bagged_21_student_bootstraps** at **52.8%**.
- Bagging OOF AP was 52.8%, versus 50.4% for the class-weighted polynomial SVM. However, mean held-out AP was 68.4% for bagging versus 69.7% for the class-weighted SVM.
- The class-weighted polynomial SVM met the selected precision floor on 33% of outer splits; bagging did so on 67%. This argues against declaring bagging a clear winner from three splits.
- Stage C AP is computed only on test students with available C snapshots; students without a C score are treated as not flagged for Stage-C-only precision/recall. This is not a complete A/B/C cascade metric.
- Failed model configurations: poly3_svm_adaboost_10_rounds. See the CSV error column for fold-level details.

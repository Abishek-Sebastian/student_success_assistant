# Eventual-dropout-aligned cascade experiment

## Objective
Train each stage to predict the same student-level eventual-dropout label, then choose stage thresholds jointly to maximize recall subject to a minimum precision constraint.

## Label and feature policy
- A: one row for every student, with demographics/signup-calendar features only and the eventual dropout label.
- B: all observed prefixes k=1 and k=2, including final observed prefixes; the target is eventual dropout, not next-attempt termination.
- C: all observed prefixes k>=3, including final observed prefixes; the target is eventual dropout.
- No attempt count, future prefix length, next-attempt availability, target-reason, or audit-only field is used as a predictor.
- Final observed prefixes are included because the historical eventual outcome is known for training; this does not mean the model knows future activity at inference time.
- Repeated snapshots are grouped by student for CV and holdout. Training weights make each student's snapshots sum to one per stage.
- B/C student scores for end-to-end evaluation are the maximum risk score over available checkpoints, matching the policy 'flag if any checkpoint crosses its threshold'.

## Dataset sizes
| stage   |   rows |   students |   positive_snapshots |   positive_rate |   students_with_checkpoint |   features_before_encoding |
|:--------|-------:|-----------:|---------------------:|----------------:|---------------------------:|---------------------------:|
| A       |    500 |        500 |                  125 |        0.25     |                        500 |                          5 |
| B       |    977 |        499 |                  228 |        0.233367 |                        499 |                         19 |
| C       |   1476 |        438 |                  166 |        0.112466 |                        438 |                         19 |

## Stage model selection (development OOF AP)
| stage   | model                |   development_oof_eventual_dropout_ap |   students_scored |   positives_scored |   positive_rate_scored |
|:--------|:---------------------|--------------------------------------:|------------------:|-------------------:|-----------------------:|
| A       | extra_trees_balanced |                              0.344485 |               400 |                100 |               0.25     |
| A       | logistic_unweighted  |                              0.283408 |               400 |                100 |               0.25     |
| A       | logistic_balanced    |                              0.28206  |               400 |                100 |               0.25     |
| A       | dummy_prior          |                              0.226417 |               400 |                100 |               0.25     |
| B       | logistic_balanced    |                              0.451194 |               400 |                100 |               0.25     |
| B       | logistic_unweighted  |                              0.449986 |               400 |                100 |               0.25     |
| B       | extra_trees_balanced |                              0.433641 |               400 |                100 |               0.25     |
| B       | dummy_prior          |                              0.226417 |               400 |                100 |               0.25     |
| C       | extra_trees_balanced |                              0.409178 |               350 |                 65 |               0.185714 |
| C       | logistic_unweighted  |                              0.395098 |               350 |                 65 |               0.185714 |
| C       | logistic_balanced    |                              0.389629 |               350 |                 65 |               0.185714 |
| C       | dummy_prior          |                              0.165917 |               350 |                 65 |               0.185714 |

## End-to-end precision-constrained recall
|   precision_floor | development_feasible   | selected_model_A     | selected_model_B   | selected_model_C     |   threshold_A |   threshold_B |   threshold_C |   development_precision |   development_recall |   development_flagged |   development_tp |   test_precision |   test_recall |   test_f1 |   test_flagged |   test_tp |   test_fp |   test_fn |   test_tn |   test_n |   test_positives |   test_prevalence | test_precision_defined   |   precision_ci_low |   precision_ci_high |   recall_ci_low |   recall_ci_high |
|------------------:|:-----------------------|:---------------------|:-------------------|:---------------------|--------------:|--------------:|--------------:|------------------------:|---------------------:|----------------------:|-----------------:|-----------------:|--------------:|----------:|---------------:|----------:|----------:|----------:|----------:|---------:|-----------------:|------------------:|:-------------------------|-------------------:|--------------------:|----------------:|-----------------:|
|               0.3 | True                   | extra_trees_balanced | logistic_balanced  | extra_trees_balanced |      0.884113 |      0.301725 |      0.513197 |                0.302632 |                 0.92 |                   304 |               92 |         0.266667 |          0.8  | 0.4       |             75 |        20 |        55 |         5 |        20 |      100 |               25 |              0.25 | True                     |           0.219504 |            0.314286 |            0.64 |             0.96 |
|               0.4 | True                   | extra_trees_balanced | logistic_balanced  | extra_trees_balanced |      0.884113 |      0.665257 |      0.617078 |                0.4      |                 0.48 |                   120 |               48 |         0.5      |          0.48 | 0.489796  |             24 |        12 |        12 |        13 |        63 |      100 |               25 |              0.25 | True                     |           0.34349  |            0.666667 |            0.28 |             0.68 |
|               0.5 | True                   | extra_trees_balanced | logistic_balanced  | extra_trees_balanced |      0.884113 |      0.790603 |      0.915981 |                0.54902  |                 0.28 |                    51 |               28 |         0.5      |          0.12 | 0.193548  |              6 |         3 |         3 |        22 |        72 |      100 |               25 |              0.25 | True                     |           0        |            1        |            0    |             0.28 |
|               0.6 | True                   | extra_trees_balanced | logistic_balanced  | extra_trees_balanced |      0.884113 |      0.966452 |      0.915981 |                1        |                 0.03 |                     3 |                3 |         0.5      |          0.04 | 0.0740741 |              2 |         1 |         1 |        24 |        74 |      100 |               25 |              0.25 | True                     |           0        |            1        |            0    |             0.12 |

## Interpretation and limitations
- Thresholds and model choices use development out-of-fold predictions only. The test set is student-level held out and is used once for evaluation.
- Bootstrap intervals resample held-out students stratified by outcome; they quantify test-cohort uncertainty, not all sources of model-selection uncertainty.
- With 500 students and few positive outcomes, results are highly split-sensitive. A precision floor met on development can fail on test.
- The supplied dropout label has no event timestamp. This experiment predicts eventual labelled dropout from observed prefixes, but cannot establish whether alerts precede a true dropout date or whether intervention changes outcomes.
- This is an OR-style alert policy over all observed checkpoints, not a claim that the deployment routing engine has been implemented or validated.
- A student with zero attempts has only the A score; B/C scores are unavailable.

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
| A       | extra_trees_balanced |                              0.351245 |               400 |                100 |               0.25     |
| A       | logistic_unweighted  |                              0.311999 |               400 |                100 |               0.25     |
| A       | logistic_balanced    |                              0.308581 |               400 |                100 |               0.25     |
| A       | dummy_prior          |                              0.25     |               400 |                100 |               0.25     |
| B       | logistic_unweighted  |                              0.466225 |               400 |                100 |               0.25     |
| B       | logistic_balanced    |                              0.464528 |               400 |                100 |               0.25     |
| B       | extra_trees_balanced |                              0.407923 |               400 |                100 |               0.25     |
| B       | dummy_prior          |                              0.25     |               400 |                100 |               0.25     |
| C       | extra_trees_balanced |                              0.41683  |               350 |                 65 |               0.185714 |
| C       | logistic_unweighted  |                              0.391665 |               350 |                 65 |               0.185714 |
| C       | logistic_balanced    |                              0.383057 |               350 |                 65 |               0.185714 |
| C       | dummy_prior          |                              0.177383 |               350 |                 65 |               0.185714 |

## End-to-end precision-constrained recall
|   precision_floor | development_feasible   | selected_model_A     | selected_model_B    | selected_model_C     |   threshold_A |   threshold_B |   threshold_C |   development_precision |   development_recall |   development_flagged |   development_tp |   test_precision |   test_recall |   test_f1 |   test_flagged |   test_tp |   test_fp |   test_fn |   test_tn |   test_n |   test_positives |   test_prevalence | test_precision_defined   |   precision_ci_low |   precision_ci_high |   recall_ci_low |   recall_ci_high |
|------------------:|:-----------------------|:---------------------|:--------------------|:---------------------|--------------:|--------------:|--------------:|------------------------:|---------------------:|----------------------:|-----------------:|-----------------:|--------------:|----------:|---------------:|----------:|----------:|----------:|----------:|---------:|-----------------:|------------------:|:-------------------------|-------------------:|--------------------:|----------------:|-----------------:|
|               0.3 | True                   | extra_trees_balanced | logistic_unweighted | extra_trees_balanced |      0.885566 |      0.130334 |      0.477003 |                0.3      |                 0.9  |                   300 |               90 |         0.28169  |          0.8  |  0.416667 |             71 |        20 |        51 |         5 |        24 |      100 |               25 |              0.25 | True                     |           0.230769 |            0.333333 |            0.64 |             0.96 |
|               0.4 | True                   | extra_trees_balanced | logistic_unweighted | extra_trees_balanced |      0.885566 |      0.327446 |      0.579832 |                0.401235 |                 0.65 |                   162 |               65 |         0.393939 |          0.52 |  0.448276 |             33 |        13 |        20 |        12 |        55 |      100 |               25 |              0.25 | True                     |           0.274943 |            0.529412 |            0.32 |             0.72 |
|               0.5 | True                   | extra_trees_balanced | logistic_unweighted | extra_trees_balanced |      0.885566 |      0.550552 |      0.863215 |                0.52     |                 0.26 |                    50 |               26 |         0.5      |          0.16 |  0.242424 |              8 |         4 |         4 |        21 |        71 |      100 |               25 |              0.25 | True                     |           0.166667 |            0.857589 |            0.04 |             0.32 |
|               0.6 | True                   | extra_trees_balanced | logistic_unweighted | extra_trees_balanced |      0.885566 |      0.941591 |      0.863215 |                1        |                 0.04 |                     4 |                4 |         0.666667 |          0.08 |  0.142857 |              3 |         2 |         1 |        23 |        74 |      100 |               25 |              0.25 | True                     |           0        |            1        |            0    |             0.2  |

## Interpretation and limitations
- Thresholds and model choices use development out-of-fold predictions only. The test set is student-level held out and is used once for evaluation.
- Bootstrap intervals resample held-out students stratified by outcome; they quantify test-cohort uncertainty, not all sources of model-selection uncertainty.
- With 500 students and few positive outcomes, results are highly split-sensitive. A precision floor met on development can fail on test.
- The supplied dropout label has no event timestamp. This experiment predicts eventual labelled dropout from observed prefixes, but cannot establish whether alerts precede a true dropout date or whether intervention changes outcomes.
- This is an OR-style alert policy over all observed checkpoints, not a claim that the deployment routing engine has been implemented or validated.
- A student with zero attempts has only the A score; B/C scores are unavailable.

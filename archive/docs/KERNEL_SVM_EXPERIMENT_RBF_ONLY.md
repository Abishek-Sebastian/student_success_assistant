# Kernel-function experiment: RBF SVM inside the existing A/B/C cascade

## Executive summary

**The RBF kernel is worth keeping for Stage C, but it does not replace the current models for the whole cascade.** The experiment preserved the current dataset, A/B/C stages, eventual-dropout target, and cascade policy. It added a focused SVM benchmark and compared it against the existing logistic-regression and Gradient Boosting controls.

### Main result

| Stage | Best practical finding | Mean inner out-of-fold average precision |
|---|---|---:|
| A: initial screen | Linear SVM was marginally ahead of logistic regression, but the signal remains weak | 30.2% (linear SVM) vs 30.0% (class-weighted logistic) |
| B: early history | Logistic regression with per-lesson features remained strongest | 46.7% (class-weighted logistic) |
| C: longer history | RBF SVM with per-lesson features ranked first | **52.7% (RBF SVM)** vs 50.7% (logistic) |

For Stage C, the RBF SVM configuration `C=1, gamma=scale, class_weight=None` had average precision of 50.9%, 53.0%, and 54.1% across the three development splits. Logistic regression with the same per-lesson feature family had 45.7%, 52.3%, and 54.1%. The RBF improvement is promising but modest and varies by split; it is not evidence that all stages should switch to SVMs.

### End-to-end test results

The cascade used thresholds selected only from grouped development OOF scores with a 40% precision floor.

| Outer split | Test precision | Test recall | Students flagged | True positives | False positives |
|---|---:|---:|---:|---:|---:|
| Seed 17 | 61.3% | 76.0% | 31 | 19 | 12 |
| Seed 42 | 40.5% | 68.0% | 42 | 17 | 25 |
| Seed 73 | 51.6% | 64.0% | 31 | 16 | 15 |
| **Mean** | **51.1%** | **69.3%** | **34.7** | **17.3** | **17.3** |

The development precision constraint was feasible on all three splits, and the test precision was at least 40% in all three. The mean is not a guarantee for future students. These outer splits overlap in students and are robustness checks, not independent external validation.

## What was tested

- **RBF SVM:** `C` in `{0.1, 1, 10}`, `gamma` in `{"scale", 0.1}`, with and without balanced class weights.
- **Linear SVM control:** `C=1`, with and without balanced class weights.
- **RBF + student-group oversampling:** small sensitivity check at `C=1` and `gamma` in `{"scale", 0.1}`. Oversampling was not combined with class weighting in this comparison.
- **Existing controls:** unweighted logistic regression, class-weighted logistic regression, student-group-oversampled logistic regression, Gradient Boosting, and group-oversampled Gradient Boosting.
- **Feature sets:** demographics/signup calendar only for A; compact history features and compact + per-lesson features for B/C. Trends were not included in this focused run to keep runtime bounded.
- **Validation:** three outer student-level 80/20 splits with seeds 17, 42, and 73; two-fold `StratifiedGroupKFold` within each development split. All preprocessing and oversampling were restricted to training folds.
- **Threshold selection:** top three stage configurations by development OOF average precision were combined, then thresholds were selected to maximize recall subject to a 40% development precision floor.

## Why use an RBF kernel?

A linear model separates observations with a linear boundary. The RBF kernel compares similarity between examples and lets an SVM learn a curved decision boundary. It implicitly represents nonlinear relationships in a richer feature space without manually creating every interaction feature. This is the appropriate first test of the hypothesis that dropout and non-dropout examples are not linearly separable.

The SVM uses standardized numeric features and one-hot encoded categorical features. We used `decision_function` margins for average-precision ranking and threshold selection rather than `probability=True`, because SVM probability calibration adds another fitting step and can be unstable with this small, imbalanced dataset. **SVM scores in these outputs are margins, not calibrated dropout probabilities.**

## Interpretation and decision

1. **Keep RBF SVM as the preferred candidate for C**, with `C=1`, `gamma=scale`, and no class weighting as the main starting configuration. The group-oversampled variant was selected in only one of three outer splits, so oversampling is not yet a reliable default for SVM.
2. **Keep logistic regression for B.** RBF SVM did not beat the per-lesson logistic models there.
3. **Do not overinterpret A.** Demographics-only prediction is weak and selected models varied between splits.
4. **Do not claim a proven end-to-end improvement yet.** The candidate pool and feature-set search differ from earlier benchmark runs. The current result supports a stage-specific nonlinear candidate, not a blanket replacement.
5. If time permits one more kernel experiment, test a very small **degree-2 polynomial-kernel** comparison at C only. Otherwise, freeze model search and finish the final report.

## Limitations

- There are only 500 students, with a small number of eventual-dropout positives. Model ranking and thresholds can vary substantially across splits.
- Repeated splits overlap in students; the three runs are not independent external validation.
- The precision floor is enforced on development OOF predictions and can only be checked, not guaranteed, on held-out data.
- The target is eventual dropout. There is no dropout timestamp, so these results do not measure how long before dropout a student is identified or whether intervention changes outcomes.
- B/C student scores are aggregated over available snapshots using the existing maximum-score policy. Students with longer histories may have more chances to cross a threshold.
- Kernel margins are not calibrated probabilities. If calibrated risk probabilities are needed for a product decision, calibration should be evaluated separately using grouped data.

## Reproducibility

Run from the project root:

```bash
python scripts/05_run_kernel_svm_benchmark.py \
  --outer-seeds 17 42 73 \
  --inner-folds 2 \
  --precision-floors 0.4 \
  --top-k 3 \
  --feature-sets compact compact_plus_module
```

Outputs are in `outputs/kernel_svm_benchmark/`:

- `inner_oof_config_ap.csv`: per-split AP for every configuration.
- `inner_oof_ap_summary.csv`: mean, standard deviation, minimum and maximum AP by stage/configuration.
- `selected_stage_configs_by_repeat.csv`: stage configuration selected for each outer split.
- `repeated_outer_cascade_metrics.csv`: per-split end-to-end test metrics.
- `repeated_outer_cascade_summary.csv`: aggregate test metrics.
- `outer_test_student_predictions.csv`: held-out predictions for inspection.
- `KERNEL_SVM_REPORT.md`: generated run summary.

## Archive: configuration search space

| Family | Search | Imbalance handling |
|---|---|---|
| RBF SVM | C = 0.1, 1, 10; gamma = scale, 0.1 | None or `class_weight=balanced` |
| Linear SVM | C = 1 | None or `class_weight=balanced` |
| RBF SVM sensitivity | C = 1; gamma = scale, 0.1 | Student-group oversampling |
| Logistic regression | Existing baseline settings | None, balanced weights, or student-group oversampling |
| Gradient Boosting | Existing conservative settings | None or student-group oversampling |

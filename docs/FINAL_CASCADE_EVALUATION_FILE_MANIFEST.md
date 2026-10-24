# Final cascade evaluation: file manifest

This is a diff-only addition for the current `student_success_assistant` project.
No existing files were modified, deleted, or archived.

## New files

- `scripts/09_run_final_cascade_evaluation.py`
  - Trains the selected fixed A/B/C configuration, chooses stage thresholds using grouped development OOF scores, and reports stagewise plus cumulative end-to-end metrics.
- `outputs/final_cascade_evaluation/stepwise_test_metrics.csv`
  - Per outer split: standalone A/B/C and cumulative A, A+B, A+B+C metrics.
- `outputs/final_cascade_evaluation/stepwise_test_summary.csv`
  - Mean and standard deviation of metrics across the three outer splits.
- `outputs/final_cascade_evaluation/end_to_end_metrics_by_split.csv`
  - End-to-end cascade metrics for each held-out split.
- `outputs/final_cascade_evaluation/end_to_end_summary.json`
  - End-to-end mean metrics across splits.
- `outputs/final_cascade_evaluation/selected_thresholds.csv`
  - Stage thresholds selected on development OOF scores, plus feasibility status.
- `outputs/final_cascade_evaluation/outer_test_student_predictions.csv`
  - Per-student held-out targets, stage scores, individual flags, and cumulative flags.
- `outputs/final_cascade_evaluation/FINAL_CASCADE_EVALUATION.md`
  - Methods, configuration, full metrics tables, and caveats.

## Fixed configuration

- A: class-weighted logistic regression with demographics/signup-calendar features.
- B: class-weighted logistic regression with compact + per-lesson features.
- C: depth-2 XGBoost with student-level positive-class weighting and compact + per-lesson features.

Thresholds are selected from grouped development OOF scores, not from held-out test labels. The outer split seeds are 17, 42, and 73. The 40% precision floor is a development threshold-selection constraint, not a guarantee on test data.

## Reproduction

From the project root:

```bash
python scripts/09_run_final_cascade_evaluation.py \
  --outer-seeds 17 42 73 \
  --inner-folds 2 \
  --precision-floor 0.4 \
  --threshold-quantiles 21
```

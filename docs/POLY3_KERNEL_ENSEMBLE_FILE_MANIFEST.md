# Degree-3 polynomial SVM + bagging/boosting experiment: file manifest

This is an additive Stage C experiment. It does not replace or delete existing scripts, data, model outputs, or archives.

## Added files

- `scripts/task1/08_run_poly3_kernel_ensemble_benchmark.py` — runnable benchmark script. Reuses the feature-preparation and Stage C snapshot-loading logic from `scripts/task1/07_run_kernel_svm_benchmark.py`.
- `outputs/poly3_kernel_ensemble/POLY3_KERNEL_ENSEMBLE_REPORT.md` — run report and interpretation.
- `outputs/poly3_kernel_ensemble/stage_c_poly3_ensemble_oof_ap.csv` — development out-of-fold AP per model and split.
- `outputs/poly3_kernel_ensemble/stage_c_poly3_ensemble_summary.csv` — aggregate development and held-out metrics.
- `outputs/poly3_kernel_ensemble/stage_c_poly3_ensemble_test_metrics.csv` — per-model/per-split metrics, selected OOF threshold, and error status.
- `outputs/poly3_kernel_ensemble/stage_c_poly3_ensemble_test_predictions.csv` — student-level held-out scores and flags.

## Modified file

- `README.md` — corrected the kernel benchmark script reference to `07_run_kernel_svm_benchmark.py` and documented the new script, command, and findings.

## Reproduce

From the repository root:

```bash
python scripts/task1/08_run_poly3_kernel_ensemble_benchmark.py --outer-seeds 17 42 73 --inner-folds 2 --precision-floor 0.4
```

Optional runtime controls include `--bagging-estimators`, `--adaboost-estimators`, and `--threshold-quantiles`.

## Current-run summary

- Highest mean development OOF AP: student-cluster bagging of degree-3 polynomial SVMs (52.1%).
- Mean held-out AP was 68.4% for bagging and 69.7% for the class-weighted polynomial SVM. The class-weighted polynomial SVM met the 40% precision floor on all three outer splits; bagging met it on two of three.
- AdaBoost with the degree-3 SVM base estimator failed in all three splits because the base learner was worse than random under the training-fold weighting. The script records this failure and continues.
- This is a Stage C-only model comparison, not an end-to-end A/B/C cascade evaluation. C AP is computed only for students with available C snapshots; the cascade, features, snapshot data, and labels are unchanged.

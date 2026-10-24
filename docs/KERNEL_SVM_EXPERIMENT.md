# Kernel-function experiments: RBF and degree-2 polynomial SVMs

## Executive summary

We ran two focused kernel experiments while preserving the existing A/B/C cascade, snapshot datasets, eventual-dropout target, and maximum-score cascade policy.

**Recommendation:** retain the RBF SVM as a promising Stage C candidate, and treat the degree-2 polynomial SVM as a near-tie worth recording, not as a clear replacement. Keep the existing Stage A/B selection process unchanged; do not replace the cascade architecture based on these results.

The strongest Stage C development result in the polynomial-inclusive run was a degree-2 polynomial SVM (`C=1`, `gamma=scale`, `coef0=1`) with mean inner out-of-fold average precision (AP) of **52.73%**. The best RBF candidate (`C=1`, `gamma=scale`, no class weighting) achieved **52.69%**. The difference is only **0.04 percentage points**, too small to treat as a meaningful win.

## Results at a glance

### Stage C: kernel comparison

| Configuration | Mean inner OOF AP | Standard deviation | Min | Max | Selection across 3 outer development splits |
|---|---:|---:|---:|---:|---:|
| Polynomial degree 2, C=1, gamma=scale, coef0=1 | **52.73%** | 2.57 pp | 50.93% | 55.67% | 1 of 3 |
| RBF, C=1, gamma=scale, no class weighting | 52.69% | 1.66 pp | 50.88% | 54.14% | 0 of 3 in this run's joint selection |
| Polynomial degree 2, C=0.1, gamma=0.1, coef0=1 | 51.43% | 3.99 pp | 47.69% | 55.63% | 1 of 3 |
| Logistic regression, compact + per-lesson features | 50.68% | 4.43 pp | 45.68% | 54.10% | -- |
| RBF, C=1, gamma=scale, group oversampling | 49.96% | 5.23 pp | 44.55% | 54.99% | 1 of 3 |

AP is a ranking metric; it is not the same as precision at the chosen alert threshold. These results come from development out-of-fold predictions.

### Cascade-level test results in the polynomial-inclusive run

| Outer split | Precision | Recall | Flagged students | True positives | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Seed 17 | 61.3% | 76.0% | 31 | 19 | 12 | 6 |
| Seed 42 | 39.6% | 76.0% | 48 | 19 | 29 | 6 |
| Seed 73 | 51.6% | 64.0% | 31 | 16 | 15 | 9 |
| **Mean** | **50.8%** | **72.0%** | **36.7** | **18.0** | **18.7** | **7.0** |

The 40% development precision floor was feasible in all three splits, but held-out precision met or exceeded 40% in only **2 of 3** splits. The seed-42 result was 39.6%, slightly below the floor. The floor is a development selection constraint, not a guarantee on unseen data.

For comparison, the archived RBF-only run had mean test precision **51.1%**, mean recall **69.3%**, and about **34.7** students flagged per 100; test precision met the 40% floor in all three splits. Adding polynomial candidates therefore raised mean recall in this rerun but also increased alert burden and reduced the floor pass rate. This is not an unambiguous end-to-end improvement.

## Experiment 1: RBF and linear SVMs

### What was tested

- RBF SVM: C in {0.1, 1, 10}, gamma in {`scale`, 0.1}, with and without balanced class weights.
- Linear SVM control: C=1, with and without balanced class weights.
- A small RBF + student-group oversampling sensitivity check.
- Existing logistic regression and Gradient Boosting controls, including student-group oversampling where configured.
- A demographics/signup-calendar feature set for A, and compact + per-lesson features for B/C in this focused run.

### Findings

- Stage A remains weak; linear SVM's mean AP advantage over class-weighted logistic regression was marginal (30.25% vs 30.05%).
- Stage B remained logistic-regression-led; class-weighted logistic regression achieved 46.70% mean AP.
- Stage C's best RBF candidate achieved 52.69% mean AP, a modest improvement over the same-feature logistic model at 50.68% in the RBF-only benchmark's comparison.
- Oversampling was not a reliable default for SVM; the group-oversampled RBF configuration was not consistently selected.

Full outputs for this run are preserved under `archive/outputs/kernel_svm_benchmark_rbf_only/`, with the report at `archive/docs/KERNEL_SVM_EXPERIMENT_RBF_ONLY.md`.

## Experiment 2: degree-2 polynomial kernel at Stage C

### Why

The RBF kernel tests flexible similarity-based nonlinear boundaries. A degree-2 polynomial kernel tests a different family of nonlinear interactions, roughly capturing pairwise feature combinations in its kernel representation. This was deliberately restricted to Stage C to keep the search small and avoid changing the Stage A/B candidate pools.

### Search

- Degree: 2
- C: {0.1, 1, 10}
- gamma: {`scale`, 0.1}
- coef0: 1
- No additional class weighting or oversampling in this polynomial grid
- Existing A/B candidates and all existing RBF/control candidates remained in the comparison.

### Findings

- The best polynomial candidate's mean OOF AP (52.73%) was essentially tied with the best non-weighted RBF candidate (52.69%).
- Polynomial configurations were selected for Stage C in one of the three outer runs for each of two parameter settings; a group-oversampled RBF candidate was selected in the remaining run.
- The polynomial-inclusive cascade run averaged 72.0% recall but flagged 36.7 students per 100 on average, and its test precision fell below 40% on one split.
- We therefore cannot claim a reliable end-to-end gain from adding the polynomial kernel. It is a close Stage C candidate, not a clear winner.

## Validation design and leakage controls

- Three outer 80/20 student-level stratified splits with seeds 17, 42, and 73.
- Two-fold `StratifiedGroupKFold` within each outer development partition.
- All snapshots for a student stay within one fold/split.
- Preprocessing is fit within each inner training fold only.
- Group-level oversampling is restricted to inner training folds; duplicated student copies retain all snapshots but each copy has total sample weight one.
- Model configuration and alert thresholds are selected using development OOF scores only. Held-out test outcomes are reported for diagnostics, not used to select a configuration.
- The stage scores continue to use the project's existing maximum-score-over-available-snapshots policy.

## Interpretation and decision

1. Keep the A/B/C cascade and current snapshot datasets unchanged.
2. For C, keep both RBF (`C=1`, `gamma=scale`, no class weighting) and polynomial degree 2 (`C=1`, `gamma=scale`, `coef0=1`) as near-tied candidates. If a single provisional choice is required, prefer the RBF model for the more stable split-to-split AP in these runs, while recording the polynomial result.
3. Do not change A or B solely because of the kernel experiments. Stage B remains logistic-regression-led, and Stage A has weak signal.
4. Do not treat kernel margins as calibrated dropout probabilities. `decision_function` scores are ranking margins.
5. Do not claim a definitive performance improvement: only 500 students are available, the outer splits overlap in students, and the polynomial-inclusive run missed the 40% test precision floor in one split.
6. Freeze the kernel search after this experiment unless a clear implementation or data issue is found. Use remaining time for final QA and the overall project report.

## Limitations

- Only 500 students and a limited number of eventual-dropout positives are available; metrics and thresholds are sensitive to the split.
- The three outer splits overlap in students and are robustness checks, not independent external validation.
- Precision floors are chosen on development OOF data and are not guarantees for unseen students.
- The target is eventual dropout. No dropout timestamp is available, so the experiment cannot estimate days of warning or intervention effectiveness.
- Maximum score across snapshots gives students with longer histories more opportunities to cross a threshold.
- SVM decision margins are not calibrated probabilities.

## Reproducibility

From the project root:

```bash
python scripts/05_run_kernel_svm_benchmark.py \
  --outer-seeds 17 42 73 \
  --inner-folds 2 \
  --precision-floors 0.4 \
  --top-k 3 \
  --feature-sets compact_plus_module
```

Current polynomial-inclusive outputs are in `outputs/kernel_svm_benchmark/`. The first RBF-only run is preserved under `archive/outputs/kernel_svm_benchmark_rbf_only/`.

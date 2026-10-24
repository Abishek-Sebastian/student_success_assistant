# Student Success Assistant: Modelling, Evaluation, and Cascade Decisions

**Status:** Proposed protocol for the next modelling stage  
**Scope:** Split strategy, preprocessing/normalisation, feature selection, imbalance handling, evaluation, and interpretation of A/B/C snapshot labels.

---

## 1. Snapshot construction audit: current facts

The current builder reports:

| Dataset | Rows | Unique students | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|---:|
| A: initial screening | 500 | 500 | 21 | 479 | 4.2% |
| B: retrospective capped | 478 | 478 | 50 | 428 | 10.5% |
| B: operational after attempt 1 | 478 | 478 | 24 | 454 | 5.0% |
| B: rolling early (k=1,2) | 916 | 478 | 50 | 866 | 5.5% |
| C1: penultimate-only | 374 | 374 | 54 | 320 | 14.4% |
| C2: rolling | 1,038 | 374 | 54 | 984 | 5.2% |

Censored rows:
- B after attempt 1: 21 students have no recorded second attempt; their next-attempt target is unobserved.
- B rolling early: 61 final prefixes at k=1 or k=2 have no subsequent recorded attempt; their next-attempt target is unobserved.
- C2: 438 final observed prefixes have no subsequent recorded attempt; their next-attempt target is unobserved.

The current attempt-count distribution is:

| Recorded attempts | Students |
|---:|---:|
| 0 | 1 |
| 1 | 21 |
| 2 | 40 |
| 3 | 64 |
| 4 | 82 |
| 5 | 98 |
| 6 | 82 |
| 7 | 62 |
| 8 | 35 |
| 9 | 14 |
| 10 | 1 |

These are dataset construction counts, not model performance.

---

## 2. Critical target and routing caveat

The data have an eventual `dropped_out` label but no true dropout timestamp. We use the final recorded attempt as a proxy. Therefore, “dropout after the next attempt” means:

> The next recorded attempt is the final recorded attempt, and the supplied eventual dropout label is positive.

This target can be labelled only when the next attempt is observed. If no next attempt exists, the next-attempt outcome is censored/unobserved; it is not automatically a negative.

### Why this matters for B

The current B datasets represent two different experiments:

- **B retrospective capped:** one row per student, using one visible attempt for students with two total attempts and two visible attempts for students with three or more. This follows the retrospective proposal, but the selected prefix length depends on eventual total attempts. It is a comparison benchmark, not a deployable routing rule as written.
- **B operational after attempt 1:** predicts whether attempt 2 is the final recorded attempt. It excludes 21 one-attempt students from supervised training because their next attempt is not observed; those cases are exported as censored.

This operational B is conditional on an observed next attempt. At live inference, however, the model is called before we know whether the student will make another attempt. It therefore does not estimate the full risk of dropout/no-return by itself.

### Rolling B has now been added

The builder now exports `B_rolling_early` with snapshots after prefixes `k = 1` and `k = 2` whenever a subsequent attempt is observed, plus `B_rolling_early_censored_rows` for final prefixes at those cutoffs with no subsequent attempt. The rolling dataset has 916 rows from 478 students (50 positives, 866 negatives); 61 final-prefix rows are censored. It includes 50 positives across the two cutoffs, not 50 distinct positive students guaranteed to be unique across snapshots.

Keep all B variants for comparison, but do not claim any is the final operational B policy yet. The rolling model is conditional on a next attempt being observed; it does not predict failure to return when no next attempt is observed.

If the product requirement is specifically to predict *whether a student will stop engaging before another attempt*, that needs a separately defined event target (or a survival/time-to-event formulation). Do not silently merge “no next attempt”, “next attempt is the final one”, and “eventual dropout” into one label.

### C-stage comparison

- **C1 penultimate-only:** retrospective benchmark; one row per student with at least four recorded attempts. Not live-triggerable without a separate rule for identifying when the penultimate point has arrived.
- **C2 rolling:** snapshots after prefixes `k >= 3` when a next attempt is observed. This is the operationally meaningful next-attempt formulation. Snapshots without a subsequent attempt are censored and kept separately.

C2 has 1,038 rows but only 374 distinct students. It must not be split randomly by row.

---

## 3. Train / validation / test strategy

### Recommended primary protocol

Use a **student-level locked holdout plus grouped cross-validation on the development portion**.

1. Reserve approximately 20% of unique students as a locked test set.
2. Use the remaining 80% as the development set.
3. On development data, perform 5-fold cross-validation with stratification and grouping by student. All snapshots from one student must stay in the same fold.
4. Select preprocessing, feature sets, models, and class-imbalance strategy using development folds only.
5. Set decision thresholds from development out-of-fold predictions or a dedicated validation procedure, never from test outcomes.
6. Freeze the pipeline and evaluate once on the locked test set.

For A/B/C1, there is one row per student, so stratified student-level splitting is straightforward. For B rolling and C2, assign folds at the student level and then include all of each student's snapshots in that fold.

### Important limitation for A

A has only 21 positives. A 20% test set will contain roughly four positive examples, so test precision/recall and the confusion matrix will be highly unstable. Do not over-interpret a small change in test metrics. Report:
- the exact positive count in each split,
- repeated stratified cross-validation results on the development data,
- the locked test result with uncertainty/caveats,
- and, where practical, student-level bootstrap confidence intervals.

If the final locked test split leaves too few positives for even a minimally useful diagnostic, use repeated stratified cross-validation with out-of-fold predictions as the primary internal estimate and clearly label it as internal validation rather than independent test performance.

### Split rules

- Group by `student_id` for every multi-snapshot dataset.
- Use target labels only to stratify split assignment, never as features.
- Keep all snapshots from a student in exactly one partition.
- Do not split by row, timestamp, or attempt.
- If the product's deployment setting is future cohorts over time and sufficient dates exist, add a temporal holdout as a separate stress test. With this small dataset, do not replace the group split with a time split without checking class counts first.

---

## 4. Normalisation and preprocessing

Preprocessing must live inside a scikit-learn `Pipeline` / `ColumnTransformer` so every transform is fit only on the training fold.

### Numeric features

- Exclude score sentinel `999` before aggregation.
- Keep missing score/time values as missing until the preprocessing pipeline.
- Add missingness indicators for important score/time aggregates and counts.
- Impute numeric missing values with the training-fold median.
- Standardise with `StandardScaler` for logistic regression, SVM, and MLP candidates.
- Optionally compare `RobustScaler` if duration features have influential outliers.
- Do not scale for tree-based models unless there is a specific reason; trees are generally insensitive to monotonic scaling.

### Categorical features

- Impute missing categories using a training-fold-only strategy.
- One-hot encode `age_group`, `country`, `device`, and lesson/module identifiers where appropriate.
- Use `handle_unknown="ignore"` and, if needed, group rare categories using a training-fold-only frequency threshold.
- Avoid ordinal-encoding nominal categories unless their ordering is real and documented.

### Date features

Do not feed raw date strings directly into a model. Derive features with an explicit prediction-time meaning, such as signup month/year or elapsed time since signup when the snapshot timestamp is known. Any elapsed-time feature must use the snapshot time, not a later timestamp. For Snapshot A, if the actual prediction timestamp is unavailable, use only date components that genuinely would have been available at A or omit the date.

### Leakage checks

- Fit imputers, scalers, encoders, feature selectors, and thresholds inside each training fold.
- Never use the held-out next attempt's score, duration, timestamp, lesson, or any aggregate derived from it.
- Keep eventual attempt count, target reason, and target-observability fields out of predictors.
- `refund_requested` remains excluded until its availability timing is verified.

---

## 5. Feature-selection strategy

Do not perform global correlation filtering or feature selection on the full dataset before splitting.

### Recommended staged ablation

1. **Baseline:** student-level features available at the stage plus compact global score/time level features.
2. **Recency:** add last-three means and one weighted mean per measure.
3. **Engagement:** add valid counts, total observed duration, attempt gaps, and history span.
4. **Module exposure:** add per-module attempt counts/shares.
5. **Module performance:** add per-module score/time means.
6. **Trends:** add simple score/time slopes and endpoint changes.
7. **Complexity:** only then test polynomial curvature and multiple alternative weighting schemes.

For A, use student-level features only; attempt-derived predictors must not be included. For B/C, include only features available in the corresponding visible prefix.

### Selection methods to compare

- **Primary:** small hypothesis-driven feature sets and family ablations.
- **Regularisation:** L1 or elastic-net logistic regression, with regularisation tuned inside cross-validation.
- **Model-based ranking:** permutation importance on validation/out-of-fold data, with caution for correlated features.
- **Optional:** univariate filters or mutual information inside each training fold only, not once on the full dataset.

Prefer feature families that improve out-of-fold PR-AUC or the chosen operational metric consistently across folds. When two features are highly correlated, compare them via ablation instead of assuming that both are necessary. Do not choose features by in-sample target correlation alone.

---

## 6. Class imbalance and oversampling

### Default: no oversampling in the first benchmark

Start with:
- unweighted regularised logistic regression,
- class-weighted logistic regression,
- a tree-based model with class weighting where supported,
- and a dummy/prevalence baseline.

Then test oversampling as a controlled ablation if recall remains inadequate.

### Why not start with SMOTE?

- A has only 21 positive examples, so synthetic samples can amplify noise.
- The feature space includes categorical variables and sparse module features; naïve interpolation may create implausible students.
- Oversampling before splitting causes leakage.
- Oversampling changes the effective class prevalence and can worsen probability calibration.

If oversampling is tested, use `RandomOverSampler` or a suitable mixed-data method only inside each training fold. Never oversample validation or test data. Do not use SMOTE on the complete dataset before splitting.

### Class weights and thresholds

- Compare `class_weight="balanced"` or explicitly chosen weights against no weights.
- Choose weights using development folds, not the test set.
- Tune the decision threshold using validation/out-of-fold probabilities based on the actual cost of missed dropouts versus false alarms.
- Evaluate probability calibration separately; class weighting and oversampling can distort raw probabilities.

### C2 repeated-snapshot weighting

In C2, students with long histories contribute more rows. Compare:
1. ordinary snapshot weighting, which treats each decision point equally;
2. student-balanced weighting, where each student's total training weight is approximately equal across their snapshots.

Report snapshot-level performance and student-level summaries separately. The weighting choice should reflect whether the product objective values every decision point equally or every student equally.

---

## 7. Models and metrics

### Small-data model ladder

1. Dummy classifier (prevalence / most-frequent) as a sanity baseline.
2. Regularised logistic regression as the interpretable baseline.
3. Random forest / ExtraTrees or another modest tree ensemble.
4. Gradient-boosted trees if supported and tuned conservatively.
5. SVM or MLP only as secondary experiments; the sample size is small and should not justify a large neural model by default.

Use the same split folds and feature availability for comparable experiments.

### Metrics

Report:
- PR-AUC / average precision as a primary ranking metric under class imbalance,
- precision, recall, F1 and balanced accuracy at a threshold selected without test data,
- confusion matrix and positive/negative support,
- ROC-AUC as a secondary ranking metric,
- calibration / Brier score only with an explicit warning when positive counts are small.

Always report the positive prevalence next to PR-AUC. A model's score should be compared with the no-skill prevalence baseline.

### Stage metrics versus cascade metrics

Report these separately:

**Standalone stage metrics:** evaluate A, B, C1, or C2 on its eligible held-out snapshots.

**End-to-end cascade metrics:** run the complete routing policy in sequence and measure final predictions on the intended population. Report coverage at each stage, the number of students routed forward, the number flagged, and false negatives/false positives. Do not present standalone C performance as end-to-end cascade performance.

---

## 8. Recommended experiment matrix

| Experiment | Features | Imbalance treatment | Purpose |
|---|---|---|---|
| E0 | Dummy baseline | None | Establish prevalence baseline |
| E1 | Compact baseline | None | Establish interpretable reference |
| E2 | Compact baseline | Class weights | Test recall/precision trade-off |
| E3 | Recency additions | Best E1/E2 imbalance setting | Test incremental recency value |
| E4 | Engagement additions | Best current setting | Test duration/gap value |
| E5 | Module exposure + levels | Best current setting | Test module information |
| E6 | Simple trends | Best current setting | Test whether changes add value |
| E7 | Feature selection / L1 | Inside CV | Test compactness and stability |
| E8 | Random oversampling | Training folds only | Test oversampling as an ablation |
| E9 | Threshold and calibration | Development only | Match operating objective |
| E10 | Frozen full pipeline | No further tuning | One locked test evaluation |

Do not change several dimensions at once in the early experiments; otherwise, improvements will be hard to attribute.

---

## 9. Decisions recommended now

1. Use student-level partitions and grouped cross-validation for every dataset.
2. Treat A's test metrics as highly uncertain because there are only 21 positives.
3. Put all preprocessing and feature selection inside cross-validation pipelines.
4. Start without oversampling; compare class weighting first.
5. Use PR-AUC plus threshold-specific precision/recall and balanced accuracy.
6. Compare C1 retrospective and C2 rolling separately.
7. Keep B retrospective and operational B as distinct experiments.
8. Compare B rolling (k=1,2) against the single-cutoff B comparator, keeping student groups intact across folds.
9. Preserve censored rows; do not label a missing next attempt as a negative.
10. Before claiming end-to-end cascade performance, finalise how the system handles students who do not make another attempt and how upstream stage predictions route them.

**Bottom line:** use a conservative, leakage-safe modelling protocol. The most urgent unresolved issue is not scaling or model choice; it is making sure the B-stage target matches the real decision the product needs to make.


## Updated operational objective: maximize recall subject to a precision floor

For intervention targeting, the primary policy objective is to **maximize recall while keeping precision at or above a prespecified floor**. The current benchmark uses 0.50 as a provisional default and reports 0.30, 0.40, and 0.60 sensitivity settings. This floor should ultimately be approved by the project/business owner.

- Do not use a fixed probability cutoff of 0.5 as the operational rule.
- Select each candidate threshold from development out-of-fold predictions only, maximizing recall among thresholds whose development precision meets the floor.
- Evaluate the selected threshold once on the locked student-level test set. Report test precision, recall, TP, FP, FN, and the number flagged. If test precision falls below the floor, the policy has not met the operating requirement.
- Keep average precision as a useful ranking diagnostic, but make precision-constrained recall the decision metric.
- Report whether any feasible operating point exists. If none exists, do not silently relax the precision floor; report that the model cannot satisfy the requirement under the current features/data.
- Sensitivity-test multiple precision floors and report uncertainty because test positives are sparse.

### End-to-end cascade evaluation

The true end-to-end objective is to maximize **student-level cascade recall** subject to a minimum **student-level cascade precision**. Thresholds must be selected jointly by simulating the full A -> B -> C routing policy on development OOF predictions, then frozen before the locked test. Count each student once, even if they have several snapshots. Report TP, FP, FN, precision, recall, number of unique students flagged, and stage-wise coverage/routing.

The baseline now computes an end-to-end policy against the common `dropout_binary_audit_only` eventual-outcome label. Stage-specific models are trained on their existing targets, but stage model selection and joint A/B/C threshold tuning use development out-of-fold scores against the common eventual outcome. The student is counted once and is flagged if any actionable A/B/C checkpoint crosses its selected threshold. B/C use only snapshots with a subsequent attempt observed; censored final prefixes are not treated as actionable next-attempt checkpoints. This is an eventual-dropout identification metric, not a validation of warning timeliness: the data has no true dropout timestamp. The stage-specific training targets still differ, so the next experiment should compare them with models trained at each prefix against the common eventual-outcome target.

### Baseline update (2026-10-23)

The baseline runner now writes:
- `precision_constrained_recall.csv`: thresholds selected from development OOF predictions for precision floors 0.30, 0.40, 0.50, and 0.60, with locked-test results.
- `development_oof_predictions.csv`: per-snapshot OOF scores used for policy selection.
- `locked_test_predictions.csv`: held-out probabilities and predictions at 0.5 and the primary selected precision floor.
- `end_to_end_cascade_precision_recall.csv`: jointly selected A/B/C thresholds and locked-test student-level cascade precision/recall for each precision floor.
- `cascade_stage_model_selection.csv`: development OOF stage model selection against the common eventual-outcome label.
- `cascade_test_predictions_precision_*.csv`: student-level locked-test predictions, stage flags, and first flagging stage.
- `BASELINE_FINDINGS_PRECISION_RECALL.md`: interpretation of stage-specific and end-to-end precision-constrained recall results.

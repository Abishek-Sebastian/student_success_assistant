# Student Success Assistant: Final Modeling Report

**Report date:** 2026-10-23  
**Project state:** uploaded project preserved; modeling work stayed within the existing A/B/C snapshot cascade.  
**Status:** internal model-development results, not production validation.

---

## Executive summary

### Objective
Identify students with an eventual `dropped_out = yes` outcome using the existing three-stage cascade and the information available at each snapshot. The immediate modeling focus was to improve the current pipeline through model comparison, XGBoost hyperparameter tuning, and student-level imbalance handling without redesigning the project.

### Main findings

1. **Per-lesson features are the most useful feature addition tested.** For stage B, the best development out-of-fold average precision (AP) was **0.467** with compact + per-lesson features, compared with **0.421** for compact-only and **0.418** for compact + trends. For stage C, the corresponding best APs were **0.529**, **0.403**, and **0.398**. Keep the current compact + per-lesson feature set for B/C.
2. **XGBoost helps most at stage C, not everywhere.** The strongest mean C-stage configuration was shallow XGBoost with student-level class weighting (AP **0.529**), closely followed by a depth-2, faster-learning-rate XGBoost configuration (AP **0.524**). At B, logistic regression remained better (AP **0.467**) than the best tested XGBoost configuration (about **0.412** with the same module features). At A, class-weighted logistic regression was the strongest mean AP baseline (**0.300**); A remains a weak standalone screen.
3. **Full student-group oversampling did not reliably improve performance.** It was not the consistent winner at A or B. Partial oversampling ratios (positive:negative group targets of 0.5 and 0.75) were also tested on the preferred feature set. They did not beat the best B/C configurations. Student-level positive-class weighting was more promising for C than forcing a 50:50 class balance.
4. **The provisional 40% precision-floor operating point is the best balanced choice from the broad search.** Across three outer student-level splits, mean test precision was **47.6%**, mean recall **73.3%**, and the system flagged **38.7 students per 100** on average. The observed precision was at least 40% in all three splits, but this is not a guarantee for future cohorts.
5. **A lower alert burden is possible at a stricter operating point, at the cost of missed cases.** In the focused imbalance follow-up, the 50% precision-floor setting averaged **59.3% precision**, **40.0% recall**, and **17 students flagged per 100**, meeting the floor in all three splits. This uses the same three outer split seeds, so it is a sensitivity result, not independent external confirmation.
6. **The data cannot establish warning lead time.** The label has no event timestamp. These experiments predict the supplied eventual dropout label from available prefixes; they do not show that an alert precedes dropout by a particular number of days, nor whether an intervention changes outcomes.

### Recommended current configuration

- **Stage A:** use a simple class-weighted logistic model as the conservative starting candidate; treat its output cautiously because demographics/signup-calendar features alone have limited signal.
- **Stage B:** use logistic regression with class weighting as the primary candidate, with compact + per-lesson features.
- **Stage C:** use shallow depth-2 XGBoost with student-level positive-class weighting as the primary boosted candidate; retain the depth-2 medium-learning-rate XGBoost and logistic regression as challengers.
- **Imbalance:** do not make full oversampling the default. Keep student-group oversampling as an experiment/ablation; use the class-weighting option where it helps.
- **Alert burden:** start evaluation with a 0.40 development precision floor for a recall-oriented balance. Keep 0.50 as a lower-burden option when the team cannot contact as many students. Thresholds must be selected on development out-of-fold predictions only.

These are recommended candidates based on the current internal evidence, not a claim that one fixed configuration has been independently validated for deployment. Model selection varied across splits, especially at A and C.

---

## 1. Project scope and retained architecture

This report documents the project in the uploaded `student_success_assistant` state. It does not replace the existing snapshot builder or introduce a new cascade design.

- **A:** initial screen; uses student/enrolment attributes and signup-calendar features only.
- **B:** early history; uses snapshots with one or two visible attempts.
- **C:** longer history; uses snapshots with at least three visible attempts.
- **Policy:** a student is flagged if any available stage's student-level maximum score crosses its selected threshold. A/B/C thresholds are selected jointly on development OOF scores under a precision-floor constraint.
- **Primary target for the latest tuning runs:** the common eventual-dropout label from `dropout_binary_audit_only`, which corresponds to the supplied outcome label. This is distinct from the historical stage-specific next-attempt labels in earlier builder/baseline experiments.

The project remains an eventual-outcome identification task because no reliable dropout timestamp is present.

## 2. Data audit and snapshot construction

### Source data

| Dataset | Rows / students | Notes |
|---|---:|---|
| Students | 500 students | One row per student |
| Quiz attempts | 2,453 rows | Chronological attempt history |
| Outcomes | 500 rows | 125 dropout labels and 375 non-dropout labels |
| Explanations | 724 rows | Available for EDA; not used in the current modeling features |

Additional audit facts preserved in the project notes:
- 499 students have at least one attempt; one student has no recorded attempt rows.
- 21 students have exactly one attempt; 478 have at least two attempts.
- Three score values use the sentinel `999`.
- 95 attempt rows have missing `time_spent_min` values.
- No negative durations or duplicate attempt IDs were found in the documented audit.

### Current model tables

The latest eventual-outcome-aligned script combines actionable rolling snapshots with final observed/censored prefixes and assigns each snapshot the student's eventual outcome label.

| Stage | Model rows | Students represented | Predictor scope |
|---|---:|---:|---|
| A | 500 | 500 (125 eventual-dropout positives) | Demographics/signup calendar only |
| B | 977 | 499 (124 eventual-dropout positives) | Rolling prefixes at k=1 and k=2, including final observed prefixes |
| C | 1,476 | 438 (80 eventual-dropout positives) | Rolling prefixes at k>=3, including final observed prefixes |

The underlying snapshot-builder exports actionable and censored rows separately. In the aligned eventual-outcome benchmark, the two sets are combined because the target is eventual outcome, not whether the next attempt is the final one. This choice must be kept explicit: it does **not** mean the model knows a future attempt exists at inference time.

### Data quality and leakage controls retained

- Attempts are sorted chronologically, with a deterministic attempt-ID tie-breaker.
- Attempt-derived features are computed only from the visible prefix.
- Score sentinel `999` is excluded from score aggregates.
- Missing duration is not replaced with zero.
- Valid-observation counts and missingness features are retained where useful.
- Refund-request status remains excluded until its timing is established.
- Audit-only fields such as eventual attempt count, prefix length, target reason, and next-attempt availability are not predictors.
- `student_id` is used for grouping and joining, not as a model feature.

## 3. Feature engineering and ablation findings

### Feature families in the current benchmark

- **Demographics:** age group, country, device, signup month, signup weekday. Used for A only.
- **Compact history:** valid score counts, score mean/last/last-three mean, valid duration counts and means, missingness fractions, total valid duration, attempt gaps, and history span.
- **Trends:** score and duration slopes.
- **Per-lesson/module features:** attempt count, mean score, and mean duration for lessons L1, L2, and L3.

### Best mean development OOF AP by feature family

| Stage | Compact-only | Compact + trends | Compact + per-lesson | Best feature family |
|---|---:|---:|---:|---|
| A | Not applicable | Not applicable | Not applicable | Demographics/signup calendar only |
| B | 0.421 | 0.418 | **0.467** | Compact + per-lesson |
| C | 0.403 | 0.398 | **0.529** | Compact + per-lesson |

Values are the best mean AP configuration within each feature family across the three development splits. They are student-level OOF ranking metrics; C is scored only for students who have a C-stage history.

### Interpretation

- Per-lesson features add information beyond global summaries and simple slopes.
- Trends did not add consistent value on top of the compact history in this benchmark.
- Keep the per-lesson feature family; do not add a large, sparse module feature matrix without a separate justified experiment.
- A's demographics-only signal is modest, so avoid treating its score as a reliable standalone dropout decision.

## 4. Modeling and imbalance experiments

### Models retained as comparisons

- Logistic regression, unweighted.
- Logistic regression with `class_weight="balanced"`.
- Logistic regression with student-group random oversampling.
- Gradient Boosting baseline.
- XGBoost variants covering tree depth, learning rate, number of trees, minimum child weight, subsampling, column subsampling, L1/L2 regularization, gamma, group-level oversampling, and positive-class weighting.

### Student-level grouping and weighting

All rows belonging to a student remain in one outer split and one inner CV fold. For repeated snapshots, each student's total training weight is normalized to one across their snapshots. Student-group oversampling duplicates the positive student's available snapshot group, rather than sampling individual rows independently. Validation and test students are not oversampled.

### Broad XGBoost grid

The first tuning run tested 11 XGBoost configurations alongside four simpler baselines, across the three feature families for B/C and demographics for A. The XGBoost candidates emphasized shallow trees and regularization because the cohort is small. Configurations included:

| Candidate | Trees | Learning rate | Depth | Minimum child weight | Sampling / regularization |
|---|---:|---:|---:|---:|---|
| `xgb_d1_slow` | 250 | 0.025 | 1 | 5 | subsample .80, columns .85, L2 8, L1 .2 |
| `xgb_d2_slow` | 300 | 0.025 | 2 | 5 | subsample .80, columns .85, L2 8, L1 .2 |
| `xgb_d3_slow` | 300 | 0.025 | 3 | 8 | subsample .80, columns .80, L2 10, L1 .5, gamma .1 |
| `xgb_d2_medium_lr` | 180 | 0.07 | 2 | 3 | subsample .90, columns .90, L2 5 |
| `xgb_d1_more_trees` | 450 | 0.015 | 1 | 3 | subsample .85, columns .90, L2 5 |
| `xgb_d2_strong_reg` | 350 | 0.03 | 2 | 10 | subsample .75, columns .75, L2 15, L1 1, gamma .2 |
| `xgb_d1_group_oversample` | 300 | 0.025 | 1 | 5 | group oversampling to 1:1 positive:negative student groups |
| `xgb_d2_group_oversample` | 300 | 0.025 | 2 | 5 | group oversampling to 1:1 positive:negative student groups |
| `xgb_d3_group_oversample` | 250 | 0.025 | 3 | 8 | group oversampling to 1:1 positive:negative student groups |
| `xgb_d1_group_weight` | 300 | 0.025 | 1 | 5 | `scale_pos_weight` based on the negative:positive student ratio |
| `xgb_d2_group_weight` | 300 | 0.025 | 2 | 5 | `scale_pos_weight` based on the negative:positive student ratio |

All XGBoost models used histogram tree construction, a fixed random seed per fit, and one worker thread for reproducibility and manageable runtime.

### Follow-up imbalance ratio sweep

After the broad comparison identified compact + per-lesson features as the strongest B/C family, a focused run fixed that feature family and compared additional imbalance choices:
- Partial student-group oversampling to target positive:negative group ratios of 0.5 and 0.75.
- XGBoost `scale_pos_weight` values of 1.5 and 2.0.
- Existing full group oversampling and student-ratio weighting for comparison.

This follow-up reused the same three outer split seeds. It is a focused sensitivity analysis, not an independent test cohort.

### Imbalance conclusion

- **Do not default to full oversampling.** It did not consistently improve AP for A/B and did not beat the strongest C-stage class-weighted XGBoost candidate.
- Partial oversampling did not beat the strongest B or C candidates.
- Class-weighted logistic regression was strongest for A/B AP.
- Student-ratio weighting was strongest for C AP in the tested grid.

## 5. Evaluation protocol

- Outer test: 80/20 stratified split at the unique-student level.
- Outer seeds: 17, 42, 73. Each test split contains 100 students, including 25 eventual-dropout positives and 75 negatives.
- Inner development validation: two-fold `StratifiedGroupKFold`, grouped by student.
- Preprocessing and model fitting occur inside each training fold.
- Candidate model selection uses student-level OOF average precision.
- Stage scores use the maximum prediction across available snapshots for that student, matching the current policy of flagging a student if any checkpoint crosses the stage threshold.
- Stage thresholds are jointly selected on development OOF scores to maximize recall subject to a precision floor. Thresholds are never selected from the outer test labels.

**Interpretation caution:** the three outer splits overlap in students. They are robustness checks on this small dataset, not three independent external validations. Repeatedly evaluating alternatives on these same split seeds also means these results should not be described as a pristine, untouched external test.

## 6. Main model-selection results

### Mean student-level development OOF AP

| Stage | Best tested configuration | Mean AP | Across-split SD | Interpretation |
|---|---|---:|---:|---|
| A | Class-weighted logistic, demographics | 0.300 | 0.037 | Weak but best mean A candidate |
| B | Class-weighted logistic, compact + per-lesson | 0.467 | 0.022 | Stronger than tested XGBoost configurations for B |
| C | XGBoost depth 2 + student-ratio weighting, compact + per-lesson | 0.529 | 0.044 | Best mean C candidate; depth-2 medium-LR XGBoost was close at 0.524 |

AP is not accuracy. It measures ranking quality and should be interpreted relative to the prevalence among students scored at each stage. A/B/C do not all have the same eligible population.

### Stage-selection stability at the 0.40 precision floor

| Outer seed | Selected A | Selected B | Selected C | Test precision | Test recall | Flagged / 100 |
|---:|---|---|---|---:|---:|---:|
| 17 | Class-weighted logistic | Class-weighted logistic | XGBoost depth 2, medium LR | 51.4% | 76% | 37 |
| 42 | XGBoost depth 2, slow LR | Logistic | XGBoost depth 1 + student-ratio weighting | 44.2% | 76% | 43 |
| 73 | XGBoost depth 3 + group oversampling | Logistic | Logistic | 47.2% | 68% | 36 |

The selected configuration changes across splits, especially at A and C. This is evidence of selection uncertainty, not a reason to select the configuration that happened to score best on the test set.

## 7. Operating-point comparison

### Broad tuning run

| Development precision floor | Mean test precision | Mean test recall | Mean flagged / 100 | Mean TP | Mean FP | Mean FN | Floor met on test splits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 29.0% | 92.0% | 79.3 | 23.0 | 56.3 | 2.0 | 1 / 3 |
| **0.40** | **47.6%** | **73.3%** | **38.7** | **18.3** | **20.3** | **6.7** | **3 / 3** |
| 0.50 | 56.1% | 42.7% | 20.0 | 10.7 | 9.3 | 14.3 | 2 / 3 |

### Focused imbalance follow-up at a 0.50 floor

| Seed | Test precision | Test recall | Flagged / 100 | TP | FP | FN | Floor met? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 50.0% | 28% | 14 | 7 | 7 | 18 | Yes |
| 42 | 56.5% | 52% | 23 | 13 | 10 | 12 | Yes |
| 73 | 71.4% | 40% | 14 | 10 | 4 | 15 | Yes |
| **Mean** | **59.3%** | **40.0%** | **17.0** | **10.0** | **7.0** | **15.0** | **3 / 3** |

The 0.40 setting is the more balanced operating point in the broad run. The 0.50 setting reduces outreach burden but misses more eventual-dropout students. The 0.30 setting catches most positives but generates many false alerts and failed its intended precision floor in two splits.

A precision floor is a development-time selection constraint, not a guarantee that the same precision will be achieved on future data.

## 8. Final recommendations

1. Keep the existing A/B/C cascade and current eventual-outcome-aligned benchmark as the principal modeling track for this task.
2. Use compact + per-lesson features for B/C. Do not add the trend family by default based on the current ablation results.
3. Use class-weighted logistic regression as the primary A/B benchmark; use XGBoost mainly for C, where the mean OOF AP improvement is supported by this small experiment.
4. Do not apply full oversampling automatically. Preserve the group-level sampler in the code for controlled comparison, but the evidence does not support making it the default.
5. Use the 0.40 precision-floor policy as the current balanced candidate. Consider 0.50 only if outreach capacity is constrained and lower recall is acceptable.
6. Before deployment, freeze one configuration and threshold-selection procedure, train it on the approved development data, and validate it on a genuinely untouched cohort if one becomes available.
7. Do not claim 7/14/30-day dropout prediction, warning lead time, or intervention effectiveness until a defensible event-time label and prospective evaluation are available.

---

# Archive A. Earlier target definitions and snapshot variants

The project retains historical alternatives and their outputs for traceability:

- **A target (`A_target`):** dropout label combined with the historical 0/1-attempt rule; 21 positives and 479 negatives. This is the original stage-specific target in `A_model_table.csv`.
- **B next-attempt targets:** positive when the supplied dropout label is positive and the next observed attempt is the final recorded attempt. Rows with no next attempt are censored/unobserved, not automatically negative.
- **C1 penultimate-only:** retrospective benchmark, not directly triggerable online because the eventual penultimate attempt is not known at inference.
- **C2 rolling next-attempt:** snapshots at prefixes k>=3 where a next attempt is observed, with final prefixes exported separately as censored.
- **Eventual-outcome-aligned experiments:** scripts 03/04 and the current XGBoost tuning join `dropout_binary_audit_only` onto each stage so A/B/C all predict the same eventual label. In this version, final observed prefixes are included with that eventual label.

Do not compare raw precision/recall from these target definitions as if they measured the same prediction task. The common eventual-outcome target is the one used for the XGBoost tuning results in this report.

# Archive B. Existing experiments and retained artifacts

| Artifact | Purpose |
|---|---|
| `outputs/baseline/BASELINE_REPORT.md` | Earlier baseline models, stage-specific target evaluation, and the initial cascade benchmark |
| `outputs/eventual_dropout_aligned/EVENTUAL_DROPOUT_ALIGNED_REPORT.md` | First common eventual-outcome-aligned A/B/C experiment |
| `outputs/repeated_boosting_imbalance/BENCHMARK_REPORT.md` | Earlier one-split comparison of logistic, Gradient Boosting, and full group oversampling |
| `outputs/xgboost_oversampling_tuning/BENCHMARK_REPORT.md` | Broad XGBoost and feature-family search across three outer seeds |
| `outputs/xgboost_sampling_ratio_tuning/BENCHMARK_REPORT.md` | Focused partial-oversampling and positive-class-weight sensitivity run |
| `docs/feature_discovery_insights.md` | Exploratory feature hypotheses and caveats |
| `docs/modeling_evaluation_protocol.md` | Data leakage, grouped validation, and threshold-selection protocol |
| `docs/cascade_plan.md` | Original A/B/C and next-attempt target design specification |
| `data/cascade_snapshots/QA_REPORT.md` | Snapshot construction and leakage QA checks |

# Archive C. Reproducibility

Run from the project root with Python 3.10+ and dependencies in `requirements.txt`:

```bash
python scripts/task1/05_run_xgboost_oversampling_tuning.py \
  --outer-seeds 17 42 73 \
  --inner-folds 2 \
  --precision-floors 0.30 0.40 0.50 \
  --top-k 2 \
  --output-dir outputs/xgboost_oversampling_tuning
```

The focused ratio sweep uses the same seeds and fixes B/C to the preferred compact + per-lesson feature family:

```bash
python scripts/task1/06_run_xgboost_sampling_ratio_tuning.py \
  --outer-seeds 17 42 73 \
  --inner-folds 2 \
  --precision-floors 0.30 0.40 0.50 \
  --top-k 2 \
  --feature-sets compact_plus_module \
  --output-dir outputs/xgboost_sampling_ratio_tuning
```

Important output files:
- `repeated_outer_cascade_summary.csv`: summary metrics across outer splits and precision floors.
- `repeated_outer_cascade_metrics.csv`: per-seed thresholds, selected configurations, and test metrics.
- `inner_oof_ap_summary.csv`: mean/std/min/max development OOF AP by stage/configuration.
- `inner_oof_config_ap.csv`: detailed per-run OOF AP results.
- `selected_config_frequency.csv`: how often configurations were selected by stage and precision floor.
- `selected_stage_configs_by_repeat.csv`: selected stage configurations for each outer split.
- `outer_test_student_predictions.csv`: held-out student predictions and stage scores.

# Archive D. Known limitations

- Only 500 students and 125 eventual-dropout positives are available. A-stage model signal is especially limited.
- Outer test sets contain only 25 positive students each, so a small number of changed predictions materially changes precision and recall.
- The three outer test splits overlap in students and are not independent external cohorts.
- The project data is documented as synthetic; results may not transfer to real student populations.
- C-stage scores apply only to students with enough recorded history to reach C. Its AP is not directly comparable to A without considering eligible-population prevalence.
- Maximum-over-history scoring gives students with more observed snapshots more chances to cross a threshold.
- There is no true dropout date, so calendar-time lead time and short-horizon risk are not measured.
- The supplied outcome label is not a measure of intervention effectiveness; a prospective study is needed to establish whether contacting flagged students improves outcomes.
- Model selection varied by split. The current recommendation should be treated as a candidate pipeline to freeze and validate, not a final production guarantee.

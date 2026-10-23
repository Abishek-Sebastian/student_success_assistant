# Student Success Assistant: Three-Stage Cascading Prediction Plan

**Status:** Design specification for the next implementation step  
**Purpose:** Formalise the three snapshot stages, training-label construction, inference flow, leakage constraints, and two alternative definitions of Snapshot C.

---

## 1. Objective and framing

We want a cascading student-success system with three prediction stages, rather than a separate model for every exact attempt number.

- **Snapshot A** operates at the initial/0–1 occurrence stage and has no attempt-performance history.
- **Snapshot B** uses a short, capped history of at most two visible attempts.
- **Snapshot C** uses a richer history, with at least three visible attempts.

The core prediction question is stage-specific: **given the information available at this snapshot, will the student's next attempt be their final recorded attempt and be associated with the supplied dropout label?**

The dataset has a supplied eventual `dropped_out` label but no explicit dropout timestamp. Therefore, the final recorded attempt is only a proxy for when dropout may have occurred. We cannot claim to know the true dropout date or prove that dropout happened immediately after a particular attempt.

### Terminology

- **Total recorded attempts (`N`)**: number of valid attempt rows available for a student in this historical dataset. This is retrospective metadata, not an inference-time feature.
- **Visible prefix length (`k`)**: number of chronologically earliest attempts allowed in the feature calculation for a snapshot.
- **Next attempt**: attempt `k + 1`, if it exists in the historical data.
- **Next-attempt dropout target**: positive only when the supplied eventual label is dropout and the next attempt is the student's final recorded attempt.
- **Censored / unobserved next step**: no next attempt exists in the data, so the data do not show what happened after the snapshot. Such rows should not automatically be treated as negatives for a next-attempt target.

---

## 2. The three stages

### Snapshot A: Initial screening

**Population:** all 500 students, including the student with no recorded attempts.

**Available predictors:**
- Student-level / enrolment-level attributes that would genuinely be available at inference.
- No score, duration, attempt timestamp, attempt count, last-attempt date, or any other feature derived from future attempt records.
- Do not use eventual attempt count to decide which students enter A or as a predictor.

**Target definition for the proposed A-stage task:**
- `A_dropout = 1` when `dropped_out = 1` and the student has 0 or 1 recorded attempts.
- `A_dropout = 0` when the student demonstrably progressed beyond the A stage (more than 1 recorded attempt), or when the supplied label is non-dropout.
- The target is a retrospective training label derived from the supplied outcome plus the observed history. The eventual attempt count must not be included in A's predictors.

**Inference behaviour:** every new student starts at A. If the model predicts continuation, the student remains eligible for later stages as the required observations become available.

**Important caveat:** a student who has made no attempt cannot literally be evaluated on performance after an attempt. A should be described as *initial / 0–1 occurrence screening*, not as a score-based prediction.

### Snapshot B: Early-performance screening

**Feature history:** cap the visible prefix at two attempts. Features are computed from the allowed prefix only.

For the retrospective, one-snapshot-per-student construction:
- Student with `N = 2`: visible prefix `k = 1`.
- Student with `N = 3`: visible prefix `k = 2`.
- Student with `N >= 4`: visible prefix `k = 2`.

This is the intended capped-history design: B represents the early point with at most two visible attempts, regardless of how long the student's later history eventually became.

**Candidate target:**
- `B_dropout_next = 1` only if `dropped_out = 1` and `N = k + 1`.
- `B_dropout_next = 0` when a next attempt is observed but it is not the final recorded attempt, or when the supplied eventual dropout label is 0.
- If no next attempt is observed, the next-attempt target is unobserved and should not be silently labelled 0.

**Inference behaviour:** use only features computed from the attempts actually observed by B. The model must not receive `N`, future attempt dates, or future attempt outcomes.

**Eligibility nuance:** a retrospective row with one or two visible attempts can be built from historical data. In a live system, B can only be invoked when the required observations are actually available. Do not assume a student who has only one observed attempt has already produced a second attempt.

### Snapshot C: Established-history stage

C uses richer history than B, beginning with at least three visible attempts. There are two alternatives to implement and compare.

#### C1. Fixed-stage / penultimate-only retrospective benchmark

- Construct a single row for students whose observed history has at least four attempts.
- Visible prefix is the student's first `N - 1` attempts; the final recorded attempt is held out as the next attempt.
- Target is `1` only if `dropped_out = 1`; otherwise `0`, because the held-out final attempt is the next attempt in this retrospective construction.
- Students with only 2 or 3 total attempts do not have a next attempt after a three-attempt prefix and cannot supply a conventional C next-attempt example.

**Limitation:** this is an offline retrospective benchmark, not a directly executable routing rule. At inference time, the system does not know which attempt will be the student's penultimate attempt. If this approach is reported, explicitly state that evaluation is restricted to retrospectively identified penultimate snapshots and does not establish performance at every live decision point.

#### C2. Operational rolling C stage

- Construct a snapshot after every observed prefix of length `k >= 3` for which a next attempt is available in the historical record.
- Compute features from attempts `1..k` only.
- The target is `1` only if `dropped_out = 1` and `N = k + 1`; otherwise `0` when the next attempt is observed.
- If the student has no observed next attempt after the prefix, the next-attempt label is censored/unobserved and is excluded from this next-attempt target unless a separate, justified censoring policy is designed.
- During inference, run C after the student has accumulated at least three attempts, then again after each subsequent attempt while the student remains active.

**Advantages:** matches a real decision process and does not require knowing the eventual penultimate attempt.

**Trade-off:** creates multiple snapshots per student. All snapshots from a student must remain in the same train/validation/test partition. Metrics should be reported both at the snapshot level and, where useful, at the student level.

---

## 3. Target-construction truth table

Let:
- `D = dropped_out` (supplied eventual label),
- `N = total recorded attempts` (retrospective label-construction metadata only),
- `k = visible prefix length`,
- `N >= k + 1` mean the next attempt is observed.

For B and rolling C, the intended next-attempt label is:

```python
if N < k + 1:
    target = UNOBSERVED  # no next attempt available
else:
    target = int(D == 1 and N == k + 1)
```

This definition makes a crucial distinction:
- A student may eventually drop out (`D = 1`) but still be a **negative example at an earlier snapshot**, because they go on to make another attempt after the next one.
- A positive label means the next observed attempt is the final recorded attempt and the supplied outcome is dropout.
- A student with `D = 0` is not labelled dropout, even if their last recorded attempt is the one being held out.

For A, use the separate stage-specific definition above, because A's question is whether the student drops out before progressing beyond the 0–1 occurrence stage.

### Illustrative examples

| Total attempts `N` | Supplied `D` | B prefix `k` | B next-attempt target | C fixed after 3 | Rolling C after 3 |
|---:|---:|---:|---|---|---|
| 0 | 1 | N/A | N/A | N/A | N/A |
| 1 | 1 | N/A | N/A | N/A | N/A |
| 2 | 1 | 1 | Positive | N/A | N/A |
| 2 | 0 | 1 | Negative | N/A | N/A |
| 3 | 1 | 2 | Positive | No next attempt after 3; unobserved | No next attempt after 3; unobserved |
| 3 | 0 | 2 | Negative | No next attempt after 3; unobserved | No next attempt after 3; unobserved |
| 4 | 1 | 2 | Negative at B (attempt 3 is not final) | Positive if predicting after attempt 3 | Positive at prefix 3 |
| 5 | 1 | 2 | Negative at B | Negative at prefix 3 (attempt 4 is not final) | Positive at prefix 4 |
| 5 | 0 | 2 | Negative | Negative at prefix 3 | Negative at prefix 3 and 4 |

This table assumes `D` is the final supplied eventual outcome label and `N` is the total number of recorded attempt rows. The labels are proxies because a true dropout timestamp is unavailable.

---

## 4. Inference cascade versus training construction

**Training:** build examples from observed prefixes and derive labels retrospectively. Do not use model predictions to decide which training rows exist.

**Inference:** route a student based on the model prediction and available observed history:
1. Every student starts at A.
2. If A predicts continuation, wait for / use the required early attempt history and evaluate B.
3. If B predicts continuation and the student has at least three observed attempts, evaluate C.
4. Under rolling C, reevaluate after each subsequent attempt.

The phrase “students predicted as non-dropout move to the next stage” describes inference routing. It should not be implemented as a filter that removes training rows based on the predictions of another model.

Report two types of performance separately:
- **Standalone stage performance:** performance on the stage's eligible evaluation snapshots.
- **End-to-end cascade performance:** performance after applying A/B/C routing and the chosen thresholds. Include how many students are routed to each stage and how many are screened out at each stage.

Do not report the standalone C score as if it were the end-to-end cascade score.

---

## 5. Leakage-prevention rules

The following rules are mandatory for all three stages.

1. **No future attempt counts as predictors.** `N`, eventual cohort, or a field derived from final history may be used only for retrospective label construction / audit, never as a model input.
2. **Strict temporal prefix.** Every attempt-derived predictor must use only attempts at or before the snapshot boundary.
3. **Hold out the next attempt.** For next-attempt targets, attempt `k + 1` is the outcome event and cannot contribute score, duration, timestamp-derived features, module exposure, or any aggregate.
4. **No future-dependent cohort selection.** In particular, do not define C training examples merely by knowing the student's eventual total attempts unless the experiment is explicitly described as the retrospective penultimate-only benchmark.
5. **No target in features.** `dropped_out` and any derived target column must be removed from predictor matrices.
6. **Split by student.** If multiple snapshots exist for a student, all snapshots must be assigned to the same train/validation/test fold.
7. **Fit preprocessing on training only.** Imputation, scaling, feature selection, category vocabulary, and thresholds must be learned within the training split.
8. **Freeze schema-defined module vocabulary.** Do not discover module columns using held-out evaluation data. Use a fixed schema or fit the vocabulary on training data.
9. **Treat missing values explicitly.** Missing duration is not zero; invalid score sentinel `999` is not a real score; do not silently turn invalid values into zero.
10. **Audit each snapshot.** Store metadata in a separate audit table: student ID, visible prefix length, prediction boundary, next-attempt availability, retrospective `N`, target-observability flag, and target. Drop all audit-only fields before modelling.

---

## 6. Evaluation questions to compare the two C variants

Compare C1 and C2 rather than selecting one before seeing results.

### C1: Penultimate-only benchmark
- What is performance when evaluated only at the retrospectively identified penultimate point?
- How many students are represented?
- How sensitive are results to the fact that only students with a recorded next attempt can be evaluated?
- State explicitly that this is not a deployable trigger without an external rule for deciding when a snapshot is penultimate.

### C2: Rolling C
- What is performance at every eligible prefix of length 3 or more?
- How does performance vary by visible-history length?
- Are students with many snapshots overweighted in snapshot-level metrics?
- What is end-to-end cascade performance when A and B can screen students out before C?
- Can the system maintain useful calibration and precision/recall at each stage?

### Metrics
Use metrics suitable for class imbalance, such as PR-AUC, precision, recall, F1, balanced accuracy, and confusion matrices. Report the positive-class prevalence and support for each stage. Choose operational thresholds using validation data only. Report test results once after decisions are fixed.

---

## 7. Recommended implementation sequence

1. Implement a pure, auditable snapshot builder that produces A, B, C1, and C2 training tables plus a separate audit table.
2. Add assertions proving each feature row uses only its allowed prefix and that the next attempt is excluded from predictors.
3. Verify target construction manually for representative students with 0, 1, 2, 3, 4, and 5+ attempts and both values of `dropped_out`.
4. Report cohort counts, observed-label counts, censored rows, and positive prevalence per stage before modelling.
5. Only after these checks pass, run feature-selection ablations and model training.

---

## 8. Decisions intentionally left open

- The operational threshold for “continue” at A, B, and C.
- Whether C1 is useful as a retrospective upper-bound / comparison benchmark despite not directly defining a live trigger.
- The final primary metric and acceptable false-negative / false-positive trade-off.
- Whether a separate survival / time-to-event framing is needed if a true dropout timestamp becomes available.

**Current recommendation:** implement both C variants, keep their results and reporting separate, and choose the deployment protocol only after comparing their coverage, performance, calibration, and operational meaning.

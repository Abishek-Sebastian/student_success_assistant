# Task 1 Report: Student Success Assistant

**Status:** Final internal modeling report  
**Scope:** Task 1, student-level eventual-dropout prediction using the existing A/B/C cascade  
**Evaluation date:** October 2026

The objective is to identify students who are likely to have the supplied eventual outcome `dropped_out = yes`, so that a student-success team can prioritize outreach and support. The system should make predictions using only information available at the relevant point in a student's learning journey.

The implementation retains a three-stage cascade:

- **Stage A - initial screen:** student/enrolment and signup-calendar information, before useful attempt history is available.
- **Stage B - early-history screen:** performance and engagement features from prefixes with one or two observed attempts.
- **Stage C - established-history screen:** richer features from prefixes with at least three observed attempts.

This is a retrospective prediction task, not a validated time-to-dropout model. The supplied outcome contains no reliable dropout timestamp, so the work cannot establish how many days before dropout an alert would be raised or whether an intervention changes outcomes.

## 1. Final results and primary evaluation objective

### Primary tracking objective

**The primary operating objective is to maximize recall while keeping precision above a chosen minimum threshold.** This is a constrained threshold-selection problem, not a task where the default 0.5 classification threshold or maximum accuracy is the goal.

- **Primary metric:** recall, because missing a student who eventually drops out is considered costly.
- **Constraint:** precision must remain at or above the selected floor, set to **40% during the final evaluation**.
- **Threshold selection:** thresholds are selected using grouped out-of-fold predictions from the development portion only. Held-out test labels are not used to select thresholds.
- **Fallback:** if the precision floor is infeasible on development predictions, the evaluation code uses its documented best-F1 fallback. A development precision floor is a selection constraint, not a guarantee that every held-out split will meet it.

### Final fixed-candidate cascade: held-out results

| Outer split seed | Precision | Recall | F1 | Students flagged | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 45.2% | 76.0% | 0.567 | 42 | 19 | 23 | 6 |
| 42 | 40.9% | 72.0% | 0.522 | 44 | 18 | 26 | 7 |
| 73 | 40.4% | 84.0% | 0.545 | 52 | 21 | 31 | 4 |
| **Mean** | **42.2%** | **77.3%** | **0.545** | **46.0** | **19.3** | **26.7** | **5.7** |

Mean end-to-end average precision (AP) was **0.558** and mean ROC AUC was **0.758**. These are ranking metrics, not calibrated probabilities. On average, each 100-student held-out cohort contained 25 positive cases; the cascade identified about 19 of them and missed about 6, while flagging about 46 students overall. The false-positive workload remains substantial.

The three outer splits use the same 500-student cohort and overlap in their held-out students. They are robustness checks, **not three independent external validations**.

### Fixed models in the final cascade

| Stage | Selected model | Inputs |
|---|---|---|
| A | Class-weighted logistic regression | Demographics and signup-calendar features |
| B | Class-weighted logistic regression | Compact history + per-lesson features |
| C | Depth-2 XGBoost with student-level positive-class weighting | Compact history + per-lesson features |

The final end-to-end AP/AUC are computed from development-referenced percentile ranks across stages because the raw scores produced by different models are not directly comparable. Do not interpret these combined scores as calibrated dropout probabilities.

### Threshold trade-off observed in the broader tuning runs

The broader threshold-sensitivity search illustrates the trade-off between finding more at-risk students and keeping the flagged group actionable. These are separate tuning results, not the final fixed-candidate cascade metrics above.

| Development precision floor | Mean held-out precision | Mean held-out recall | Mean students flagged per 100 | Floor met |
|---:|---:|---:|---:|---:|
| 30% | 29.0% | 92.0% | 79.3 | 1/3 |
| 40% | 47.6% | 73.3% | 38.7 | 3/3 |
| 50% | 56.1% | 42.7% | 20.0 | 2/3 |

A focused 50% floor follow-up reported 59.3% mean precision, 40.0% mean recall, and 17 flagged students per 100. These experiments reuse the same split seeds and are not independent evidence. They show that stricter precision requirements reduce the number of students flagged but can sharply reduce recall.

## 2. Available data

The source data describe a synthetic online course with three lessons and 500 students.

| Source | Size | Relevant fields / use |
|---|---:|---|
| `students.csv` | 500 students | Student ID, age group, country, signup date, device, refund-request status |
| `quiz_attempts.csv` | 2,453 attempt rows | Student and lesson IDs, score, time spent, attempt timestamp |
| `outcome.csv` | 500 students | Supplied `dropped_out` yes/no outcome |
| `explanations.csv` | 724 rows | Free-text explanations and a label; only 60 labels were populated |
| `lessons.md` | 3 lessons | Lesson summaries and key points |

### Data audit and quality handling

- 125 students have a positive eventual-dropout label and 375 have a negative label.
- 499 students have at least one recorded attempt; one student has no attempt rows.
- 21 students have exactly one attempt; 478 have at least two.
- Three score entries use `999` as an invalid/sentinel value and are excluded from score aggregates.
- 95 attempt rows have missing duration values. Missing duration is not treated as zero.
- Attempts are sorted chronologically, with a deterministic attempt-ID tie-breaker.
- `explanations.csv` is available for analysis but its text and sparsely populated labels are not used as predictors in the final dropout models.
- Refund-request status is excluded from the current modeling features because its timing relative to prediction is not established.

### Snapshot tables used by the eventual-outcome-aligned experiments

| Stage | Snapshot rows | Students represented | Feature scope |
|---|---:|---:|---|
| A | 500 | 500 | Demographics and signup calendar |
| B | 977 | 499 | Rolling prefixes at one or two visible attempts |
| C | 1,476 | 438 | Rolling prefixes with at least three visible attempts |

B and C contain multiple snapshots for some students. All snapshots from a student are kept together during validation, and training weights are normalized so a student with many snapshots does not automatically dominate training.

## 3. Assumptions and evaluation constraints

1. **The supplied outcome is the target.** The latest benchmark trains all stages to predict the same eventual `dropped_out` label. This makes stage results comparable in target definition, but does not mean the system knows the outcome at inference.
2. **No event time is available.** The final recorded attempt is not proof of the true dropout date. The model cannot be evaluated for 7-, 14-, or 30-day risk, warning lead time, or intervention effect.
3. **Only prefix-available features are predictors.** Attempt-derived features use only attempts visible at that snapshot. Future attempt count, future prefix length, next-attempt availability, target-reason fields, and audit-only fields are excluded from model inputs.
4. **Student-level splitting is mandatory.** No student's snapshots may be split between training and validation/test.
5. **Preprocessing and threshold selection are training-only.** Imputation, scaling, encoding, model fitting, and threshold selection occur within the development folds. Held-out test labels are not used to choose thresholds.
6. **The data are small and synthetic.** Results may be unstable and may not transfer to real student populations.
7. **The current alert rule is an OR / maximum-over-history baseline.** A student is flagged if any available stage's maximum snapshot score crosses its selected threshold. Students with more snapshots have more opportunities to cross a threshold.
8. **Precision floors are selection constraints, not guarantees.** A threshold selected to meet a precision floor on development data may not meet it on a test cohort.

An earlier design explored a next-attempt target and censored rows. Those metrics are not directly comparable with the later eventual-outcome-aligned experiments. This report uses the common eventual-dropout target for its principal results and identifies earlier experiments as exploratory where relevant.

## 4. How the cascading system works: data flow, training, and inference

The cascade is intended to use the strongest evidence available for each student at their current stage of participation. It is a **sequence of ordinary classification models**, not one end-to-end time-series model.

### 4.1 Stage routing and information available

1. **Stage A: initial screen.** Every student receives one row containing student/enrolment and signup-calendar features. No attempt-derived predictors are used. This is the only stage available for a student with no useful attempt history.
2. **Stage B: early-history screen.** The model uses features computed from the student's visible prefix of one or two attempts. Features include valid-score summaries, duration summaries and missingness, attempt gaps, and per-lesson counts/aggregates.
3. **Stage C: established-history screen.** Once at least three attempts are visible, richer prefix-derived features are available and the Stage C classifier can score that snapshot.

The snapshot builder sorts attempts by `attempted_at`, with `attempt_id` as a deterministic tie-breaker. Each snapshot's features are calculated only from the prefix available at that point. The next attempt can contribute to label construction in experiments that define a next-attempt target, but it must not contribute to the predictor features for that snapshot.

### 4.2 What is passed between stages?

The stages do **not** pass hidden states or learned embeddings from one model to the next. They operate as a cascade at the **student decision/routing level**:

- A stage receives a feature row for the student at the information level available to that stage and emits a score.
- As more attempts become available, the feature builder creates a new snapshot with the expanded visible history, and the appropriate later-stage model scores it.
- The final alert policy combines available stage/snapshot scores using the configured threshold logic. In the evaluated baseline, a student is flagged if any stage's maximum-over-history score crosses its selected threshold (an OR rule).

This means the cascade combines decisions/scores from separate classifiers; it is not a neural network whose intermediate representation is forwarded from A to B to C. Because scores from different model families are not inherently on the same scale, the end-to-end ranking metrics use development-referenced percentile ranks. These ranks are for ranking/evaluation and are not calibrated probabilities.

### 4.3 Training versus inference

**Training / offline evaluation**

- Historical student and attempt records are used to construct prefix snapshots and labels.
- The supplied eventual `dropped_out` label is the common target for the final A/B/C experiment. The historical number of attempts and whether the final recorded attempt occurred can be used to construct retrospective labels, but these fields are retained as audit/label metadata and excluded from predictors.
- Model fitting, preprocessing, class weighting, and threshold selection occur using development data only.
- For B/C, a student can contribute multiple training snapshots. Training weights are normalized at the student level so students with many snapshots do not automatically dominate.
- All snapshots belonging to one student remain in the same split. This prevents the model from training on one snapshot of a student and being tested on another snapshot of that same student.

**Inference / intended deployment**

- Only fields available at that moment are supplied: student/signup fields for A, and the current chronological attempt prefix for B/C.
- The model must not be given the eventual number of attempts, future attempts, the final-attempt indicator, target reason, or outcome-derived audit fields.
- New attempts update the student's prefix features; they do not reveal future behavior. A model can score the student again when new information arrives.
- Since there is no reliable dropout timestamp, inference produces an eventual-outcome risk ranking under the supplied label definition. It cannot honestly be described as a 7-, 14-, or 30-day dropout forecast.

The current maximum-over-history OR rule is a retrospective evaluation policy. For a production system, a first-alert policy and explicit rules for whether/when to rescore should be specified and validated to avoid giving students with more snapshots more opportunities to be flagged.

## 5. Data splitting and why this is ordinary classification rather than time-series forecasting

### 5.1 Unit of splitting: student, not row or attempt

The independent unit is the **student**. The dataset contains repeated attempt records and multiple feature snapshots for some students, so a random row-level split would leak student-specific information across training and evaluation. Instead, each student's complete set of snapshots is assigned to one side of the split.

The final evaluation uses three outer **stratified 80/20 student-level splits**, with seeds 17, 42, and 73:

- **Development/train portion:** approximately 400 students per outer split. This portion is used for fitting candidate models and generating grouped out-of-fold predictions for threshold selection.
- **Validation role:** two-fold grouped out-of-fold (OOF) evaluation within the development portion. Each development student is scored by a model that did not train on that student's group. These OOF predictions are used to compare/choose operating thresholds while keeping the outer test labels untouched.
- **Held-out test portion:** approximately 100 students per outer split, including about 25 positive cases given the cohort's class balance. This portion is used only for final held-out reporting for that outer split.

This is a nested development/test procedure with OOF validation, **not** a single permanent train/validation/test partition. The three outer splits overlap in students; the mean summarizes robustness across resamplings of the same small cohort, not independent external validation. All preprocessing and fitting that can learn from data must be confined to the appropriate development folds.

### 5.2 Why the current approach is not a time-series model

Although attempts have timestamps and features such as time gaps, the final task is formulated as **ordinary binary classification on snapshots**:

- Each row is a feature vector summarizing information visible at a particular student snapshot.
- The target is the supplied eventual dropout label, shared across the final A/B/C stages.
- The model predicts from that row's current features. It does not model a continuous sequence state, forecast a future timestamp, or estimate a survival curve.
- Attempt timestamps are used to sort attempts and compute prefix-available temporal summary features, such as gap and history-span statistics. Their presence does not by itself make the estimator a time-series model.
- The final outcome lacks a trustworthy event time, so a lead-time target such as “dropout within the next 14 days” cannot be defined reliably from the available data.

The snapshot rows are repeated observations of the same student, so **grouped splitting is still required** even though the estimators are ordinary classifiers. Chronological ordering is used within each student's history to construct valid features, while train/test separation is performed by student rather than by timestamp.

## 6. Methodology and experiments

### 6.1 Exploratory data analysis and snapshot construction

The initial work audited cohort size, class balance, attempt histories, missing durations, invalid score values, and date-related features. A snapshot builder then constructed A/B/C model tables and separate audit metadata. Audit fields were retained for quality checks and target construction but removed from predictor matrices.

### 6.2 Feature engineering and ablation

The B/C feature families compared were:

- **Compact history:** score levels and recency, valid-observation counts, duration summaries, missingness, attempt gaps, and history span.
- **Compact + trends:** compact features plus score/duration slopes.
- **Compact + per-lesson features:** compact features plus lesson-specific attempt counts, score summaries, and duration summaries.

Mean development out-of-fold average precision (AP) from the broad repeated benchmark:

| Stage | Compact-only | Compact + trends | Compact + per-lesson |
|---|---:|---:|---:|
| B | 0.421 | 0.418 | **0.467** |
| C | 0.403 | 0.398 | **0.529** |

**Finding:** per-lesson features were the most useful feature addition tested for B and C. Trend features did not provide a consistent gain over compact features.

### 6.3 Baseline models and class imbalance

The experiments compared unweighted logistic regression, class-weighted logistic regression, student-group random oversampling with logistic regression, and Gradient Boosting. Oversampling was applied at the student-group level within training folds, preserving all snapshots for each sampled student. It was not applied to validation or test students.

**Finding:** full oversampling was not a consistent winner. Class weighting was a more useful candidate for the final models, particularly for B and C. The dataset is small enough that balancing classes aggressively can distort the operating point without improving ranking.

### 6.4 XGBoost tuning

A focused XGBoost grid compared shallow trees, learning rates, regularization, sampling, student-level class weighting, and group-level oversampling. The grid was intentionally conservative given the cohort size. A subsequent sensitivity run tested partial oversampling ratios and alternative positive-class weights while holding the preferred B/C feature family fixed.

Main development OOF AP results:

| Stage | Strong candidate | Mean OOF AP | Interpretation |
|---|---|---:|---|
| A | Class-weighted logistic regression | 0.300 | Modest signal from demographics/signup-calendar features |
| B | Class-weighted logistic regression with per-lesson features | 0.467 | Outperformed the tested XGBoost candidates for B |
| C | Depth-2 XGBoost with student-level positive-class weighting and per-lesson features | 0.529 | Best mean candidate in the broad tested grid; logistic and kernel models remain useful challengers |

The best configuration varied across splits, especially at A and C. Model selection should therefore be treated as uncertain rather than proof that one model family is universally superior.

### 6.5 Kernel SVM experiments

Linear and RBF SVMs were tested alongside logistic and Gradient Boosting controls. A focused degree-2 polynomial SVM grid was included for Stage C. SVM decision-function margins were used for ranking and threshold selection; they are not calibrated dropout probabilities.

The SVMs were competitive in some stage/configuration comparisons, particularly C, but did not establish a clear, stable end-to-end advantage over the simpler selected candidate. The search was deliberately limited rather than exhaustive.

### 6.6 Degree-3 polynomial SVM ensembles

A Stage C-only follow-up compared a degree-3 polynomial SVM, student-cluster bagging with 21 bootstrap models, and AdaBoost with a polynomial SVM base estimator.

| Candidate | Mean development OOF AP | Mean held-out AP | Mean precision | Mean recall | Precision-floor pass rate |
|---|---:|---:|---:|---:|---:|
| Bagged degree-3 SVM | 0.521 | 0.684 | 0.425 | 0.467 | 2/3 |
| Unweighted degree-3 SVM | 0.505 | 0.606 | 0.414 | 0.360 | 2/3 |
| Class-weighted degree-3 SVM | 0.493 | 0.697 | 0.458 | 0.467 | 3/3 |
| Gradient Boosting | 0.475 | 0.633 | 0.428 | 0.387 | 2/3 |
| AdaBoost with degree-3 SVM | Failed | Not available | 0.000 | 0.000 | 0/3 |

The bagged model had the highest mean development AP, but the class-weighted polynomial SVM had higher mean held-out AP and a better precision-floor pass rate. Therefore bagging was not declared the clear winner. AdaBoost could not proceed because the chosen base estimator did not meet the algorithm's weak-learner requirement under the fold weighting. The failed run is retained as an experiment rather than hidden.

These are Stage C-only results and should not be compared directly with full-cascade metrics.

### 6.7 Final fixed-candidate cascade evaluation

The final fixed configuration was evaluated with three outer student-level 80/20 stratified splits (seeds 17, 42, and 73). The splits overlap in students and are robustness checks, not independent external cohorts. Two-fold grouped out-of-fold predictions on each development set were used to select thresholds. Thresholds were jointly selected to maximize recall subject to a 40% development precision floor; where infeasible, the documented fallback selected the best F1 combination.

| Stage | Fixed candidate | Feature set |
|---|---|---|
| A | Class-weighted logistic regression | Demographics/signup calendar |
| B | Class-weighted logistic regression | Compact + per-lesson |
| C | Depth-2 XGBoost with student-level positive-class weighting | Compact + per-lesson |

The end-to-end decision metrics on the held-out cohorts were:

| Outer seed | Precision | Recall | F1 | Students flagged | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 45.2% | 76.0% | 0.567 | 42 | 19 | 23 | 6 |
| 42 | 40.9% | 72.0% | 0.522 | 44 | 18 | 26 | 7 |
| 73 | 40.4% | 84.0% | 0.545 | 52 | 21 | 31 | 4 |
| **Mean** | **42.2%** | **77.3%** | **0.545** | **46.0** | **19.3** | **26.7** | **5.7** |

Mean end-to-end average precision was **0.558**, and mean ROC AUC was **0.758**. These ranking metrics describe how well the composite score orders students; they do not establish calibrated probabilities. The end-to-end AP/AUC use development-referenced percentile ranks across stages because the raw scores from different models are not directly comparable.

The held-out results indicate a recall-oriented system: on average it identified about 19 of the 25 positive students in each 100-student test cohort, while flagging about 46 students. The false-positive burden remains material. The A-only screen was weak; adding B increased recall substantially, and C added further recall, but also increased the total number flagged.

### 6.8 Threshold sensitivity

The broader XGBoost/imbalance benchmark also compared operating points:

| Development precision floor | Mean test precision | Mean test recall | Mean flagged per 100 | Test splits meeting floor |
|---:|---:|---:|---:|---:|
| 0.30 | 29.0% | 92.0% | 79.3 | 1/3 |
| **0.40** | **47.6%** | **73.3%** | **38.7** | **3/3** |
| 0.50 | 56.1% | 42.7% | 20.0 | 2/3 |

A focused imbalance follow-up at the 0.50 floor reported 59.3% mean precision, 40.0% mean recall, and 17 flagged per 100 across the same outer split seeds. Since it reused the same seeds, this is a sensitivity result rather than independent confirmation.

The final fixed-candidate cascade's mean precision (42.2%) differs from the broad-search 40%-floor summary (47.6%) because it uses a fixed A/B/C configuration and a separate threshold-selection run. These numbers describe different experiments and should not be merged into one estimate.

## 7. Results and interpretation

1. **Per-lesson features were the strongest consistent feature improvement** for B and C.
2. **A is a weak standalone screen.** Demographic and signup-calendar features alone provide limited ranking signal. It should not be treated as a decisive exclusion gate without considering the cost of missed students.
3. **Logistic regression is a strong B-stage baseline.** In the tested comparisons, it was more reliable than the boosted candidates for early-history prediction.
4. **Shallow XGBoost is a promising C-stage candidate.** It performed well with per-lesson features and student-level positive-class weighting, but model choice varied across splits.
5. **Full student-level oversampling did not consistently improve results.** Class weighting was generally a more useful default to retain.
6. **The 40% development precision-floor policy is a reasonable provisional operating point** when recall matters, but it is not a promise of 40% precision on future cohorts. The fixed-candidate evaluation achieved 42.2% mean precision and 77.3% mean recall across its three overlapping holdouts.
7. **The alert burden remains high.** The final cascade flagged 46 of every 100 held-out students on average, with approximately 27 false positives per 100. Operational usefulness depends on how many students the support team can contact and the relative cost of missed cases versus unnecessary outreach.
8. **The evidence is preliminary.** With 500 synthetic students and only 25 positives in each test split, a small number of changed predictions materially affects reported metrics.

## 8. Future improvements

### Data and target

- Collect a reliable dropout/inactivity timestamp and define an operational outcome horizon. This would enable true 7/14/30-day or survival-style evaluation.
- Record outreach eligibility, intervention delivery, student response, and subsequent outcomes. Evaluate whether support changes outcomes, not only whether the model predicts labels.
- Audit label quality and clarify what `dropped_out` means operationally.
- Collect larger, more representative cohorts across courses, terms, and student populations.

### Modeling and evaluation

- Freeze one model-selection and threshold procedure before evaluating on a genuinely untouched cohort or a later temporal cohort.
- Compare the current maximum-over-history policy with a first-alert policy that issues one alert at the first eligible checkpoint. Evaluate alert timing by observed attempt count, not calendar lead time unless event timestamps become available.
- Evaluate calibration on a separate validation cohort and consider calibrated probabilities only if they improve decision-making.
- Report precision-recall curves, subgroup performance, confidence intervals, and alert burden alongside the primary precision/recall metrics.
- Assess fairness and error rates across relevant groups where sample sizes permit; avoid overinterpreting very small subgroup counts.
- Investigate whether missingness and engagement features generalize, and conduct ablations with confidence intervals rather than expanding the feature set indiscriminately.
- Use a cost-sensitive or capacity-constrained threshold policy based on the real cost of false positives, false negatives, and available support capacity.

### Deployment and monitoring

- Run a prospective shadow evaluation before using alerts to drive interventions.
- Monitor data drift, label prevalence, precision, recall, alert volume, and subgroup disparities over time.
- Provide human-readable reasons for each alert using observed features, while making clear that these are model associations rather than causal explanations.
- Keep a human support process around alerts; a prediction should prioritize a supportive check-in, not automatically penalize or exclude a student.

## 9. Final assessment

The work establishes a defensible experimental baseline for a three-stage student-success screening system. The clearest modeling result is the value of per-lesson history features for B/C, with class-weighted logistic regression a strong choice for B and shallow class-weighted XGBoost a promising candidate for C. The fixed cascade achieves high recall on the small internal holdouts, but does so by flagging a substantial share of students and accepting many false positives.

The appropriate next step is not to claim production readiness. It is to freeze the protocol, evaluate it on a genuinely new cohort, improve the outcome timestamp and intervention data, and compare first-alert routing against the current maximum-over-history baseline. Until then, this should be described as a promising prototype for prioritizing supportive outreach, not a validated dropout prediction service.

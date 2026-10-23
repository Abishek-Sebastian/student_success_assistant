# Student Success Assistant: Feature Discovery Insights from Notebook 03

**Source:** saved outputs and exploratory analyses in `03_extended_feature_discovery.ipynb`  
**Purpose:** preserve the feature hypotheses, observed patterns, caveats, and recommendations for feature selection and snapshot dataset creation.

> **Interpretation warning:** these are exploratory associations, not causal claims or validated out-of-sample predictive performance. The notebook's original Group B/Group C cohort definitions used eventual total attempt count. Those cohort-specific target associations are preliminary and must be recomputed after the snapshot design is corrected.

---

## 1. Data context and quality rules

The data audit recorded:

- `students.csv`: 500 students, 6 columns.
- `quiz_attempts.csv`: 2,453 attempt rows, 6 columns.
- `outcome.csv`: 500 students, 2 columns.
- `explanations.csv`: 724 rows, 5 columns.
- 499 students have at least one attempt; one student has no attempt rows.
- 21 students have exactly one attempt; 478 have at least two attempts.
- Outcome labels: 375 non-dropouts and 125 dropouts.
- All attempt timestamps were present in the audit.
- No duplicate attempt IDs were found.
- Three score values used sentinel `999`.
- 95 attempt rows had missing `time_spent_min`.
- No negative durations were found in the audited attempt rows.

These counts describe the current data snapshot and should be rechecked by the data-builder notebook rather than treated as immutable.

### Required handling
- Exclude score `999` from all score means, slopes, weighted averages, quantiles, and score comparisons.
- Do not treat missing score or duration as zero.
- Keep valid observation counts and missingness information where useful.
- Exclude non-finite values from aggregate calculations.
- A module with no attempts has zero exposure but missing performance aggregates.
- A module with only one valid score can contribute to a level feature, but not to a within-module trend.
- Use chronologically sorted attempts and ensure all feature windows are restricted to the allowed snapshot prefix.

---

## 2. Highest-priority feature family: score level

The saved Group C quantile plots showed a monotonic descriptive association between score-level features and dropout label.

### Prefix mean score quantiles

| Prefix mean score bin | Non-dropout share | Dropout share |
|---|---:|---:|
| 39.5–63.2 | 75.8% | 24.2% |
| 63.2–70.6 | 81.5% | 18.5% |
| 70.6–79.0 | 89.2% | 10.8% |
| 79.0–100 | 95.7% | 4.3% |

The last-three-attempt mean showed a similar pattern: the lowest quantile had a 26.8% dropout share, while the highest quantile had a 4.3% dropout share.

### Candidates to test first
1. Mean valid score over the visible prefix.
2. Most recent valid score.
3. Mean valid score over the last three global attempt rows, when enough rows exist.
4. One recency-weighted score mean.

### Interpretation and cautions
- Absolute performance level appears more promising than complicated trend descriptors.
- The above bins are exploratory; their cut points must not be used as fixed production thresholds without validation.
- A recent mean and an overall mean are correlated. Compare them in ablation experiments rather than retaining every variant automatically.
- Score availability differs from score validity. Preserve valid score counts so the model can distinguish a genuine low score from insufficient valid score observations.

---

## 3. Duration / time-spent features

The saved Group C duration quantiles also showed a clear descriptive pattern.

| Mean valid time per attempt | Dropout share |
|---|---:|
| 10.8–21.5 min | 29.8% |
| 21.5–24.7 min | 16.1% |
| 24.7–27.9 min | 8.6% |
| 27.9–40.0 min | 3.2% |

The exploratory association table reported `time_total_valid_sum` means of approximately 120.3 minutes for non-dropouts and 72.1 minutes for dropouts.

### Candidates to test first
1. Mean valid duration per observed attempt.
2. Most recent valid duration.
3. Total observed valid duration.
4. Count of attempts with valid duration.
5. Fraction of visible attempts with missing duration.
6. A recent-window duration mean.

### Interpretation and cautions
- Duration deserves a first-class feature family, not just a supporting variable.
- Total observed duration is not the same as true total time spent when some durations are missing.
- Duration associations may reflect engagement, task difficulty, or other confounding factors; they do not establish that longer duration causes better outcomes.
- Missing duration must not be converted to zero.
- Use the visible prefix only: the held-out next attempt's duration and timestamp cannot be predictors.

---

## 4. Recency and weighted means

The notebook explored linear, quadratic, cubic, and exponential recency-weighted means.

### Working hypothesis
Recent attempts may be more representative of a student's current state than older attempts, especially when performance or engagement changes over time.

### Recommended approach
- Start with one simple exponential-weighted mean for score and one for duration.
- Compare each against the corresponding unweighted mean and last-three mean.
- Use the same definition and hyperparameters across train/validation/test.
- If the decay parameter is tuned, tune it using training/validation data only.

### Lower priority for now
- Retaining all linear, quadratic, cubic, and multiple exponential variants together.
- Assuming recency weighting improves performance before running an ablation.

Recency-weighted means should preserve the intended chronological positional weights when some values are missing; they should not accidentally reassign the newest weight to an older valid value merely because a recent value is missing.

---

## 5. Trends and change features

The saved plots showed weaker and less monotonic relationships for slope and change features than for absolute score/time levels.

Features explored included:
- Score slope.
- Time slope.
- First-to-last score change.
- Recent-minus-early score differences.
- Polynomial curvature.
- Recency-window differences.

### Recommendation
Keep a small trend family for comparison:
- Simple score slope where at least two valid scores exist.
- First-to-last score change where both endpoint scores are valid.
- Simple duration slope where at least two valid durations exist.

Defer quadratic/cubic curvature and many overlapping change features until the simple features demonstrate incremental value.

### Important small-history caveat
- At one visible attempt, a trend is undefined.
- At two visible attempts, a slope is defined but is entirely determined by a single transition and may be noisy.
- Do not replace undefined trends with zero without an accompanying validity/count indicator. Zero trend and uncomputable trend are different situations.

---

## 6. Module-specific and cross-module features

The notebook explored module-specific exposure, score/time levels, trends, recent windows, and cross-module comparisons for modules `L1`, `L2`, and `L3`. The Group C module feature matrix contained 167 columns.

The cross-module summaries showed:
- Mean score across attempted modules: dropout share declined from 27.2% in the lowest quantile to 5.4% in the highest.
- Mean duration across modules: dropout share declined from 30.8% in the lowest quantile to 6.6% in the highest.
- Best-minus-worst score and score spread had some association, but the patterns were non-monotonic.

### Candidate module features, staged by complexity

**First module ablation**
- Attempt count per module.
- Share of visible attempts belonging to each module.
- Mean valid score per module.
- Mean valid duration per module.

**Second module ablation**
- Recent score/time per module.
- Within-module score change where at least two valid observations exist.
- Cross-module score differences or spread.
- Cross-module duration differences, only if interpretable and sufficiently supported.

### Cautions
- Do not begin with all 167 module features; sparse columns and correlated summaries increase selection noise.
- Module vocabulary should come from a fixed product/data schema or be learned from training data, not discovered from held-out evaluation records.
- Keep missing performance aggregates distinct from zero exposure.
- Module trends are undefined if there are fewer than two valid observations for that module.
- Recompute all module associations after correcting the snapshot population and label construction.

---

## 7. Missingness and validity indicators

The saved Group C summaries reported average missing/invalid score fractions of approximately 0.1% for non-dropouts and 0.0% for dropouts; time missingness was approximately 4.3% and 2.7%, respectively. Invalid score sentinel values were rare.

### Recommendation
- Preserve valid score count and valid duration count.
- Consider missing-duration fraction as a low-cost candidate.
- Retain explicit invalid-score handling for correctness, even if the sentinel is rare.
- Do not prioritise missingness-only predictors ahead of score level and duration level based on these exploratory results.

The difference in missingness rates is small in these summaries and may not be stable after the cohort definition is corrected.

---

## 8. Attempt gaps and history span

The notebook explored gaps between attempts, duration aggregates, and related history summaries.

These can potentially capture engagement cadence:
- Mean/median gap between observed attempts.
- Most recent inter-attempt gap, if available at the snapshot.
- Span from first to most recent visible attempt.
- Counts of attempts and valid measurements.

### Cautions
- A future gap that requires the held-out next attempt's timestamp is leakage.
- For a single visible attempt, gap and history-span features may be undefined or zero by definition; document each feature's semantics carefully.
- History span and attempt count can be strongly related, so test their incremental value.
- Dates and gaps may be sensitive to logging practices or batch imports, not just student behaviour.

---

## 9. Priority ranking for feature selection

### Tier 1: Compact baseline
- Student-level features known at inference, subject to domain review.
- Visible-prefix attempt count.
- Mean valid score.
- Most recent valid score.
- Mean valid duration.
- Most recent valid duration.
- Valid score count and valid duration count.

### Tier 2: Low-cost temporal/engagement additions
- Last-three score mean where defined.
- Last-three duration mean where defined.
- Total observed valid duration.
- One recency-weighted score mean.
- One recency-weighted duration mean.
- Mean attempt gap / history span where defined.
- Missing-duration fraction.

### Tier 3: Incremental module features
- Per-module exposure.
- Per-module mean valid score and duration.
- A small number of cross-module contrasts.

### Tier 4: Experimental complexity
- Multiple weighting families at once.
- Polynomial curvature.
- Many overlapping trend/window features.
- Dense module-specific trajectory features.

The tiers are an experimental order, not a guarantee that Tier 1 features will always be selected. Use validation performance and stability to make the final choice.

---

## 10. Feature-selection experiment design

Use controlled ablations so each family must demonstrate incremental value.

1. **Baseline:** student-level predictors plus compact global score/time level features.
2. **Recency:** add last-three and one weighted mean per measure.
3. **Engagement:** add total observed duration, valid counts, gaps, and history span.
4. **Module exposure:** add per-module counts/shares.
5. **Module performance:** add per-module score/time summaries.
6. **Trends:** add simple slopes and endpoint changes.
7. **Complexity test:** only then test curvature and additional weighting schemes.

### Evaluation requirements
- Use the same student-level splits for comparable experiments.
- If students have multiple snapshots, keep all snapshots for a student in one fold.
- Fit imputation, scaling, feature selection, and decay-parameter tuning on training/validation only.
- Report class prevalence, support, PR-AUC, precision, recall, F1, balanced accuracy, and confusion matrices where appropriate.
- Compare standalone stage metrics separately from end-to-end cascade metrics.
- Report coverage and sample counts for each snapshot type.
- Recompute feature-target plots after the cohort/label logic is corrected.

---

## 11. Cohort caveat from the original notebook

The original saved cohort table showed:

| Original retrospective cohort | Students | Dropouts | Dropout share |
|---|---:|---:|---:|
| B: 2–3 eventual total attempts | 104 | 50 | 48.1% |
| C: 4+ eventual total attempts | 374 | 54 | 14.4% |
| Combined | 478 | 104 | 21.8% |

These are included for historical context only. They are **not valid final cascade performance results**, and their target rates must not be interpreted as evidence that the cohort itself is a predictor available at inference. The original cohort selection depends on eventual total attempt count, so these associations must be recomputed using the formalised A/B/C snapshot and target definitions.

---

## 12. Recommended next step

Before selecting features, implement a leakage-audited snapshot builder that creates:
- A: all 500 students, initial-stage features and A-specific label.
- B: capped prefix of at most two attempts, with next-attempt target observability explicitly recorded.
- C1: retrospective penultimate-only benchmark.
- C2: rolling C snapshots after every prefix of at least three attempts when the next attempt is observed.
- A separate audit table containing retrospective metadata, never passed to the model.

Then verify prefix boundaries and target labels manually for students with 0, 1, 2, 3, 4, and 5+ attempts. Only after those checks pass should the exploratory feature-target relationships be rerun and feature-selection experiments begin.

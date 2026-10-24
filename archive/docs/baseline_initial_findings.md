# Initial baseline findings

## What was run

Four snapshot datasets were evaluated independently: A, B rolling early, C1 penultimate-only, and C2 rolling. Models were a prevalence dummy, unweighted logistic regression, class-weighted logistic regression, and class-weighted ExtraTrees. A uses demographic/signup-calendar fields only. B/C use a compact set of prefix-available global score/time level, recency, count, and engagement features plus student-level categorical fields.

Preprocessing is fold-local in scikit-learn pipelines. Numeric missing values are median-imputed with missingness indicators; categorical values are most-frequent-imputed and one-hot encoded; logistic regression numeric features are standardised. The 20% locked holdout is assigned by student, stratifying student groups using whether any snapshot is positive. Development evaluation uses five-fold `StratifiedGroupKFold`; all snapshots from a student stay in one fold.

## Key development results

Pooled development out-of-fold average precision (AP) is the primary early comparison. Compare it with the dataset's positive prevalence, which is the no-skill AP reference.

| Dataset | Positive prevalence | Dummy AP | Logistic AP | Class-weighted logistic AP | Balanced ExtraTrees AP |
|---|---:|---:|---:|---:|---:|
| A | 4.25% | 3.99% | 3.91% | 4.49% | 3.10% |
| B rolling early | 5.46% | 5.44% | 10.03% | 9.88% | 9.55% |
| C1 penultimate-only | 14.38% | 14.01% | 48.47% | 46.71% | 44.82% |
| C2 rolling | 5.22% | 5.06% | 13.59% | 11.86% | 16.78% |

Values are from the one fixed development split and its grouped out-of-fold predictions. They are preliminary, not a final model selection.

## Initial interpretation

- **A:** Demographics/signup calendar alone do not show convincing ranking signal in development OOF; the AP values are around or below the prevalence baseline. Only 17 positives are in development, so estimates are noisy. The unusually high ExtraTrees locked-test AP is based on just four positive students and conflicts with its weak development result, so it is not credible evidence of generalisation.
- **B rolling early:** Compact attempt features appear to add ranking signal over prevalence in development OOF, but the locked test AP is only 6.12% for unweighted logistic and 5.98% for balanced logistic versus 5.46% prevalence. This is weak and unstable evidence, not a deployment-ready model.
- **C1:** The strongest development signal appears in the retrospective C1 dataset. This should not be interpreted as proof of live performance: C1 is a retrospective penultimate-only cohort selected using eventual attempt history and has a different target/cohort definition from rolling C2.
- **C2:** Development OOF AP is above prevalence, but fold variability is substantial. Locked-test AP is unusually high for logistic regression (37.08% unweighted, 41.24% balanced) relative to development OOF (13.59% and 11.86%). With only 11 positive test snapshots, this gap is a warning about split sensitivity, not a reliable claim of generalisation.
- At the fixed 0.5 threshold, unweighted logistic regression predicts almost/all negatives for the low-prevalence B/C2 datasets. Class weighting recovers recall but creates many false positives. A later threshold analysis must use development OOF probabilities only and reflect the cost of false alarms versus missed students.

## Limitations to keep visible

1. This is a single locked student holdout plus grouped CV, on a small synthetic dataset. Metrics are highly uncertain, especially for A and B.
2. Snapshot-level C2 metrics weight students with more observed decision points more heavily. Student-balanced training weights and student-level summaries remain to be tested.
3. B rolling and C2 are conditional on an observed next attempt. No-next-attempt rows are censored, not negatives. These experiments do not directly estimate no-return/dropout risk.
4. C1 is a retrospective benchmark, not a directly deployable routing rule.
5. Metrics are stage-specific. No end-to-end cascade routing or coverage evaluation has been performed yet.
6. Test metrics are reported for transparency only. Do not choose features, thresholds, or the final model based on this test set.

## Recommended next experiments

1. Inspect per-fold predictions and error cases, especially C1/C2 positives.
2. Add a compact feature-family ablation: global levels only, then recency, then engagement, then module exposure/performance, then trends.
3. Select an operating threshold from development OOF predictions and report precision/recall trade-offs; leave the locked test untouched.
4. Compare student-balanced snapshot weights for C2.
5. Define a separate no-return/event-time target before claiming a full student dropout warning system or evaluating cascade routing.

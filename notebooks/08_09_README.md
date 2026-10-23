# Rich snapshots and cascaded student-risk policy

Run in order:

1. `08_richer_snapshot_features.ipynb`
   - Rebuilds 7/14/30-day student snapshots from raw event data with richer recency, score trend, activity regularity, recent-window, and lesson-level aggregates.
   - Enforces an event-time cutoff before calculating each feature.
   - Trains ordinary logistic regression, random forest, and histogram gradient boosting.
   - Selects thresholds on validation data for precision floors 0.40/0.50/0.60.
   - Saves features, models, and validation/test probabilities under `outputs/rich_snapshot/`.

2. `09_cascaded_student_risk_policy.ipynb`
   - Uses the same student split and snapshot probabilities.
   - Fits a temporal-branch model using one latest causal landmark per eligible student by day 30, rather than max/any-landmark aggregation.
   - Evaluates route-specific thresholds and test performance under the precision-floor objective.
   - Saves reports under `outputs/cascade/`.

## Cascade policy

- `<7 days`: defer/no alert.
- `7–<14 days`: frozen day-7 snapshot model and its threshold.
- `14–<30 days`: if more than 3 attempts are available by the decision cutoff, use the latest causal temporal landmark; otherwise use the frozen day-14 snapshot model.
- `>=30 days`: frozen day-30 snapshot model and its threshold.

This is routing, not a voting ensemble. At each decision time, exactly one route is active. Do not OR every model prediction by default. If the deployed system accumulates alerts over time, jointly tune stage thresholds on validation for cumulative student-level precision and recall.

## Important caveats

- `outcome.csv` has no dropout timestamp. Metrics describe the supplied label and do not prove that dropout occurs after a snapshot.
- Test precision may fall below the validation precision floor; such a route is not considered operationally successful.
- The chronological split is small and results are exploratory. Prefer rolling-origin evaluation and additional independent cohorts before deployment.
- A frozen day-7/day-14/day-30 snapshot model must receive features from that exact snapshot cutoff. Do not pass later features into an earlier-horizon model.

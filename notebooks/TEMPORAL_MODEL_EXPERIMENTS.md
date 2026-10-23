# Temporal model experiments

New notebooks:

- `06_temporal_gradient_boosting.ipynb`: compares compact, regularized gradient-boosting configurations.
- `07_temporal_small_mlp.ipynb`: compares two small MLPs. The MLP training resamples landmark rows inversely by each student's row count to reduce domination by students with more attempts; this is an approximation, not a substitute for a student-level sequence model.

Both notebooks:
- Keep Model A at attempt-level evaluation.
- Keep Model B at student-level evaluation, using max predicted probability across eligible landmarks (any landmark above threshold flags the student).
- Select thresholds on validation data to maximize recall subject to precision floors of 40%, 50%, and 60%.
- Keep random student split as a diagnostic and chronological cohort split as the primary generalization check.
- Write separate outputs to:
  - `outputs/temporal_boosting/metrics/`
  - `outputs/temporal_boosting/predictions/`
  - `outputs/temporal_small_mlp/metrics/`
  - `outputs/temporal_small_mlp/predictions/`

The notebooks have been syntax-checked but were not fully executed in this environment. Run them from top to bottom in the project environment.

Important limitations:
- The supplied dropout label has no event timestamp.
- The chronological split is by first eligible landmark/cohort and is not a full event-time backtest.
- Model B's max-over-landmarks evaluation is retrospective unless landmarks are restricted to what would have been available at the actual intervention time.

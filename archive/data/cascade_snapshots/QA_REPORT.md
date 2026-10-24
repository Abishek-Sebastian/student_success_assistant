# Cascade Snapshot Builder: QA Report

## Status

**PASS:** dataset construction and programmed assertions completed.

## Dataset counts

| Dataset | Rows | Unique students | Positive | Negative | Unobserved target | Positive rate (observed) |
|---|---:|---:|---:|---:|---:|---:|
| A | 500 | 500 | 21 | 479 | 0 | 0.042 |
| B_retrospective_capped | 478 | 478 | 50 | 428 | 0 | 0.105 |
| B_operational_after_attempt_1 | 478 | 478 | 24 | 454 | 0 | 0.050 |
| B_operational_censored_rows | 21 | 21 | 0 | 0 | 21 | — |
| B_rolling_early | 916 | 478 | 50 | 866 | 0 | 0.055 |
| B_rolling_early_censored_rows | 61 | 61 | 0 | 0 | 61 | — |
| C1_penultimate_only | 374 | 374 | 54 | 320 | 0 | 0.144 |
| C2_rolling | 1038 | 374 | 54 | 984 | 0 | 0.052 |
| C2_censored_rows | 438 | 438 | 0 | 0 | 438 | — |

## Leakage and correctness checks

- All 500 students appear exactly once in A.
- Attempt rows are sorted by `attempted_at`, with `attempt_id` as deterministic tie-breaker.
- Every attempt-derived feature is recomputed from the visible chronological prefix only.
- The next attempt is excluded from feature aggregates in B, C1, and C2.
- Eventual attempt count, prefix length, target reason, and next-attempt availability are audit metadata, not predictors.
- `dropped_out` and the normalized label are excluded from feature columns.
- `refund_requested` is excluded from model features pending timing validation.
- Score sentinel 999 is excluded from score aggregates; missing duration is not replaced by zero.
- B rolling and C2 rows without a historical next attempt are exported separately with unobserved target.

## Interpretation cautions

- B_retrospective_capped is the requested retrospective design but its snapshot time depends on eventual attempt count; it is not a deployable routing rule as written.
- B_operational_after_attempt_1 is included as a strict operational comparator; B_rolling_early adds k=1 and k=2 snapshots.
- B_rolling_early and C2 are conditional next-attempt models, not direct no-return/dropout-time models.
- C1 is retrospective penultimate-only; C2 is rolling and operationally meaningful when each new attempt is observed.
- These are dataset/label audits, not model-performance results.

# Cascade policy notes

The cascade is a routing policy, not a voting ensemble. At each decision time, select exactly one model according to signup age and attempts observed by that time. Do not use an OR over independent model predictions by default; that usually increases false positives.

- <7 days: defer/no alert.
- 7–<14 days: day-7 snapshot model using features frozen at day 7.
- 14–<30 days: if >3 attempts are available by the decision cutoff, use the latest causal temporal landmark; otherwise use day-14 snapshot features frozen at day 14.
- >=30 days: day-30 snapshot model using features frozen at day 30.

Use the latest temporal landmark available by the actual decision time, not the maximum score across all historical/future landmarks. Tune thresholds on validation for the deployed policy. If the deployed policy accumulates alerts over time, tune the thresholds jointly on validation for cumulative student-level precision and recall; route-wise precision floors alone do not guarantee cumulative precision.

The outcome table has no dropout timestamp, so all metrics are against the supplied outcome label and cannot establish prediction of a future event after a cutoff.

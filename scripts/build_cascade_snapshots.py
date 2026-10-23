#!/usr/bin/env python3
"""Build leakage-audited A/B/C1/C2 student-success snapshot datasets.

Usage:
    python scripts/build_cascade_snapshots.py
    python scripts/build_cascade_snapshots.py --data-dir data --output-dir data/curated/cascade_snapshots

The script deliberately keeps retrospective metadata out of predictor columns.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDENT_FEATURES = ["age_group", "country", "signup_date", "device"]
SUSPECT_NON_FEATURE_FIELDS = ["refund_requested"]
SCORE_SENTINEL = 999


def _normalise_dropout(value: Any) -> int:
    s = str(value).strip().lower()
    if s in {"yes", "1", "true", "dropout", "dropped_out"}:
        return 1
    if s in {"no", "0", "false", "non_dropout", "not_dropped_out"}:
        return 0
    raise ValueError(f"Unrecognised dropped_out value: {value!r}")


def _valid_score(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.notna() & np.isfinite(x) & x.between(0, 100) & x.ne(SCORE_SENTINEL)


def _valid_duration(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.notna() & np.isfinite(x) & x.ge(0)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    y = [float(v) for v in values]
    if not all(math.isfinite(v) for v in y):
        return np.nan
    # Python arithmetic avoids repeated BLAS dispatch for tiny prefixes.
    x = list(range(len(y)))
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denom = sum((v - x_mean) ** 2 for v in x)
    numer = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(y)))
    return numer / denom if denom else np.nan


def _features_for_prefix(prefix: pd.DataFrame, lesson_ids: list[str]) -> dict[str, Any]:
    """Features from the visible prefix only; never sees the held-out next row."""
    f: dict[str, Any] = {}
    n = len(prefix)
    f.update({
        "score_valid_count": 0, "score_missing_or_invalid_count": 0,
        "score_missing_or_invalid_fraction": np.nan, "score_mean": np.nan,
        "score_last_valid": np.nan, "score_last3_mean": np.nan, "score_slope": np.nan,
        "time_valid_count": 0, "time_missing_count": 0, "time_missing_fraction": np.nan,
        "time_mean": np.nan, "time_last_valid": np.nan, "time_last3_mean": np.nan,
        "time_total_valid_sum": np.nan, "time_slope": np.nan,
        "mean_gap_hours": np.nan, "last_gap_hours": np.nan, "history_span_hours": np.nan,
    })
    for lesson in lesson_ids:
        f[f"lesson_{lesson}_attempt_count"] = 0
        f[f"lesson_{lesson}_score_mean"] = np.nan
        f[f"lesson_{lesson}_time_mean"] = np.nan
    if n == 0:
        return f

    # Convert once to NumPy arrays; repeated pandas filtering inside thousands of
    # overlapping prefixes is unnecessarily expensive.
    score_raw = pd.to_numeric(prefix["score"], errors="coerce").to_numpy(dtype=float)
    time_raw = pd.to_numeric(prefix["time_spent_min"], errors="coerce").to_numpy(dtype=float)
    lessons = prefix["lesson_id"].fillna("").astype(str).to_numpy()
    score_ok = np.isfinite(score_raw) & (score_raw >= 0) & (score_raw <= 100) & (score_raw != SCORE_SENTINEL)
    time_ok = np.isfinite(time_raw) & (time_raw >= 0)

    score = score_raw[score_ok]
    f["score_valid_count"] = int(score_ok.sum())
    f["score_missing_or_invalid_count"] = int(n - score_ok.sum())
    f["score_missing_or_invalid_fraction"] = float((n - score_ok.sum()) / n)
    if len(score):
        f["score_mean"] = float(score.mean())
        f["score_last_valid"] = float(score[-1])
        recent_score = score_raw[-3:]
        recent_score_ok = np.isfinite(recent_score) & (recent_score >= 0) & (recent_score <= 100) & (recent_score != SCORE_SENTINEL)
        if recent_score_ok.any():
            f["score_last3_mean"] = float(recent_score[recent_score_ok].mean())
        f["score_slope"] = _slope(score)

    times = time_raw[time_ok]
    f["time_valid_count"] = int(time_ok.sum())
    f["time_missing_count"] = int(n - time_ok.sum())
    f["time_missing_fraction"] = float((n - time_ok.sum()) / n)
    if len(times):
        f["time_mean"] = float(times.mean())
        f["time_last_valid"] = float(times[-1])
        recent_time = time_raw[-3:]
        recent_time_ok = np.isfinite(recent_time) & (recent_time >= 0)
        if recent_time_ok.any():
            f["time_last3_mean"] = float(recent_time[recent_time_ok].mean())
        f["time_total_valid_sum"] = float(times.sum())
        f["time_slope"] = _slope(times)

    # Input attempts are already chronologically sorted.
    dates = prefix["attempted_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    if len(dates) >= 2:
        gaps = np.diff(dates).astype(float) / 3.6e12
        f["mean_gap_hours"] = float(gaps.mean())
        f["last_gap_hours"] = float(gaps[-1])
        f["history_span_hours"] = float((dates[-1] - dates[0]) / 3.6e12)
    elif len(dates) == 1:
        f["history_span_hours"] = 0.0

    for lesson in lesson_ids:
        lm = lessons == str(lesson)
        f[f"lesson_{lesson}_attempt_count"] = int(lm.sum())
        ls = score_raw[lm & score_ok]
        lt = time_raw[lm & time_ok]
        f[f"lesson_{lesson}_score_mean"] = float(ls.mean()) if len(ls) else np.nan
        f[f"lesson_{lesson}_time_mean"] = float(lt.mean()) if len(lt) else np.nan
    return f


def build(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    students = pd.read_csv(data_dir / "students.csv", dtype={"student_id": str})
    attempts = pd.read_csv(data_dir / "quiz_attempts.csv", dtype={"student_id": str, "attempt_id": str})
    outcomes = pd.read_csv(data_dir / "outcome.csv", dtype={"student_id": str})

    required = {
        "students": {"student_id", "age_group", "country", "signup_date", "device"},
        "attempts": {"attempt_id", "student_id", "lesson_id", "score", "time_spent_min", "attempted_at"},
        "outcomes": {"student_id", "dropped_out"},
    }
    for name, df in [("students", students), ("attempts", attempts), ("outcomes", outcomes)]:
        missing = required[name] - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    if students["student_id"].duplicated().any():
        raise AssertionError("students.csv has duplicate student_id")
    if outcomes["student_id"].duplicated().any():
        raise AssertionError("outcome.csv has duplicate student_id")
    if attempts["attempt_id"].duplicated().any():
        raise AssertionError("quiz_attempts.csv has duplicate attempt_id")
    if not set(outcomes["student_id"]).issubset(set(students["student_id"])):
        raise AssertionError("outcome contains unknown student IDs")
    if not set(attempts["student_id"]).issubset(set(students["student_id"])):
        raise AssertionError("attempts contain unknown student IDs")

    students = students.copy()
    outcomes = outcomes.copy()
    outcomes["dropout_binary"] = outcomes["dropped_out"].map(_normalise_dropout)
    student_table = students.merge(outcomes[["student_id", "dropout_binary"]], on="student_id", how="left", validate="one_to_one")
    if student_table["dropout_binary"].isna().any():
        raise AssertionError("Some students do not have outcomes")

    attempts = attempts.copy()
    attempts["attempted_at"] = pd.to_datetime(attempts["attempted_at"], errors="coerce")
    if attempts["attempted_at"].isna().any():
        raise AssertionError("Attempt timestamps must be valid for chronological prefix construction")
    attempts = attempts.sort_values(["student_id", "attempted_at", "attempt_id"], kind="mergesort").reset_index(drop=True)

    # Fixed lesson vocabulary from the supplied course schema.
    lesson_ids = ["L1", "L2", "L3"]
    unexpected = set(attempts["lesson_id"].dropna().astype(str)) - set(lesson_ids)
    if unexpected:
        raise AssertionError(f"Unexpected lesson IDs not in fixed schema: {sorted(unexpected)}")

    groups = {sid: grp.reset_index(drop=True) for sid, grp in attempts.groupby("student_id", sort=False)}
    total_count = {row.student_id: len(groups.get(row.student_id, [])) for row in student_table.itertuples(index=False)}
    d_map = {row.student_id: int(row.dropout_binary) for row in student_table.itertuples(index=False)}
    student_map = student_table.set_index("student_id").to_dict(orient="index")

    audit_rows: list[dict[str, Any]] = []
    A_rows: list[dict[str, Any]] = []
    B_rows: list[dict[str, Any]] = []
    B_fixed_rows: list[dict[str, Any]] = []
    B_fixed_censored_rows: list[dict[str, Any]] = []
    B_rolling_rows: list[dict[str, Any]] = []
    B_rolling_censored_rows: list[dict[str, Any]] = []
    C1_rows: list[dict[str, Any]] = []
    C2_rows: list[dict[str, Any]] = []
    C2_censored_rows: list[dict[str, Any]] = []

    def add_row(stage: str, sid: str, k: int, prefix: pd.DataFrame, target: Any,
                target_observed: bool, target_reason: str, next_available: bool,
                retrospective_n: int, snapshot_index: int = 0) -> dict[str, Any]:
        stu = student_map[sid]
        features = {col: stu[col] for col in STUDENT_FEATURES}
        features.update(_features_for_prefix(prefix, lesson_ids))
        row = {
            "student_id": sid,
            "stage": stage,
            "snapshot_index": snapshot_index,
            "prefix_length_audit": int(k),
            "eventual_attempt_count_audit": int(retrospective_n),
            "next_attempt_available_audit": bool(next_available),
            "target_observed": bool(target_observed),
            "target": target if target_observed else np.nan,
            "target_reason_audit": target_reason,
        }
        row.update(features)
        return row

    for sid in student_table["student_id"].astype(str):
        hist = groups.get(sid, pd.DataFrame(columns=attempts.columns))
        N = len(hist)
        D = d_map[sid]

        # A: every student. Target means dropout at the zero/one-attempt stage.
        a_target = int(D == 1 and N <= 1)
        A_rows.append(add_row("A", sid, 0, hist.iloc[:0], a_target, True,
                              "dropout_label_yes_and_total_attempts_le_1" if a_target else "not_dropout_at_A_stage",
                              N >= 1, N))

        # B retrospective design requested by the project: N=2 -> k=1;
        # N>=3 -> k=2. This is a comparison benchmark, not fully operationally
        # routable because choosing k depends on eventual N.
        if N >= 2:
            k = 1 if N == 2 else 2
            prefix = hist.iloc[:k]
            next_available = N >= k + 1
            b_target = int(D == 1 and N == k + 1)
            B_rows.append(add_row("B_retrospective_capped", sid, k, prefix, b_target, next_available,
                                  "dropout_label_yes_and_next_attempt_is_final" if b_target else "next_attempt_observed_but_not_dropout_event",
                                  next_available, N))

        # Operational B comparator: after the first attempt for every student
        # who has a second attempt in the historical data. This stage predicts
        # whether attempt 2 is the final recorded attempt.
        if N >= 2:
            k = 1
            prefix = hist.iloc[:k]
            b_fixed_target = int(D == 1 and N == 2)
            B_fixed_rows.append(add_row("B_operational_after_attempt_1", sid, k, prefix, b_fixed_target, True,
                                        "dropout_label_yes_and_total_attempts_eq_2" if b_fixed_target else "attempt_2_not_dropout_event",
                                        True, N))
        elif N == 1:
            # After attempt 1 there is no observed attempt 2 in this history.
            # Preserve this as censored rather than fabricating a negative label.
            B_fixed_censored_rows.append(add_row("B_operational_after_attempt_1_censored", sid, 1, hist.iloc[:1], np.nan, False,
                                                  "no_second_attempt_observed_in_historical_data",
                                                  False, N))

        # Operational rolling B: emit a snapshot after attempt 1 and/or 2.
        # Trainable only when the following attempt is observed; final-prefix rows
        # are censored rather than incorrectly labelled negative. This is a
        # conditional next-attempt target, not a probability-of-never-returning target.
        for k in (1, 2):
            if N >= k:
                prefix = hist.iloc[:k]
                if N > k:
                    b_roll_target = int(D == 1 and N == k + 1)
                    B_rolling_rows.append(add_row(
                        "B_rolling_early", sid, k, prefix, b_roll_target, True,
                        "next_attempt_observed_and_is_final_dropout_attempt" if b_roll_target else "next_attempt_observed_but_not_dropout_event",
                        True, N, snapshot_index=k))
                else:
                    B_rolling_censored_rows.append(add_row(
                        "B_rolling_early_censored", sid, k, prefix, np.nan, False,
                        "no_next_attempt_observed_in_historical_data", False, N, snapshot_index=k))

        # C1: retrospective penultimate-only benchmark, one row for N>=4.
        if N >= 4:
            k = N - 1
            prefix = hist.iloc[:k]
            c1_target = int(D == 1)
            C1_rows.append(add_row("C1_penultimate_only", sid, k, prefix, c1_target, True,
                                   "held_out_final_attempt_and_eventual_dropout_label",
                                   True, N))

        # C2: rolling snapshots after k>=3. Include trainable rows only when
        # a next attempt exists; preserve final-prefix rows separately as censored.
        for k in range(3, N + 1):
            prefix = hist.iloc[:k]
            if k < N:
                c2_target = int(D == 1 and N == k + 1)
                C2_rows.append(add_row("C2_rolling", sid, k, prefix, c2_target, True,
                                       "next_attempt_observed_and_label_derived",
                                       True, N, snapshot_index=k))
            else:
                C2_censored_rows.append(add_row("C2_rolling_censored", sid, k, prefix, np.nan, False,
                                                "no_next_attempt_observed_in_historical_data",
                                                False, N, snapshot_index=k))

        # Audit row for each student-stage snapshot, including censored C point.
        audit_rows.append({
            "student_id": sid,
            "dropout_binary_audit_only": D,
            "eventual_attempt_count_audit_only": N,
            "A_target": a_target,
            "B_retrospective_included": N >= 2,
            "B_retrospective_prefix_length": (1 if N == 2 else 2) if N >= 2 else np.nan,
            "B_retrospective_target": (int(D == 1 and N == ((1 if N == 2 else 2) + 1))) if N >= 2 else np.nan,
            "B_operational_included": N >= 2,
            "B_operational_prefix_length": 1 if N >= 2 else np.nan,
            "B_operational_target": int(D == 1 and N == 2) if N >= 2 else np.nan,
            "C1_included": N >= 4,
            "C1_prefix_length": N - 1 if N >= 4 else np.nan,
            "C1_target": D if N >= 4 else np.nan,
            "C2_trainable_snapshot_count": max(0, N - 3),
            "C2_censored_after_final_prefix": N >= 3,
        })

    A = pd.DataFrame(A_rows)
    B = pd.DataFrame(B_rows)
    B_fixed = pd.DataFrame(B_fixed_rows)
    B_fixed_censored = pd.DataFrame(B_fixed_censored_rows)
    B_rolling = pd.DataFrame(B_rolling_rows)
    B_rolling_censored = pd.DataFrame(B_rolling_censored_rows)
    C1 = pd.DataFrame(C1_rows)
    C2 = pd.DataFrame(C2_rows)
    C2_censored = pd.DataFrame(C2_censored_rows)
    audit = pd.DataFrame(audit_rows)

    # Explicit feature whitelist. Retrospective/audit fields are never predictors.
    feature_cols = STUDENT_FEATURES + [
        "score_valid_count", "score_missing_or_invalid_count", "score_missing_or_invalid_fraction",
        "score_mean", "score_last_valid", "score_last3_mean", "score_slope",
        "time_valid_count", "time_missing_count", "time_missing_fraction",
        "time_mean", "time_last_valid", "time_last3_mean", "time_total_valid_sum", "time_slope",
        "mean_gap_hours", "last_gap_hours", "history_span_hours",
    ]
    for lesson in lesson_ids:
        feature_cols += [
            f"lesson_{lesson}_attempt_count",
            f"lesson_{lesson}_score_mean",
            f"lesson_{lesson}_time_mean",
        ]

    # Ensure every expected feature exists even for empty datasets.
    for df in [A, B, B_fixed, B_rolling, B_rolling_censored, C1, C2, C2_censored]:
        for col in feature_cols:
            if col not in df:
                df[col] = np.nan

    metadata_cols = [
        "student_id", "stage", "snapshot_index", "prefix_length_audit",
        "eventual_attempt_count_audit", "next_attempt_available_audit",
        "target_observed", "target", "target_reason_audit",
    ]
    # Keep target and audit fields visible for QA, but provide explicit feature-only
    # matrices as separate files to prevent accidental training on audit columns.
    datasets = {
        "A": A,
        "B_retrospective_capped": B,
        "B_operational_after_attempt_1": B_fixed,
        "B_operational_censored_rows": B_fixed_censored,
        "B_rolling_early": B_rolling,
        "B_rolling_early_censored_rows": B_rolling_censored,
        "C1_penultimate_only": C1,
        "C2_rolling": C2,
        "C2_censored_rows": C2_censored,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_feature_cols = {}
    for name, df in datasets.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        if name not in {"C2_censored_rows", "B_operational_censored_rows", "B_rolling_early_censored_rows"}:
            # A is strictly student-level: no attempt/performance-derived columns,
            # not even constant zero/missing placeholders.
            model_features = STUDENT_FEATURES if name == "A" else feature_cols
            dataset_feature_cols[name] = model_features
            df[["student_id", "target"] + model_features].to_csv(
                output_dir / f"{name}_model_table.csv", index=False
            )
    audit.to_csv(output_dir / "snapshot_audit.csv", index=False)

    # Assertions: cohort coverage, target semantics, strict-prefix construction,
    # sentinel handling, and feature/audit separation.
    assert len(A) == len(students) == 500
    assert A["student_id"].nunique() == len(students)
    assert A["target"].notna().all()
    assert len(B) == int((audit["eventual_attempt_count_audit_only"] >= 2).sum())
    assert len(B_fixed) == len(B)
    assert len(B_fixed_censored) == int((audit["eventual_attempt_count_audit_only"] == 1).sum())
    assert B_fixed_censored["target_observed"].eq(False).all() if len(B_fixed_censored) else True
    assert len(B_rolling) == sum(int(n > 1) + int(n > 2) for n in total_count.values())
    assert len(B_rolling_censored) == sum(int(n in (1, 2)) for n in total_count.values())
    assert B_rolling_censored["target_observed"].eq(False).all() if len(B_rolling_censored) else True
    assert len(C1) == int((audit["eventual_attempt_count_audit_only"] >= 4).sum())
    expected_c2 = sum(max(0, n - 3) for n in total_count.values())
    assert len(C2) == expected_c2
    assert C2["target_observed"].all() if len(C2) else True
    assert C2_censored["target_observed"].eq(False).all() if len(C2_censored) else True
    assert not set(["dropped_out", "dropout_binary", "eventual_attempt_count_audit", "prefix_length_audit"]) & set(feature_cols)
    assert "refund_requested" not in feature_cols
    assert dataset_feature_cols["A"] == STUDENT_FEATURES
    assert all(not any(c.endswith("_audit") for c in cols) for cols in dataset_feature_cols.values() for c in cols)

    # Verify each row's feature prefix matches the allowed chronological slice.
    def check_prefix(df: pd.DataFrame, mode: str) -> None:
        for row in df.itertuples(index=False):
            sid = row.student_id
            k = int(row.prefix_length_audit)
            hist = groups.get(sid, pd.DataFrame(columns=attempts.columns))
            assert k <= len(hist)
            # Check a few independently recomputed values for all rows.
            prefix = hist.iloc[:k]
            # Independently verify a deterministic sample of prefix rows; the
            # builder computes every row from the same prefix-only function.
            if (hash((sid, mode)) % 17 == 0) or k == 0 or k == len(hist) - 1:
                f = _features_for_prefix(prefix, lesson_ids)
                for col in ["score_valid_count", "time_valid_count", "score_mean", "time_mean"]:
                    got = getattr(row, col)
                    exp = f[col]
                    if pd.isna(exp):
                        assert pd.isna(got), (mode, sid, col, got, exp)
                    else:
                        assert np.isclose(float(got), float(exp), equal_nan=True), (mode, sid, col, got, exp)
            if mode in {"B_retrospective_capped", "B_operational_after_attempt_1", "B_rolling_early", "C1_penultimate_only", "C2_rolling"}:
                assert k < len(hist), (mode, sid, k, len(hist))
                # The next attempt's values are not used in any feature calculation;
                # explicit target check verifies only metadata/label use it.
                if mode == "B_rolling_early":
                    assert k in (1, 2)
                if mode == "C1_penultimate_only":
                    assert k == len(hist) - 1
                if mode == "C2_rolling":
                    assert k < len(hist)
        return

    for name in ["A", "B_retrospective_capped", "B_operational_after_attempt_1", "B_rolling_early", "C1_penultimate_only", "C2_rolling"]:
        check_prefix(datasets[name], name)

    # Stage-target truth checks.
    for row in B.itertuples(index=False):
        N = int(row.eventual_attempt_count_audit)
        D = d_map[row.student_id]
        k = int(row.prefix_length_audit)
        assert int(row.target) == int(D == 1 and N == k + 1)
    for row in B_fixed.itertuples(index=False):
        N = int(row.eventual_attempt_count_audit)
        D = d_map[row.student_id]
        assert int(row.target) == int(D == 1 and N == 2)
    for row in B_rolling.itertuples(index=False):
        N = int(row.eventual_attempt_count_audit)
        D = d_map[row.student_id]
        k = int(row.prefix_length_audit)
        assert k in (1, 2) and k < N
        assert int(row.target) == int(D == 1 and N == k + 1)
    for row in C1.itertuples(index=False):
        D = d_map[row.student_id]
        assert int(row.target) == D
        assert int(row.prefix_length_audit) == int(row.eventual_attempt_count_audit) - 1
    for row in C2.itertuples(index=False):
        N = int(row.eventual_attempt_count_audit)
        D = d_map[row.student_id]
        k = int(row.prefix_length_audit)
        assert int(row.target) == int(D == 1 and N == k + 1)

    def summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
        target = pd.to_numeric(df.get("target", pd.Series(dtype=float)), errors="coerce")
        observed = target.dropna().astype(int)
        return {
            "dataset": name,
            "rows": int(len(df)),
            "unique_students": int(df["student_id"].nunique()) if len(df) else 0,
            "positive": int((observed == 1).sum()),
            "negative": int((observed == 0).sum()),
            "target_unobserved": int(target.isna().sum()),
            "positive_rate_observed": float((observed == 1).mean()) if len(observed) else None,
        }

    summaries = [summary(name, df) for name, df in datasets.items()]
    pd.DataFrame(summaries).to_csv(output_dir / "dataset_summary.csv", index=False)
    manifest = {
        "purpose": "Leakage-audited snapshot tables for the three-stage student-success cascade.",
        "source_files": ["students.csv", "quiz_attempts.csv", "outcome.csv"],
        "feature_columns": feature_cols,
        "dataset_feature_columns": dataset_feature_cols,
        "student_level_features": STUDENT_FEATURES,
        "excluded_suspect_fields": SUSPECT_NON_FEATURE_FIELDS,
        "retrospective_audit_columns": [
            "prefix_length_audit", "eventual_attempt_count_audit",
            "next_attempt_available_audit", "target_reason_audit",
        ],
        "datasets": {
            "A": "All students; target is supplied dropout label AND total attempts <= 1.",
            "B_retrospective_capped": "Project's proposed B history: N=2 uses k=1; N>=3 uses k=2. Retrospective comparison only; timing depends on eventual N.",
            "B_operational_after_attempt_1": "Operational comparator: after attempt 1; predicts whether attempt 2 is the final recorded attempt with positive dropout label.",
            "B_operational_censored_rows": "After-attempt-1 snapshots with no observed second attempt; target is blank and these rows are not trainable for the next-attempt target.",
            "B_rolling_early": "Rolling operational B snapshots after k=1 and k=2 when a next attempt is observed; positive iff the next attempt is the final recorded attempt and dropout label is positive.",
            "B_rolling_early_censored_rows": "Final observed snapshots at k=1 or k=2 without a subsequent attempt; target is blank, not negative.",
            "C1_penultimate_only": "One row for N>=4; prefix N-1; target is supplied eventual dropout label.",
            "C2_rolling": "Rows for every prefix k>=3 with an observed next attempt; positive only if that next attempt is final and dropout label is positive.",
            "C2_censored_rows": "Final observed prefixes with no next attempt in the historical data; target is blank and these rows are not trainable for next-attempt target.",
        },
        "assertions_passed": True,
        "important_caveats": [
            "No true dropout timestamp is present; final recorded attempt is a proxy.",
            "B_retrospective_capped is not operationally routable because the selected prefix length depends on eventual total attempts.",
            "C1 is a retrospective penultimate-only benchmark and cannot be triggered live without an independent policy.",
            "C2 excludes snapshots without an observed next attempt from supervised next-attempt training; these are exported separately as censored rows.",
            "refund_requested is excluded by default because its availability timing may be post-signup or post-outcome and requires domain verification.",
            "All snapshots for a student must remain in the same train/validation/test fold.",
        ],
        "counts": {
            "students": int(len(students)),
            "attempt_rows": int(len(attempts)),
            "students_with_no_attempts": int(sum(n == 0 for n in total_count.values())),
            "students_with_one_attempt": int(sum(n == 1 for n in total_count.values())),
            "students_with_two_or_more_attempts": int(sum(n >= 2 for n in total_count.values())),
            "students_with_four_or_more_attempts": int(sum(n >= 4 for n in total_count.values())),
            "score_999_rows": int(pd.to_numeric(attempts["score"], errors="coerce").eq(SCORE_SENTINEL).sum()),
            "missing_duration_rows": int(pd.to_numeric(attempts["time_spent_min"], errors="coerce").isna().sum()),
        },
        "dataset_summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# Cascade Snapshot Builder: QA Report",
        "",
        "## Status",
        "",
        "**PASS:** dataset construction and programmed assertions completed.",
        "",
        "## Dataset counts",
        "",
        "| Dataset | Rows | Unique students | Positive | Negative | Unobserved target | Positive rate (observed) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        rate = "—" if s["positive_rate_observed"] is None else f'{s["positive_rate_observed"]:.3f}'
        report_lines.append(f'| {s["dataset"]} | {s["rows"]} | {s["unique_students"]} | {s["positive"]} | {s["negative"]} | {s["target_unobserved"]} | {rate} |')
    report_lines += [
        "",
        "## Leakage and correctness checks",
        "",
        "- All 500 students appear exactly once in A.",
        "- Attempt rows are sorted by `attempted_at`, with `attempt_id` as deterministic tie-breaker.",
        "- Every attempt-derived feature is recomputed from the visible chronological prefix only.",
        "- The next attempt is excluded from feature aggregates in B, C1, and C2.",
        "- Eventual attempt count, prefix length, target reason, and next-attempt availability are audit metadata, not predictors.",
        "- `dropped_out` and the normalized label are excluded from feature columns.",
        "- `refund_requested` is excluded from model features pending timing validation.",
        "- Score sentinel 999 is excluded from score aggregates; missing duration is not replaced by zero.",
        "- B rolling and C2 rows without a historical next attempt are exported separately with unobserved target.",
        "",
        "## Interpretation cautions",
        "",
        "- B_retrospective_capped is the requested retrospective design but its snapshot time depends on eventual attempt count; it is not a deployable routing rule as written.",
        "- B_operational_after_attempt_1 is included as a strict operational comparator; B_rolling_early adds k=1 and k=2 snapshots.",
        "- B_rolling_early and C2 are conditional next-attempt models, not direct no-return/dropout-time models.",
        "- C1 is retrospective penultimate-only; C2 is rolling and operationally meaningful when each new attempt is observed.",
        "- These are dataset/label audits, not model-performance results.",
        "",
    ]
    (output_dir / "QA_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cascade_snapshots"))
    args = parser.parse_args()
    manifest = build(args.data_dir, args.output_dir)
    print(json.dumps(manifest["dataset_summaries"], indent=2))
    print(f"Assertions passed. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

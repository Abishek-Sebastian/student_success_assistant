#!/usr/bin/env python3
"""Leakage-aware baseline models for the Student Success cascade snapshots.

- Student-grouped locked holdout and grouped CV for repeated-snapshot datasets.
- Compact, stage-available feature sets; no audit/target-construction columns.
- Preprocessing is fitted inside each pipeline/fold.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score, precision_recall_curve,
    f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASETS = {
    "A": ("A_model_table.csv", "student"),
    "B_rolling_early": ("B_rolling_early_model_table.csv", "snapshot"),
    "C1_penultimate_only": ("C1_penultimate_only_model_table.csv", "student"),
    "C2_rolling": ("C2_rolling_model_table.csv", "snapshot"),
}
CATEGORICAL = ["age_group", "country", "device", "signup_month", "signup_weekday"]
ATTEMPT_FEATURES = [
    "score_valid_count",
    "score_missing_or_invalid_fraction",
    "score_mean",
    "score_last_valid",
    "score_last3_mean",
    "time_valid_count",
    "time_missing_fraction",
    "time_mean",
    "time_last_valid",
    "time_last3_mean",
    "time_total_valid_sum",
    "mean_gap_hours",
    "last_gap_hours",
    "history_span_hours",
]
MODELS = {
    "dummy_prior": "dummy",
    "logistic_unweighted": "logistic",
    "logistic_balanced": "logistic_balanced",
    "extra_trees_balanced": "extra_trees",
}

def prepare_features(df: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create prediction-time-safe features from the exported model table."""
    x = df.drop(columns=["student_id", "target", "_source_row_index"], errors="ignore").copy()
    if "signup_date" in x:
        dates = pd.to_datetime(x.pop("signup_date"), errors="coerce")
        x["signup_month"] = dates.dt.month.astype("float")
        x["signup_weekday"] = dates.dt.dayofweek.astype("float")
    # A is explicitly demographics-only. No attempt-derived fields are allowed.
    if stage == "A":
        x = x[[c for c in CATEGORICAL if c in x.columns]]
    else:
        # Deliberately compact baseline. Avoid slopes and module-specific features
        # until the simple level/recency/engagement benchmark is established.
        keep = [c for c in CATEGORICAL + ATTEMPT_FEATURES if c in x.columns]
        x = x[keep]
    cat_cols = [c for c in CATEGORICAL if c in x.columns]
    num_cols = [c for c in x.columns if c not in cat_cols]
    return x, cat_cols, num_cols

def make_pipeline(cat_cols, num_cols, model_name):
    transformers = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if model_name in {"logistic_unweighted", "logistic_balanced"}:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    prep = ColumnTransformer(transformers, remainder="drop")
    if model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
    elif model_name == "logistic_unweighted":
        estimator = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
    elif model_name == "logistic_balanced":
        estimator = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42)
    elif model_name == "extra_trees_balanced":
        estimator = ExtraTreesClassifier(
            n_estimators=60, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=1
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", prep), ("model", estimator)])

def metrics(y_true, prob, pred):
    out = {
        "average_precision": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "precision_at_0_5": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, pred)),
        "accuracy_at_0_5": float(accuracy_score(y_true, pred)),
        "tn": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, pred, labels=[0, 1])[1, 1]),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
    }
    return out

def choose_threshold_for_precision_floor(y_true, prob, precision_floor):
    """Maximize recall subject to a precision floor using development OOF scores."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # Last precision/recall point has no corresponding threshold (predict none).
    feasible_idx = np.flatnonzero(precision[:-1] >= precision_floor)
    if len(feasible_idx) == 0:
        return {"threshold": float(np.nextafter(np.max(prob), np.inf)),
                "development_precision": 0.0, "development_recall": 0.0,
                "development_predicted_positive": 0, "feasible": False}
    best_recall = np.max(recall[feasible_idx])
    best_idx_candidates = feasible_idx[np.isclose(recall[feasible_idx], best_recall)]
    # Among equal-recall thresholds, prefer the highest precision, then the lower threshold.
    best_precision = np.max(precision[best_idx_candidates])
    best_idx_candidates = best_idx_candidates[np.isclose(precision[best_idx_candidates], best_precision)]
    best_idx = best_idx_candidates[np.argmin(thresholds[best_idx_candidates])]
    threshold = float(thresholds[best_idx])
    pred_count = int(np.sum(prob >= threshold))
    return {"threshold": threshold,
            "development_precision": float(precision[best_idx]),
            "development_recall": float(recall[best_idx]),
            "development_predicted_positive": pred_count,
            "feasible": True}


def threshold_metrics(y_true, prob, threshold):
    pred = (np.asarray(prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {"threshold": float(threshold),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "predicted_positive": int(pred.sum()),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            "positive_rate": float(np.mean(y_true)),
            "n": int(len(y_true)), "positives": int(np.sum(y_true))}


def split_student_ids(df, test_size, seed):
    # Stratify student groups by whether any of their snapshots is positive.
    group_y = df.groupby("student_id", sort=True)["target"].max()
    ids = group_y.index.to_numpy()
    y = group_y.to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(splitter.split(ids.reshape(-1, 1), y))
    return set(ids[tr]), set(ids[te])

def evaluate_dataset(name, filename, kind, input_dir, output_dir, test_size, n_splits, seed, precision_floors):
    path = input_dir / filename
    df = pd.read_csv(path)
    if not {"student_id", "target"}.issubset(df.columns):
        raise ValueError(f"{path} must contain student_id and target")
    if df["target"].isna().any():
        raise ValueError(f"{name} model table contains unobserved targets; use the corresponding non-censored table")
    df["target"] = df["target"].astype(int)
    df["_source_row_index"] = np.arange(len(df))
    x, cat_cols, num_cols = prepare_features(df, name)
    y = df["target"].to_numpy()
    groups = df["student_id"].astype(str).to_numpy()
    train_ids, test_ids = split_student_ids(df.assign(student_id=groups), test_size, seed)
    train_mask = np.isin(groups, list(train_ids))
    test_mask = np.isin(groups, list(test_ids))
    if set(groups[train_mask]) & set(groups[test_mask]):
        raise AssertionError("student leakage across locked split")
    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    X_train = x.loc[train_mask].reset_index(drop=True)
    X_test = x.loc[test_mask].reset_index(drop=True)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    g_train = train_df["student_id"].astype(str).to_numpy()

    # Grouped CV on development students only. Fold count is bounded by the
    # number of positive student groups; the protocol default is five.
    positive_groups = int(train_df.groupby("student_id")["target"].max().sum())
    folds = min(n_splits, positive_groups, len(np.unique(g_train)))
    if folds < 2:
        raise ValueError(f"Not enough positive student groups for CV in {name}")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows, oof_rows, test_rows, pred_rows, oof_pred_rows, operating_rows = [], [], [], [], [], []
    for model_name in MODELS:
        pipe = make_pipeline(cat_cols, num_cols, model_name)
        oof_prob = np.full(len(train_df), np.nan)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups=g_train), start=1):
            # Explicit leakage guard.
            if set(g_train[tr_idx]) & set(g_train[va_idx]):
                raise AssertionError("student leakage across CV fold")
            fitted = clone(pipe)
            fitted.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            prob = fitted.predict_proba(X_train.iloc[va_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            oof_prob[va_idx] = prob
            m = metrics(y_train[va_idx], prob, pred)
            fold_rows.append({"dataset": name, "model": model_name, "fold": fold, **m})
        valid = ~np.isnan(oof_prob)
        oof_y = y_train[valid]
        oof_p = oof_prob[valid]
        oof_pred = (oof_p >= 0.5).astype(int)
        oof_m = metrics(oof_y, oof_p, oof_pred)
        oof_rows.append({"dataset": name, "model": model_name, "evaluation": "development_oof_at_0_5", **oof_m})
        valid_rows = train_df.loc[valid].reset_index(drop=True)
        for row_idx, (sid, prob, target) in enumerate(zip(valid_rows["student_id"], oof_p, oof_y)):
            oof_pred_rows.append({"dataset": name, "model": model_name,
                "row_index": int(valid_rows["_source_row_index"].iloc[row_idx]),
                "student_id": sid, "target": int(target), "probability": float(prob)})

        # Fit once on all development students. Threshold selection only sees OOF scores.
        fitted = clone(pipe)
        fitted.fit(X_train, y_train)
        test_prob = fitted.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        test_m = metrics(y_test, test_prob, test_pred)
        test_rows.append({"dataset": name, "model": model_name, "evaluation": "locked_test_at_0_5", **test_m})

        for floor in precision_floors:
            selected = choose_threshold_for_precision_floor(oof_y, oof_p, floor)
            test_operating = threshold_metrics(y_test, test_prob, selected["threshold"])
            operating_rows.append({
                "dataset": name, "model": model_name, "precision_floor": floor,
                "selection_source": "development_oof_only",
                "development_threshold": selected["threshold"],
                "development_precision": selected["development_precision"],
                "development_recall": selected["development_recall"],
                "development_predicted_positive": selected["development_predicted_positive"],
                "development_feasible": selected["feasible"],
                **{f"test_{k}": v for k, v in test_operating.items()},
            })
            if abs(floor - precision_floors[0]) < 1e-9:
                selected_threshold = selected["threshold"]
            else:
                selected_threshold = None

        primary = choose_threshold_for_precision_floor(oof_y, oof_p, precision_floors[0])
        for row_idx, (sid, prob, pred, target) in enumerate(zip(test_df["student_id"], test_prob, test_pred, y_test)):
            pred_rows.append({"dataset": name, "model": model_name,
                "row_index": int(test_df["_source_row_index"].iloc[row_idx]),
                "student_id": sid, "target": int(target), "probability": float(prob),
                "prediction_at_0_5": int(pred),
                "prediction_at_primary_precision_floor": int(prob >= primary["threshold"]),
                "selected_threshold": float(primary["threshold"])})
    # Aggregate fold metrics mean and std (macro over folds).
    fold_df = pd.DataFrame([r for r in fold_rows if r["dataset"] == name])
    agg_rows = []
    metric_cols = ["average_precision", "roc_auc", "precision_at_0_5", "recall_at_0_5",
                   "f1_at_0_5", "balanced_accuracy_at_0_5"]
    for model_name, part in fold_df.groupby("model"):
        row = {"dataset": name, "model": model_name, "evaluation": "grouped_cv_fold_mean"}
        for col in metric_cols:
            row[col] = part[col].mean()
            row[col + "_std"] = part[col].std(ddof=1) if len(part) > 1 else 0.0
        row["folds"] = len(part)
        row["positive_rate"] = float(y_train.mean())
        row["n"] = len(y_train)
        row["positives"] = int(y_train.sum())
        agg_rows.append(row)
    # Keep split membership and test predictions for auditability.
    split_out = output_dir / "splits"
    split_out.mkdir(parents=True, exist_ok=True)
    with open(split_out / f"{name}_split_ids.json", "w") as f:
        json.dump({"seed": seed, "test_size": test_size, "train_student_ids": sorted(train_ids),
                   "test_student_ids": sorted(test_ids)}, f, indent=2)
    return fold_rows, agg_rows, oof_rows, test_rows, pred_rows, oof_pred_rows, operating_rows, {
        "dataset": name, "rows": len(df), "unique_students": df.student_id.nunique(),
        "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "train_rows": int(train_mask.sum()), "train_students": len(train_ids),
        "train_positives": int(y_train.sum()), "test_rows": int(test_mask.sum()),
        "test_students": len(test_ids), "test_positives": int(y_test.sum()),
        "categorical_features": ",".join(cat_cols), "numeric_features": ",".join(num_cols),
        "n_features_before_encoding": int(x.shape[1]), "cv_folds": folds,
    }


def evaluate_end_to_end_cascade(input_dir, output_dir, test_size, n_splits, seed, precision_floors):
    """Jointly tune a student-level A/B/C flag policy against eventual dropout.

    Stage models are trained on their existing stage-specific targets. Their
    thresholds are jointly selected against the common eventual-dropout label
    using development OOF student-level scores. B/C scores use only snapshots
    with a subsequent attempt observed, so censored post-final snapshots do not
    become artificial actionable checkpoints.
    """
    audit = pd.read_csv(input_dir / "snapshot_audit.csv")
    if not {"student_id", "dropout_binary_audit_only"}.issubset(audit.columns):
        raise ValueError("snapshot_audit.csv must contain student_id and dropout_binary_audit_only")
    student_meta = audit[["student_id", "dropout_binary_audit_only"]].drop_duplicates("student_id").copy()
    student_meta["target_eventual_dropout"] = student_meta["dropout_binary_audit_only"].astype(int)
    student_ids = student_meta["student_id"].astype(str).to_numpy()
    y_student = student_meta["target_eventual_dropout"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(student_ids.reshape(-1, 1), y_student))
    train_ids = set(student_ids[train_idx])
    test_ids = set(student_ids[test_idx])
    if train_ids & test_ids:
        raise AssertionError("cascade split leaked students")
    train_meta = student_meta.iloc[train_idx].copy().reset_index(drop=True)
    test_meta = student_meta.iloc[test_idx].copy().reset_index(drop=True)
    train_student_ids = train_meta["student_id"].astype(str).to_numpy()
    test_student_ids = test_meta["student_id"].astype(str).to_numpy()
    y_train_student = train_meta["target_eventual_dropout"].to_numpy()
    y_test_student = test_meta["target_eventual_dropout"].to_numpy()

    stage_specs = {
        "A": ("A_model_table.csv", "A"),
        "B": ("B_rolling_early_model_table.csv", "B_rolling_early"),
        "C": ("C2_rolling_model_table.csv", "C2_rolling"),
    }
    stage_data = {}
    for stage, (filename, feature_stage) in stage_specs.items():
        df = pd.read_csv(input_dir / filename)
        df["_source_row_index"] = np.arange(len(df))
        if df["target"].isna().any():
            raise ValueError(f"{filename} contains unobserved training targets")
        df["target"] = df["target"].astype(int)
        X, cat_cols, num_cols = prepare_features(df, feature_stage)
        stage_data[stage] = {
            "df": df, "X": X, "cat_cols": cat_cols, "num_cols": num_cols,
            "feature_stage": feature_stage,
        }

    positive_students = int(y_train_student.sum())
    folds = min(n_splits, positive_students, int((y_train_student == 0).sum()))
    if folds < 2:
        raise ValueError("Not enough positive/negative students for cascade CV")
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    stage_oof_snapshot_scores = {stage: {model: np.full(len(info["df"]), np.nan) for model in MODELS}
                                 for stage, info in stage_data.items()}
    fold_ids = train_meta["student_id"].astype(str).to_numpy()

    # One shared student fold assignment is used across all three stages.
    fold_assignments = []
    for fold, (tr_meta_idx, va_meta_idx) in enumerate(
        cv.split(train_student_ids.reshape(-1, 1), y_train_student, groups=train_student_ids), start=1
    ):
        fold_train_ids = set(train_student_ids[tr_meta_idx])
        fold_val_ids = set(train_student_ids[va_meta_idx])
        if fold_train_ids & fold_val_ids:
            raise AssertionError("cascade CV student leakage")
        fold_assignments.append((fold, fold_train_ids, fold_val_ids))
        for stage, info in stage_data.items():
            df, X = info["df"], info["X"]
            row_ids = df["student_id"].astype(str).to_numpy()
            tr_mask = np.isin(row_ids, list(fold_train_ids))
            va_mask = np.isin(row_ids, list(fold_val_ids))
            if not va_mask.any():
                continue
            y_rows = df["target"].to_numpy()
            for model_name in MODELS:
                fitted = clone(make_pipeline(info["cat_cols"], info["num_cols"], model_name))
                fitted.fit(X.loc[tr_mask], y_rows[tr_mask])
                probs = fitted.predict_proba(X.loc[va_mask])[:, 1]
                stage_oof_snapshot_scores[stage][model_name][np.flatnonzero(va_mask)] = probs

    # Collapse each stage's OOF snapshot scores to a single score per student:
    # the maximum risk score at any actionable checkpoint for that stage.
    def collapse_scores(stage, snapshot_scores):
        df = stage_data[stage]["df"]
        result = {}
        for model_name, scores in snapshot_scores.items():
            tmp = pd.DataFrame({"student_id": df["student_id"].astype(str), "score": scores})
            result[model_name] = tmp.groupby("student_id")["score"].max()
        return result

    oof_student_scores = {stage: collapse_scores(stage, stage_oof_snapshot_scores[stage])
                          for stage in stage_data}
    stage_model_rows = []
    selected_models = {}
    for stage in stage_data:
        for model_name in MODELS:
            series = oof_student_scores[stage][model_name].reindex(train_student_ids)
            valid = series.notna().to_numpy()
            scores = series.to_numpy(dtype=float)
            ap = float(average_precision_score(y_train_student[valid], scores[valid])) if len(np.unique(y_train_student[valid])) == 2 else np.nan
            stage_model_rows.append({
                "stage": stage, "model": model_name, "development_eventual_outcome_ap": ap,
                "students_scored": int(valid.sum()), "positive_students_scored": int(y_train_student[valid].sum()),
                "positive_rate_among_scored": float(y_train_student[valid].mean()) if valid.any() else np.nan,
            })
        # Select the model by development OOF AP against the common eventual outcome,
        # not by locked-test performance.
        stage_candidates = [r for r in stage_model_rows if r["stage"] == stage and np.isfinite(r["development_eventual_outcome_ap"])]
        selected_models[stage] = max(stage_candidates, key=lambda r: r["development_eventual_outcome_ap"])["model"]

    selected_oof = {}
    selected_test = {}
    for stage, model_name in selected_models.items():
        series = oof_student_scores[stage][model_name].reindex(train_student_ids)
        selected_oof[stage] = series.to_numpy(dtype=float)
        info = stage_data[stage]
        df, X = info["df"], info["X"]
        row_ids = df["student_id"].astype(str).to_numpy()
        tr_mask = np.isin(row_ids, list(train_ids))
        te_mask = np.isin(row_ids, list(test_ids))
        fitted = clone(make_pipeline(info["cat_cols"], info["num_cols"], model_name))
        fitted.fit(X.loc[tr_mask], df.loc[tr_mask, "target"].to_numpy())
        probs = fitted.predict_proba(X.loc[te_mask])[:, 1]
        tmp = pd.DataFrame({"student_id": row_ids[te_mask], "score": probs})
        max_scores = tmp.groupby("student_id")["score"].max()
        selected_test[stage] = max_scores.reindex(test_student_ids).to_numpy(dtype=float)

    # Candidate thresholds from development score quantiles. Joint search evaluates
    # the OR policy: a student is flagged if any stage ever crosses its threshold.
    def candidate_thresholds(scores):
        values = scores[np.isfinite(scores)]
        if len(values) == 0:
            return [float("inf")]
        q = np.quantile(values, np.linspace(0, 1, 41))
        return sorted(set(float(v) for v in q) | {float(np.nextafter(np.max(values), np.inf))})

    stage_candidates = {stage: candidate_thresholds(scores) for stage, scores in selected_oof.items()}
    policy_rows = []
    for floor in precision_floors:
        best = None
        for ta in stage_candidates["A"]:
            flag_a = np.isfinite(selected_oof["A"]) & (selected_oof["A"] >= ta)
            for tb in stage_candidates["B"]:
                flag_ab = flag_a | (np.isfinite(selected_oof["B"]) & (selected_oof["B"] >= tb))
                for tc in stage_candidates["C"]:
                    flags = flag_ab | (np.isfinite(selected_oof["C"]) & (selected_oof["C"] >= tc))
                    n_flagged = int(flags.sum())
                    if n_flagged == 0:
                        continue
                    tp = int(np.sum(flags & (y_train_student == 1)))
                    precision = float(tp / n_flagged) if n_flagged else 0.0
                    if precision + 1e-12 < floor:
                        continue
                    recall = float(tp / np.sum(y_train_student == 1))
                    candidate = (recall, precision, -n_flagged, ta, tb, tc, n_flagged)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
        if best is None:
            thresholds = {"A": float("inf"), "B": float("inf"), "C": float("inf")}
            dev_flags = np.zeros(len(y_train_student), dtype=bool)
            dev_precision = dev_recall = 0.0
            dev_n_flagged = 0
            feasible = False
        else:
            dev_recall, dev_precision, _, ta, tb, tc, dev_n_flagged = best
            thresholds = {"A": ta, "B": tb, "C": tc}
            dev_flags = ((np.isfinite(selected_oof["A"]) & (selected_oof["A"] >= ta)) |
                         (np.isfinite(selected_oof["B"]) & (selected_oof["B"] >= tb)) |
                         (np.isfinite(selected_oof["C"]) & (selected_oof["C"] >= tc)))
            feasible = True

        test_flags = ((np.isfinite(selected_test["A"]) & (selected_test["A"] >= thresholds["A"])) |
                      (np.isfinite(selected_test["B"]) & (selected_test["B"] >= thresholds["B"])) |
                      (np.isfinite(selected_test["C"]) & (selected_test["C"] >= thresholds["C"])))
        cm = confusion_matrix(y_test_student, test_flags.astype(int), labels=[0, 1])
        test_precision = float(precision_score(y_test_student, test_flags, zero_division=0))
        test_recall = float(recall_score(y_test_student, test_flags, zero_division=0))
        policy_rows.append({
            "precision_floor": floor, "feasible_on_development": feasible,
            "selected_model_A": selected_models["A"], "selected_model_B": selected_models["B"],
            "selected_model_C": selected_models["C"],
            "threshold_A": thresholds["A"], "threshold_B": thresholds["B"], "threshold_C": thresholds["C"],
            "development_precision": dev_precision, "development_recall": dev_recall,
            "development_flagged_students": dev_n_flagged,
            "test_precision": test_precision, "test_recall": test_recall,
            "test_flagged_students": int(test_flags.sum()),
            "test_tn": int(cm[0,0]), "test_fp": int(cm[0,1]),
            "test_fn": int(cm[1,0]), "test_tp": int(cm[1,1]),
            "test_students": int(len(y_test_student)), "test_positive_students": int(y_test_student.sum()),
        })
        pred_df = test_meta.copy()
        pred_df["stage_A_max_score"] = selected_test["A"]
        pred_df["stage_B_max_score"] = selected_test["B"]
        pred_df["stage_C_max_score"] = selected_test["C"]
        pred_df["flag_A"] = np.isfinite(selected_test["A"]) & (selected_test["A"] >= thresholds["A"])
        pred_df["flag_B"] = np.isfinite(selected_test["B"]) & (selected_test["B"] >= thresholds["B"])
        pred_df["flag_C"] = np.isfinite(selected_test["C"]) & (selected_test["C"] >= thresholds["C"])
        pred_df["cascade_flag"] = test_flags
        pred_df["first_flag_stage"] = np.select(
            [pred_df["flag_A"], (~pred_df["flag_A"]) & pred_df["flag_B"],
             (~pred_df["flag_A"]) & (~pred_df["flag_B"]) & pred_df["flag_C"]],
            ["A", "B", "C"], default="none"
        )
        pred_df["precision_floor"] = floor
        pred_df["threshold_A"] = thresholds["A"]
        pred_df["threshold_B"] = thresholds["B"]
        pred_df["threshold_C"] = thresholds["C"]
        pred_df.to_csv(output_dir / f"cascade_test_predictions_precision_{int(floor*100)}.csv", index=False)

    stage_models_df = pd.DataFrame(stage_model_rows)
    stage_models_df.to_csv(output_dir / "cascade_stage_model_selection.csv", index=False)
    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(output_dir / "end_to_end_cascade_precision_recall.csv", index=False)
    with open(output_dir / "cascade_split_ids.json", "w") as f:
        json.dump({"seed": seed, "test_size": test_size,
                   "train_student_ids": sorted(train_ids), "test_student_ids": sorted(test_ids)}, f, indent=2)
    return stage_models_df, policy_df, selected_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/cascade_snapshots")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision-floors", type=float, nargs="+", default=[0.50, 0.40, 0.30, 0.60], help="Precision floors for recall-maximizing thresholds. First value is primary.")
    args = parser.parse_args()
    root = Path.cwd()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute(): input_dir = root / input_dir
    if not output_dir.is_absolute(): output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_folds, all_agg, all_oof, all_test, all_preds, all_oof_preds, all_operating, summaries = [], [], [], [], [], [], [], []
    for name, (filename, kind) in DATASETS.items():
        print(f"Evaluating {name}...")
        result = evaluate_dataset(name, filename, kind, input_dir, output_dir, args.test_size, args.folds, args.seed, args.precision_floors)
        folds, agg, oof, test, preds, oof_preds, operating, summary = result
        all_folds.extend(folds); all_agg.extend(agg); all_oof.extend(oof); all_test.extend(test); all_preds.extend(preds)
        all_oof_preds.extend(oof_preds); all_operating.extend(operating); summaries.append(summary)
    cascade_models_df, cascade_policy_df, selected_cascade_models = evaluate_end_to_end_cascade(
        input_dir, output_dir, args.test_size, args.folds, args.seed, args.precision_floors
    )
    pd.DataFrame(all_folds).to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(all_agg).to_csv(output_dir / "cv_summary.csv", index=False)
    pd.DataFrame(all_oof).to_csv(output_dir / "development_oof_metrics.csv", index=False)
    pd.DataFrame(all_test).to_csv(output_dir / "locked_test_metrics.csv", index=False)
    pd.DataFrame(all_preds).to_csv(output_dir / "locked_test_predictions.csv", index=False)
    pd.DataFrame(all_oof_preds).to_csv(output_dir / "development_oof_predictions.csv", index=False)
    pd.DataFrame(all_operating).to_csv(output_dir / "precision_constrained_recall.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "dataset_split_summary.csv", index=False)
    # Human-readable report with caveats.
    cv_df = pd.DataFrame(all_agg)
    test_df = pd.DataFrame(all_test)
    report = [
        "# Baseline modelling report",
        "",
        "## Scope and safeguards",
        "- Models: prevalence dummy, unweighted logistic regression, class-weighted logistic regression, class-weighted ExtraTrees.",
        "- Features: compact baseline only. A uses demographic/signup-calendar features only; B/C use those plus global score/time levels, recency, counts and engagement features.",
        "- Preprocessing (median imputation, missing indicators, one-hot encoding, scaling for logistic regression) is fitted inside each training fold.",
        "- Locked holdout is split by student. Repeated snapshots from a student never cross partitions. Development CV uses StratifiedGroupKFold.",
        "- Primary ranking metric: average precision. Operational objective: maximize recall subject to a minimum precision floor.",
        "- Operating thresholds are selected from development out-of-fold predictions only, then evaluated on the locked test set. The first configured precision floor is primary (default 0.50); additional floors are reported for sensitivity analysis.",
        "- Threshold metrics at 0.5 are retained only as a reference, not as the recommended operating policy.",
        "- These are conditional snapshot-label benchmarks, not end-to-end cascade results and not proof of no-return prediction.",
        "",
        "## Dataset split summary",
        pd.DataFrame(summaries).to_markdown(index=False),
        "",
        "## Development grouped-CV results",
        cv_df.sort_values(["dataset", "average_precision"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Locked test results (fixed 0.5 threshold; reference only)",
        test_df.sort_values(["dataset", "average_precision"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Recall maximization subject to a precision floor (per-stage benchmarks)",
        pd.DataFrame(all_operating).to_markdown(index=False),
        "",
        "## End-to-end cascade policy",
        "This section uses a common eventual-dropout label from the audit table for student-level evaluation. Stage models are selected using development OOF AP against this common label, then A/B/C thresholds are jointly tuned to maximize student-level cascade recall under the precision floor. B/C use only snapshots with a subsequent attempt observed; censored final prefixes are not treated as actionable checkpoints.",
        "",
        cascade_models_df.to_markdown(index=False),
        "",
        cascade_policy_df.to_markdown(index=False),
        "",
        "## Interpretation cautions",
        "- The datasets are small and positives are sparse, especially A. Test metrics can move substantially with only a few positive students.",
        "- Compare average precision against each dataset's positive prevalence; a model that does not beat prevalence is not useful as a ranker.",
        "- C2 has repeated snapshots. Snapshot-level metrics weight students with more decision points more heavily; a later experiment should compare student-balanced weights.",
        "- B rolling and C2 only label rows where a next attempt is observed. Censored no-next-attempt rows are intentionally excluded, not treated as negatives.",
        "- C1 is retrospective penultimate-only and should be treated as a benchmark, not directly deployable routing logic.",
        "- A uses no attempt-derived predictors, but its target is the supplied label restricted to students with at most one recorded attempt.",
        "- Do not choose the final model from locked test metrics; use development CV and evaluate the test once after model/policy freeze.",
        "- Threshold selection on pooled development OOF predictions can be optimistic; locked-test results are the independent check. With few positives, test precision/recall are highly discrete and uncertain.",
        "- End-to-end cascade metrics use eventual `dropped_out` as the common student-level outcome and count each student once. Because no dropout timestamp exists, they evaluate eventual-outcome identification, not the timeliness of an alert. The cascade policy only scores B/C at snapshots where a next attempt is observed; censored final prefixes are excluded as non-actionable for the next-attempt target.",
    ]
    (output_dir / "BASELINE_REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Done. Outputs: {output_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

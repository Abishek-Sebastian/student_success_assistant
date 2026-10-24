#!/usr/bin/env python3
"""Final fixed-configuration evaluation for the existing A/B/C dropout cascade.

Uses the best-supported configuration from the completed experiments:
  A: class-weighted logistic regression, demographics/signup calendar
  B: class-weighted logistic regression, compact + per-lesson features
  C: shallow depth-2 XGBoost with student-level positive-class weighting,
     compact + per-lesson features

No architecture, dataset, target, or feature-generation changes are made.
Thresholds are selected on grouped development OOF scores only. Test students
are held out by student ID, and every snapshot from a student stays in one fold.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

CATS = ["age_group", "country", "device", "signup_month", "signup_weekday"]
COMPACT = [
    "score_valid_count", "score_missing_or_invalid_fraction", "score_mean",
    "score_last_valid", "score_last3_mean", "time_valid_count",
    "time_missing_fraction", "time_mean", "time_last_valid", "time_last3_mean",
    "time_total_valid_sum", "mean_gap_hours", "last_gap_hours",
    "history_span_hours",
]
MODULE = [
    f"lesson_L{i}_{suffix}"
    for i in (1, 2, 3)
    for suffix in ("attempt_count", "score_mean", "time_mean")
]
FEATURE_SETS = {
    "demographics": [],
    "compact_plus_module": COMPACT + MODULE,
}
STAGE_CONFIGS = {
    "A": {
        "name": "logistic_class_weight",
        "family": "logistic",
        "feature_set": "demographics",
        "balance": "balanced",
    },
    "B": {
        "name": "logistic_class_weight",
        "family": "logistic",
        "feature_set": "compact_plus_module",
        "balance": "balanced",
    },
    "C": {
        "name": "xgb_d2_group_weight",
        "family": "xgboost",
        "feature_set": "compact_plus_module",
        "balance": "group_ratio",
        "n_estimators": 300,
        "learning_rate": 0.025,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "reg_alpha": 0.2,
        "gamma": 0.0,
    },
}


def feature_frame(df: pd.DataFrame, stage: str, feature_set: str):
    excluded = [
        "student_id", "target", "_row_id", "snapshot_index", "target_observed",
        "target_reason_audit", "eventual_attempt_count_audit",
        "prefix_length_audit", "next_attempt_available_audit",
    ]
    x = df.drop(columns=excluded, errors="ignore").copy()
    if "signup_date" in x:
        d = pd.to_datetime(x.pop("signup_date"), errors="coerce")
        x["signup_month"] = d.dt.month.astype(float)
        x["signup_weekday"] = d.dt.dayofweek.astype(float)
    keep = list(CATS)
    if stage != "A":
        keep += FEATURE_SETS[feature_set]
    x = x[[c for c in keep if c in x.columns]]
    cats = [c for c in CATS if c in x.columns]
    nums = [c for c in x.columns if c not in cats]
    return x, cats, nums


def build_preprocessor(cats, nums, family):
    transformers = []
    if nums:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if family == "logistic":
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(numeric_steps), nums))
    if cats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cats,
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def make_stage_data(input_dir: Path, audit: pd.DataFrame):
    a = pd.read_csv(input_dir / "A_model_table.csv").drop(columns=["target"], errors="ignore")
    a = a.merge(
        audit[["student_id", "eventual_dropout"]],
        on="student_id", how="left", validate="one_to_one",
    ).rename(columns={"eventual_dropout": "target"})
    a["_row_id"] = np.arange(len(a))

    b = pd.concat([
        pd.read_csv(input_dir / "B_rolling_early.csv"),
        pd.read_csv(input_dir / "B_rolling_early_censored_rows.csv"),
    ], ignore_index=True).drop_duplicates(["student_id", "snapshot_index"])
    b = b.drop(columns=[
        "target", "target_observed", "target_reason_audit",
        "eventual_attempt_count_audit", "prefix_length_audit",
        "next_attempt_available_audit",
    ], errors="ignore")
    b = b.merge(
        audit[["student_id", "eventual_dropout"]],
        on="student_id", how="left", validate="many_to_one",
    ).rename(columns={"eventual_dropout": "target"})
    b["_row_id"] = np.arange(len(b))

    c = pd.concat([
        pd.read_csv(input_dir / "C2_rolling.csv"),
        pd.read_csv(input_dir / "C2_censored_rows.csv"),
    ], ignore_index=True).drop_duplicates(["student_id", "snapshot_index"])
    c = c.drop(columns=[
        "target", "target_observed", "target_reason_audit",
        "eventual_attempt_count_audit", "prefix_length_audit",
        "next_attempt_available_audit",
    ], errors="ignore")
    c = c.merge(
        audit[["student_id", "eventual_dropout"]],
        on="student_id", how="left", validate="many_to_one",
    ).rename(columns={"eventual_dropout": "target"})
    c["_row_id"] = np.arange(len(c))

    stages = {"A": a, "B": b, "C": c}
    for stage, df in stages.items():
        if df.target.isna().any():
            raise ValueError(f"Missing eventual label for stage {stage}")
        df["target"] = df.target.astype(int)
        df["student_id"] = df.student_id.astype(str)
        if df.groupby("student_id").target.nunique().max() > 1:
            raise AssertionError(f"Label inconsistent within student at stage {stage}")
    return stages


def sample_weights_by_student(y, student_ids):
    """Each student's total training weight sums to one across its snapshots."""
    frame = pd.DataFrame({"sid": np.asarray(student_ids).astype(str)})
    counts = frame.sid.value_counts()
    return np.asarray([1.0 / counts[s] for s in frame.sid], dtype=float)


def fit_model(X, y, student_ids, cats, nums, cfg, seed):
    family = cfg["family"]
    prep = build_preprocessor(cats, nums, family)
    Xt = prep.fit_transform(X)
    weights = sample_weights_by_student(y, student_ids)

    if family == "logistic":
        model = LogisticRegression(
            C=0.5, solver="liblinear", max_iter=300,
            class_weight="balanced", random_state=seed,
        )
    elif family == "xgboost":
        sid = np.asarray(student_ids).astype(str)
        yy = np.asarray(y, dtype=int)
        labels_by_student = {}
        for s, label in zip(sid, yy):
            labels_by_student.setdefault(s, int(label))
        n_pos = sum(v == 1 for v in labels_by_student.values())
        n_neg = sum(v == 0 for v in labels_by_student.values())
        params = {k: cfg[k] for k in (
            "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma",
        )}
        model = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=1, random_state=seed, verbosity=0,
            scale_pos_weight=float(n_neg / max(1, n_pos)), **params,
        )
    else:
        raise ValueError(f"Unsupported model family: {family}")
    model.fit(Xt, np.asarray(y, dtype=int), sample_weight=weights)
    return prep, model


def predict_scores(fitted, X):
    prep, model = fitted
    return np.asarray(model.predict_proba(prep.transform(X))[:, 1], dtype=float)


def max_score_per_student(df, row_scores, student_order):
    temp = pd.DataFrame({
        "student_id": df.student_id.astype(str).to_numpy(),
        "score": np.asarray(row_scores, dtype=float),
    })
    return temp.groupby("student_id").score.max().reindex(student_order).to_numpy(dtype=float)


def threshold_candidates(scores, n_quantiles=21):
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("inf")]
    qs = np.quantile(values, np.linspace(0, 1, n_quantiles))
    return sorted(set(qs.tolist() + [float(np.nextafter(values.max(), np.inf))]))


def binary_metrics(y, flags):
    y = np.asarray(y, dtype=int)
    flags = np.asarray(flags, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, flags.astype(int), labels=[0, 1]).ravel()
    precision = precision_score(y, flags, zero_division=0)
    recall = recall_score(y, flags, zero_division=0)
    return {
        "n_test": int(len(y)),
        "positive_support": int(y.sum()),
        "negative_support": int((y == 0).sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(y, flags, zero_division=0)),
        "accuracy": float(accuracy_score(y, flags)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, flags)),
        "mcc": float(matthews_corrcoef(y, flags)) if len(np.unique(flags)) > 1 else 0.0,
        "flagged": int(flags.sum()),
        "flag_rate": float(flags.mean()) if len(flags) else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def ranking_metrics(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    yy, ss = y[ok], scores[ok]
    out = {"n_scored": int(ok.sum()), "average_precision": np.nan, "roc_auc": np.nan,
           "brier_score": np.nan, "log_loss": np.nan}
    if len(yy) and len(np.unique(yy)) == 2:
        out["average_precision"] = float(average_precision_score(yy, ss))
        out["roc_auc"] = float(roc_auc_score(yy, ss))
        # These are raw model scores; class weighting may make them uncalibrated.
        if np.all((ss >= 0) & (ss <= 1)):
            out["brier_score"] = float(brier_score_loss(yy, ss))
            out["log_loss"] = float(log_loss(yy, np.clip(ss, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return out


def select_joint_thresholds(y, stage_scores, precision_floor, n_quantiles=9):
    """Select joint thresholds using fast count arithmetic on development OOF scores."""
    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    candidates = {
        s: threshold_candidates(stage_scores[s], n_quantiles=n_quantiles)
        for s in ("A", "B", "C")
    }
    best = None
    best_fallback = None
    for ta in candidates["A"]:
        fa = np.isfinite(stage_scores["A"]) & (stage_scores["A"] >= ta)
        for tb in candidates["B"]:
            fab = fa | (np.isfinite(stage_scores["B"]) & (stage_scores["B"] >= tb))
            for tc in candidates["C"]:
                flags = fab | (np.isfinite(stage_scores["C"]) & (stage_scores["C"] >= tc))
                flagged = int(flags.sum())
                tp = int(np.sum(flags & (y == 1)))
                fp = flagged - tp
                fn = positives - tp
                tn = int(len(y) - tp - fp - fn)
                precision = tp / flagged if flagged else 0.0
                recall = tp / positives if positives else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                metrics = {
                    "n_test": int(len(y)), "positive_support": positives,
                    "negative_support": int(len(y) - positives),
                    "prevalence": float(y.mean()), "precision": precision,
                    "recall": recall, "f1": f1,
                    "accuracy": (tp + tn) / len(y) if len(y) else np.nan,
                    "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                    "balanced_accuracy": (recall + (tn / (tn + fp) if (tn + fp) else 0.0)) / 2,
                    "mcc": np.nan, "flagged": flagged,
                    "flag_rate": flagged / len(y) if len(y) else np.nan,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                }
                if flagged and precision + 1e-12 >= precision_floor:
                    key = (recall, precision, -flagged, f1)
                    if best is None or key > best["key"]:
                        best = {"key": key, "thresholds": {"A": ta, "B": tb, "C": tc},
                                "metrics": metrics, "flags": flags, "fallback_used": False}
                fkey = (f1, recall, precision, -flagged)
                if best_fallback is None or fkey > best_fallback["key"]:
                    best_fallback = {"key": fkey, "thresholds": {"A": ta, "B": tb, "C": tc},
                                     "metrics": metrics, "flags": flags, "fallback_used": True}
    return best if best is not None else best_fallback


def empirical_percentile(train_scores, scores):
    ref = np.sort(np.asarray(train_scores, dtype=float))
    ref = ref[np.isfinite(ref)]
    out = np.full(len(scores), np.nan, dtype=float)
    ok = np.isfinite(scores)
    if len(ref):
        out[ok] = np.searchsorted(ref, np.asarray(scores)[ok], side="right") / len(ref)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/cascade_snapshots")
    parser.add_argument("--output-dir", default="outputs/final_cascade_evaluation")
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--precision-floor", type=float, default=0.40)
    parser.add_argument("--threshold-quantiles", type=int, default=21)
    args = parser.parse_args()

    root = Path.cwd()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(input_dir / "snapshot_audit.csv")[
        ["student_id", "dropout_binary_audit_only"]
    ].drop_duplicates("student_id")
    audit = audit.rename(columns={"dropout_binary_audit_only": "eventual_dropout"})
    audit["student_id"] = audit.student_id.astype(str)
    audit["eventual_dropout"] = audit.eventual_dropout.astype(int)
    stages = make_stage_data(input_dir, audit)

    all_metrics, all_thresholds, all_predictions = [], [], []
    for repeat, seed in enumerate(args.outer_seeds, 1):
        print(f"\n=== Final cascade evaluation {repeat}/{len(args.outer_seeds)}; seed={seed} ===", flush=True)
        meta = audit.sort_values("student_id").reset_index(drop=True)
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_idx, te_idx = next(outer.split(meta[["student_id"]], meta.eventual_dropout))
        dev_meta = meta.iloc[tr_idx].reset_index(drop=True)
        test_meta = meta.iloc[te_idx].reset_index(drop=True)
        dev_ids = set(dev_meta.student_id)
        test_ids = set(test_meta.student_id)
        assert not (dev_ids & test_ids)
        dev_order = dev_meta.student_id.tolist()
        test_order = test_meta.student_id.tolist()
        y_dev = dev_meta.eventual_dropout.to_numpy(dtype=int)
        y_test = test_meta.eventual_dropout.to_numpy(dtype=int)

        n_folds = min(args.inner_folds, int(y_dev.sum()), int((y_dev == 0).sum()))
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + 7)
        folds = list(cv.split(dev_meta[["student_id"]], y_dev, groups=dev_meta.student_id))

        dev_stage_scores, stage_feature_data = {}, {}
        # Generate out-of-fold scores for each fixed stage model on development students.
        for stage in ("A", "B", "C"):
            cfg = STAGE_CONFIGS[stage]
            df = stages[stage]
            X, cats, nums = feature_frame(df, stage, cfg["feature_set"])
            row_ids = df.student_id.astype(str).to_numpy()
            row_y = df.target.to_numpy(dtype=int)
            row_oof = np.full(len(df), np.nan, dtype=float)
            for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
                train_students = set(dev_meta.student_id.iloc[fit_idx])
                val_students = set(dev_meta.student_id.iloc[val_idx])
                train_mask = row_ids.astype(str)
                train_mask = np.isin(row_ids, list(train_students))
                val_mask = np.isin(row_ids, list(val_students))
                if not train_mask.any() or not val_mask.any():
                    continue
                fitted = fit_model(
                    X.loc[train_mask].reset_index(drop=True),
                    row_y[train_mask],
                    row_ids[train_mask],
                    cats, nums, cfg, seed + fold_idx,
                )
                row_oof[val_mask] = predict_scores(fitted, X.loc[val_mask].reset_index(drop=True))
            dev_rows_mask = np.isin(row_ids, list(dev_ids))
            dev_stage_scores[stage] = max_score_per_student(
                df.loc[dev_rows_mask].reset_index(drop=True),
                row_oof[dev_rows_mask],
                dev_order,
            )
            stage_feature_data[stage] = (X, cats, nums)
            print(f"  OOF scores ready: {stage} ({cfg['name']})", flush=True)

        # Select thresholds using development OOF scores only.
        selected = select_joint_thresholds(y_dev, dev_stage_scores, args.precision_floor, n_quantiles=args.threshold_quantiles)
        thresholds = selected["thresholds"]
        for stage in ("A", "B", "C"):
            all_thresholds.append({
                "repeat": repeat, "seed": seed, "stage": stage,
                "model": STAGE_CONFIGS[stage]["name"],
                "feature_set": STAGE_CONFIGS[stage]["feature_set"],
                "threshold": thresholds[stage],
                "development_precision_floor": args.precision_floor,
                "development_precision_floor_feasible": not selected.get("fallback_used", False),
            })
        print("  thresholds:", {k: round(v, 5) for k, v in thresholds.items()}, flush=True)

        # Refit each fixed model on all development students and score held-out snapshots.
        test_stage_scores = {}
        for stage in ("A", "B", "C"):
            cfg = STAGE_CONFIGS[stage]
            df = stages[stage]
            X, cats, nums = stage_feature_data[stage]
            row_ids = df.student_id.astype(str).to_numpy()
            row_y = df.target.to_numpy(dtype=int)
            train_mask = np.isin(row_ids, list(dev_ids))
            test_mask = np.isin(row_ids, list(test_ids))
            fitted = fit_model(
                X.loc[train_mask].reset_index(drop=True),
                row_y[train_mask],
                row_ids[train_mask],
                cats, nums, cfg, seed + 999,
            )
            row_test = predict_scores(fitted, X.loc[test_mask].reset_index(drop=True))
            test_stage_scores[stage] = max_score_per_student(
                df.loc[test_mask].reset_index(drop=True), row_test, test_order
            )

        stage_flags = {
            stage: np.isfinite(test_stage_scores[stage]) &
                   (test_stage_scores[stage] >= thresholds[stage])
            for stage in ("A", "B", "C")
        }
        cumulative_flags = {
            "A_only": stage_flags["A"],
            "A_plus_B": stage_flags["A"] | stage_flags["B"],
            "A_plus_B_plus_C": stage_flags["A"] | stage_flags["B"] | stage_flags["C"],
        }

        # Ranking metrics for individual stages use only students with a score at that stage.
        for scope, stage in [("stage_A_standalone", "A"), ("stage_B_standalone", "B"), ("stage_C_standalone", "C")]:
            eligible = np.isfinite(test_stage_scores[stage])
            y_scope = y_test.copy()
            # Binary alert metrics apply to the full test cohort; unscored students are unflagged.
            bm = binary_metrics(y_test, stage_flags[stage])
            rm = ranking_metrics(y_test[eligible], test_stage_scores[stage][eligible])
            all_metrics.append({
                "repeat": repeat, "seed": seed, "scope": scope,
                "model_A": STAGE_CONFIGS["A"]["name"],
                "model_B": STAGE_CONFIGS["B"]["name"],
                "model_C": STAGE_CONFIGS["C"]["name"],
                "threshold_A": thresholds["A"], "threshold_B": thresholds["B"], "threshold_C": thresholds["C"],
                "scored_student_coverage": int(eligible.sum()),
                **bm, **rm,
            })

        # Cumulative alert-policy metrics at each cascade step.
        for scope, flags in cumulative_flags.items():
            bm = binary_metrics(y_test, flags)
            if scope == "A_only":
                raw = test_stage_scores["A"]
                norm = empirical_percentile(dev_stage_scores["A"], raw)
            elif scope == "A_plus_B":
                norm_a = empirical_percentile(dev_stage_scores["A"], test_stage_scores["A"])
                norm_b = empirical_percentile(dev_stage_scores["B"], test_stage_scores["B"])
                norm = np.fmax(norm_a, norm_b)
            else:
                norm_a = empirical_percentile(dev_stage_scores["A"], test_stage_scores["A"])
                norm_b = empirical_percentile(dev_stage_scores["B"], test_stage_scores["B"])
                norm_c = empirical_percentile(dev_stage_scores["C"], test_stage_scores["C"])
                norm = np.fmax(np.fmax(norm_a, norm_b), norm_c)
            rm = ranking_metrics(y_test, norm)
            all_metrics.append({
                "repeat": repeat, "seed": seed, "scope": f"cascade_{scope}",
                "model_A": STAGE_CONFIGS["A"]["name"],
                "model_B": STAGE_CONFIGS["B"]["name"],
                "model_C": STAGE_CONFIGS["C"]["name"],
                "threshold_A": thresholds["A"], "threshold_B": thresholds["B"], "threshold_C": thresholds["C"],
                "scored_student_coverage": int(np.isfinite(norm).sum()),
                "precision_floor_met_on_test": bool(bm["flagged"] > 0 and bm["precision"] + 1e-12 >= args.precision_floor),
                **bm, **rm,
            })

        for i, sid in enumerate(test_order):
            all_predictions.append({
                "repeat": repeat, "seed": seed, "student_id": sid, "target": int(y_test[i]),
                "score_A": test_stage_scores["A"][i],
                "score_B": test_stage_scores["B"][i],
                "score_C": test_stage_scores["C"][i],
                "flag_A": int(stage_flags["A"][i]),
                "flag_B": int(stage_flags["B"][i]),
                "flag_C": int(stage_flags["C"][i]),
                "flag_after_A": int(cumulative_flags["A_only"][i]),
                "flag_after_B": int(cumulative_flags["A_plus_B"][i]),
                "flag_end_to_end": int(cumulative_flags["A_plus_B_plus_C"][i]),
            })
        end = binary_metrics(y_test, cumulative_flags["A_plus_B_plus_C"])
        print(
            f"  test end-to-end: precision={end['precision']:.3f}, recall={end['recall']:.3f}, "
            f"F1={end['f1']:.3f}, flagged={end['flagged']}/{end['n_test']}",
            flush=True,
        )

    metrics_df = pd.DataFrame(all_metrics)
    thresholds_df = pd.DataFrame(all_thresholds)
    predictions_df = pd.DataFrame(all_predictions)
    metrics_df.to_csv(output_dir / "stepwise_test_metrics.csv", index=False)
    metrics_df[metrics_df.scope == "cascade_A_plus_B_plus_C"].to_csv(
        output_dir / "end_to_end_metrics_by_split.csv", index=False
    )
    thresholds_df.to_csv(output_dir / "selected_thresholds.csv", index=False)
    predictions_df.to_csv(output_dir / "outer_test_student_predictions.csv", index=False)

    summary = metrics_df.groupby("scope").agg({
        "precision": ["mean", "std"], "recall": ["mean", "std"],
        "f1": ["mean", "std"], "accuracy": ["mean", "std"],
        "specificity": ["mean", "std"], "balanced_accuracy": ["mean", "std"],
        "mcc": ["mean", "std"], "average_precision": ["mean", "std"],
        "roc_auc": ["mean", "std"], "brier_score": ["mean", "std"],
        "log_loss": ["mean", "std"], "flagged": ["mean", "std"],
        "flag_rate": ["mean", "std"], "tp": ["mean", "std"],
        "fp": ["mean", "std"], "fn": ["mean", "std"], "tn": ["mean", "std"],
        "n_scored": ["mean", "std"], "scored_student_coverage": ["mean", "std"],
    }).reset_index()
    summary.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in summary.columns]
    summary.to_csv(output_dir / "stepwise_test_summary.csv", index=False)

    end_rows = metrics_df[metrics_df.scope == "cascade_A_plus_B_plus_C"]
    end_summary = {
        "outer_seeds": args.outer_seeds,
        "precision_floor_selected_on_development": args.precision_floor,
        "mean_precision": float(end_rows.precision.mean()),
        "mean_recall": float(end_rows.recall.mean()),
        "mean_f1": float(end_rows.f1.mean()),
        "mean_accuracy": float(end_rows.accuracy.mean()),
        "mean_specificity": float(end_rows.specificity.mean()),
        "mean_balanced_accuracy": float(end_rows.balanced_accuracy.mean()),
        "mean_mcc": float(end_rows.mcc.mean()),
        "mean_average_precision": float(end_rows.average_precision.mean()),
        "mean_roc_auc": float(end_rows.roc_auc.mean()),
        "mean_flagged": float(end_rows.flagged.mean()),
        "mean_tp": float(end_rows.tp.mean()), "mean_fp": float(end_rows.fp.mean()),
        "mean_fn": float(end_rows.fn.mean()), "mean_tn": float(end_rows.tn.mean()),
    }
    (output_dir / "end_to_end_summary.json").write_text(json.dumps(end_summary, indent=2))

    report = [
        "# Final A/B/C cascade evaluation",
        "",
        "## Fixed configuration",
        "- A: class-weighted logistic regression; demographics/signup-calendar features.",
        "- B: class-weighted logistic regression; compact + per-lesson features.",
        "- C: depth-2 XGBoost with `scale_pos_weight` calculated from unique training students; compact + per-lesson features.",
        "- C uses XGBoost rather than the degree-3 polynomial SVM because the existing development OOF comparisons were approximately 0.529 AP for weighted depth-2 XGBoost, 0.521 for bagged degree-3 SVM, and 0.493 for class-weighted degree-3 SVM. This is a fixed final candidate, not a claim that XGBoost wins on every split.",
        "- Existing eventual-dropout target, snapshot builder, feature construction, and A/B/C cascade are unchanged.",
        "",
        "## Evaluation protocol",
        f"- Outer student-level 80/20 stratified holdouts: seeds {args.outer_seeds}. The splits may overlap across seeds.",
        f"- Thresholds are selected only from {args.inner_folds}-fold grouped OOF scores on each outer development set.",
        f"- Joint threshold selection maximizes recall subject to a {args.precision_floor:.0%} development precision floor; if infeasible, the script falls back to the best F1 threshold combination and records this.",
        "- Stage B/C snapshot scores are aggregated by maximum per student, matching the existing cascade policy.",
        "- All binary policy metrics are evaluated on the full outer test cohort. Stage-specific AP/AUC are computed among students with a score available at that stage; `scored_student_coverage` reports this coverage.",
        "- End-to-end AP/AUC use the maximum of development-OOF-referenced percentile ranks across available stages, since raw scores from different models are not directly comparable. This composite rank is not a calibrated probability.",
        "- Brier score and log loss are reported only for stage-level raw model scores in [0,1]. Class weighting can make these scores poorly calibrated; interpret them cautiously.",
        "",
        "## Stepwise metrics (mean/std across outer splits)",
        summary.to_markdown(index=False),
        "",
        "## End-to-end results by split",
        end_rows[["seed", "precision", "recall", "f1", "accuracy", "specificity", "balanced_accuracy", "mcc", "average_precision", "roc_auc", "flagged", "tp", "fp", "fn", "tn"]].to_markdown(index=False),
        "",
        "## Mean end-to-end summary",
        "```json",
        json.dumps(end_summary, indent=2),
        "```",
        "",
        "## Interpretation cautions",
        "- The dataset has 500 students and 125 eventual-dropout positives; three overlapping holdouts are internal robustness checks, not independent external validation.",
        "- There is no true dropout timestamp. Results predict the supplied eventual-dropout label and do not establish warning lead time or intervention impact.",
        "- Because B/C use a maximum over available snapshots, students with more snapshots have more opportunities to cross a threshold.",
        "- The development precision floor is a threshold-selection constraint, not a guarantee on test or future data.",
        "- Class-weighted logistic/XGBoost scores are not guaranteed to be calibrated probabilities. Ranking metrics and thresholded metrics answer different questions.",
        "",
    ]
    (output_dir / "FINAL_CASCADE_EVALUATION.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved final metrics to: {output_dir}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

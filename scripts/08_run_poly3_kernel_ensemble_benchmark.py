#!/usr/bin/env python3
"""Stage-C experiment: degree-3 polynomial SVM with bagging/boosting.

Keeps the existing A/B/C cascade and snapshot tables unchanged. This script
benchmarks Stage C only, using the same C2 rolling + censored snapshots,
student-grouped outer split, grouped OOF model selection, and student-level
max-score aggregation used by script 07.

Models:
  * class-weighted logistic regression baseline
  * Gradient Boosting baseline
  * degree-3 polynomial SVM (unweighted and class-weighted)
  * student-cluster bagging of degree-3 SVMs (average decision margins)
  * AdaBoost (SAMME) using a degree-3 SVM as the base estimator

All feature preprocessing is fitted inside each training fold. Bagging resamples
whole students, retaining all their available C-prefix rows. The outer test
students are not used for model or threshold selection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "07_run_kernel_svm_benchmark.py"
spec = importlib.util.spec_from_file_location("kernel_benchmark_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)


def student_normalized_weights(student_ids: np.ndarray) -> np.ndarray:
    """Each student's available prefix rows sum to weight 1."""
    sid = np.asarray(student_ids).astype(str)
    counts = pd.Series(sid).value_counts().to_dict()
    return np.asarray([1.0 / counts[s] for s in sid], dtype=float)


def student_group_bagging_fit(Xt, y, student_ids, cfg, seed):
    """Fit a cluster-bootstrap ensemble of degree-3 polynomial SVCs.

    A bootstrap unit is a student, not a row. Every sampled copy of a student
    contributes total weight 1 (or class-balanced group weight), regardless of
    how many C-prefix rows that student has.
    """
    rng = np.random.default_rng(seed)
    sid = np.asarray(student_ids).astype(str)
    y = np.asarray(y, dtype=int)
    groups: dict[str, np.ndarray] = {}
    for student in pd.unique(sid):
        groups[student] = np.flatnonzero(sid == student)
    students = np.asarray(list(groups), dtype=object)
    label_by_student = {s: int(y[groups[s][0]]) for s in students}
    for s in students:
        if np.unique(y[groups[s]]).size != 1:
            raise ValueError(f"C target varies within student {s}")

    n_estimators = int(cfg.get("n_estimators", 21))
    models = []
    for b in range(n_estimators):
        sampled = rng.choice(students, size=len(students), replace=True)
        selected_labels = np.asarray([label_by_student[s] for s in sampled], dtype=int)
        class_counts = {c: int(np.sum(selected_labels == c)) for c in (0, 1)}
        ix_parts, weight_parts = [], []
        for s in sampled:
            ix = groups[s]
            # Each sampled student-copy has total weight 1 by default.
            group_weight = 1.0
            if cfg.get("balance_groups", True) and class_counts[0] and class_counts[1]:
                # Equal total training weight per class, defined over students.
                group_weight = len(sampled) / (2.0 * class_counts[label_by_student[s]])
            ix_parts.append(ix)
            weight_parts.append(np.full(len(ix), group_weight / max(1, len(ix)), dtype=float))
        boot_ix = np.concatenate(ix_parts)
        boot_weights = np.concatenate(weight_parts)
        estimator = SVC(
            kernel="poly", degree=3, C=float(cfg.get("C", 1.0)),
            gamma=cfg.get("gamma", "scale"), coef0=float(cfg.get("coef0", 1.0)),
            class_weight=None, probability=False, cache_size=512,
            random_state=seed + b,
        )
        estimator.fit(Xt[boot_ix], y[boot_ix], sample_weight=boot_weights)
        models.append(estimator)
    return models


def fit_model(X, y, student_ids, cats, nums, cfg, seed):
    family = cfg["family"]
    prep_family = "kernel_svm" if family in {"poly3_svm", "bagged_poly3_svm", "adaboost_poly3_svm"} else family
    prep = base.build_preprocessor(cats, nums, prep_family)
    Xt = prep.fit_transform(X)
    y = np.asarray(y, dtype=int)
    sid = np.asarray(student_ids).astype(str)

    if family == "bagged_poly3_svm":
        fitted = student_group_bagging_fit(Xt, y, sid, cfg, seed)
        return prep, fitted

    if family == "poly3_svm":
        est = SVC(
            kernel="poly", degree=3, C=float(cfg.get("C", 1.0)),
            gamma=cfg.get("gamma", "scale"), coef0=float(cfg.get("coef0", 1.0)),
            class_weight="balanced" if cfg.get("balance", False) else None,
            probability=False, cache_size=512, random_state=seed,
        )
        est.fit(Xt, y, sample_weight=student_normalized_weights(sid))
        return prep, est

    if family == "adaboost_poly3_svm":
        # SAMME uses hard predictions from the degree-3 SVM. SVC accepts
        # sample_weight, which AdaBoost updates on each boosting round.
        base_svc = SVC(
            kernel="poly", degree=3, C=float(cfg.get("C", 0.5)),
            gamma=cfg.get("gamma", "scale"), coef0=float(cfg.get("coef0", 1.0)),
            class_weight="balanced" if cfg.get("balance", True) else None,
            probability=False, cache_size=512, random_state=seed,
        )
        est = AdaBoostClassifier(
            estimator=base_svc, n_estimators=int(cfg.get("n_estimators", 10)),
            learning_rate=float(cfg.get("learning_rate", 0.35)), random_state=seed,
        )
        est.fit(Xt, y, sample_weight=student_normalized_weights(sid))
        return prep, est

    # Controls use the existing project's estimator definitions and preprocessing.
    est_cfg = dict(cfg)
    est = base.build_estimator(est_cfg, seed)
    est.fit(Xt, y, sample_weight=student_normalized_weights(sid))
    return prep, est


def predict_model(fitted, X):
    prep, est = fitted
    Xt = prep.transform(X)
    if isinstance(est, list):
        margins = np.vstack([m.decision_function(Xt).reshape(-1) for m in est])
        return np.mean(margins, axis=0)
    if isinstance(est, SVC):
        return np.asarray(est.decision_function(Xt), dtype=float).reshape(-1)
    if hasattr(est, "decision_function") and callable(est.decision_function):
        return np.asarray(est.decision_function(Xt), dtype=float).reshape(-1)
    return np.asarray(est.predict_proba(Xt)[:, 1], dtype=float)


def student_max_scores(df, row_scores, ids):
    tmp = pd.DataFrame({"student_id": df.student_id.astype(str).to_numpy(), "score": np.asarray(row_scores, float)})
    return tmp.groupby("student_id").score.max().reindex(ids).to_numpy(dtype=float)


def select_threshold(y, scores, precision_floor=0.40, quantiles=21):
    """Select threshold from development OOF scores only; maximize recall first."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    if valid.sum() == 0 or np.unique(y[valid]).size < 2:
        return float("inf"), {"feasible": False, "precision": 0.0, "recall": 0.0, "flagged": 0}
    candidates = np.unique(np.r_[np.quantile(scores[valid], np.linspace(0, 1, quantiles)), np.nextafter(np.max(scores[valid]), np.inf)])
    best = None
    for threshold in candidates:
        flags = valid & (scores >= threshold)
        n_flag = int(flags.sum())
        if n_flag == 0:
            continue
        p = float(precision_score(y, flags, zero_division=0))
        if p + 1e-12 < precision_floor:
            continue
        r = float(recall_score(y, flags, zero_division=0))
        key = (r, p, -n_flag)
        if best is None or key > best[0]:
            best = (key, float(threshold), {"feasible": True, "precision": p, "recall": r, "flagged": n_flag})
    if best is None:
        return float("inf"), {"feasible": False, "precision": 0.0, "recall": 0.0, "flagged": 0}
    return best[1], best[2]


def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    flags = valid & (scores >= threshold)
    if valid.sum() and np.unique(y[valid]).size == 2:
        ap = float(average_precision_score(y[valid], scores[valid]))
    else:
        ap = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, flags.astype(int), labels=[0, 1]).ravel()
    return {
        "ap": ap,
        "precision": float(precision_score(y, flags, zero_division=0)),
        "recall": float(recall_score(y, flags, zero_division=0)),
        "f1": float(f1_score(y, flags, zero_division=0)),
        "flagged": int(flags.sum()), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn), "n_students": int(len(y)),
        "n_scored": int(valid.sum()), "positives": int(y.sum()),
    }


def get_configs(bagging_estimators=21, adaboost_estimators=10):
    return [
        {"name": "logistic_class_weight", "family": "logistic", "balance": "balanced", "sampling": "none"},
        {"name": "gradient_boosting", "family": "gradient_boosting", "balance": "none", "sampling": "none"},
        {"name": "poly3_svm_unweighted", "family": "poly3_svm", "balance": False},
        {"name": "poly3_svm_class_weighted", "family": "poly3_svm", "balance": True},
        {"name": f"poly3_svm_bagged_{bagging_estimators}_student_bootstraps", "family": "bagged_poly3_svm", "n_estimators": bagging_estimators, "balance_groups": True},
        {"name": f"poly3_svm_adaboost_{adaboost_estimators}_rounds", "family": "adaboost_poly3_svm", "n_estimators": adaboost_estimators, "learning_rate": 0.35, "balance": True},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/cascade_snapshots")
    parser.add_argument("--output-dir", default="outputs/poly3_kernel_ensemble")
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--precision-floor", type=float, default=0.40)
    parser.add_argument("--threshold-quantiles", type=int, default=21)
    parser.add_argument("--bagging-estimators", type=int, default=21)
    parser.add_argument("--adaboost-estimators", type=int, default=10)
    args = parser.parse_args()

    root = Path.cwd()
    inp = Path(args.input_dir); out = Path(args.output_dir)
    if not inp.is_absolute(): inp = root / inp
    if not out.is_absolute(): out = root / out
    out.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(inp / "snapshot_audit.csv")[["student_id", "dropout_binary_audit_only"]].drop_duplicates("student_id")
    audit = audit.rename(columns={"dropout_binary_audit_only": "eventual_dropout"})
    audit.student_id = audit.student_id.astype(str); audit.eventual_dropout = audit.eventual_dropout.astype(int)
    stages = base.make_stage_data(inp, audit)
    cdf = stages["C"].copy().reset_index(drop=True)
    # The same feature family that performed well in the previous Stage C kernel tests.
    X, cats, nums = base.feature_frame(cdf, "C", "compact_plus_module")
    row_sid = cdf.student_id.astype(str).to_numpy()
    row_y = cdf.target.to_numpy(dtype=int)
    configs = get_configs(args.bagging_estimators, args.adaboost_estimators)

    all_rows = []
    all_oof = []
    all_test_preds = []
    for repeat, seed in enumerate(args.outer_seeds, 1):
        print(f"\n=== Repeat {repeat}/{len(args.outer_seeds)} seed={seed} ===", flush=True)
        meta = audit.sort_values("student_id").reset_index(drop=True)
        outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        tr_ix, te_ix = next(outer.split(meta.student_id.to_numpy().reshape(-1, 1), meta.eventual_dropout.to_numpy()))
        dev_meta = meta.iloc[tr_ix].reset_index(drop=True)
        test_meta = meta.iloc[te_ix].reset_index(drop=True)
        dev_ids = set(dev_meta.student_id.astype(str)); test_ids = set(test_meta.student_id.astype(str))
        assert not dev_ids & test_ids
        dev_order = dev_meta.student_id.tolist(); test_order = test_meta.student_id.tolist()
        y_dev = dev_meta.eventual_dropout.to_numpy(dtype=int); y_test = test_meta.eventual_dropout.to_numpy(dtype=int)
        nfold = min(args.inner_folds, int(y_dev.sum()), int((y_dev == 0).sum()))
        folds = list(StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=seed + 7).split(
            dev_meta.student_id.to_numpy().reshape(-1, 1), y_dev, groups=dev_meta.student_id.to_numpy()))
        oof_scores_by_cfg = {}
        config_status = {}

        for ci, cfg in enumerate(configs, 1):
            print(f"  OOF {ci}/{len(configs)}: {cfg['name']}", flush=True)
            row_oof = np.full(len(cdf), np.nan, dtype=float)
            failure = None
            for fold, (train_meta_ix, val_meta_ix) in enumerate(folds, 1):
                train_ids = set(dev_meta.student_id.iloc[train_meta_ix].astype(str))
                val_ids = set(dev_meta.student_id.iloc[val_meta_ix].astype(str))
                tr = np.isin(row_sid, list(train_ids)); va = np.isin(row_sid, list(val_ids))
                if not tr.any() or not va.any(): continue
                try:
                    fitted = fit_model(X.loc[tr].reset_index(drop=True), row_y[tr], row_sid[tr], cats, nums, cfg, seed + fold * 101)
                    row_oof[va] = predict_model(fitted, X.loc[va].reset_index(drop=True))
                except Exception as exc:
                    failure = f"fold {fold}: {type(exc).__name__}: {exc}"
                    break
            dev_mask = cdf.student_id.astype(str).isin(dev_ids).to_numpy()
            scores = student_max_scores(cdf.loc[dev_mask].reset_index(drop=True), row_oof[dev_mask], dev_order)
            if failure:
                scores[:] = np.nan
                config_status[cfg["name"]] = failure
            else:
                config_status[cfg["name"]] = "ok"
            oof_scores_by_cfg[cfg["name"]] = scores
            valid = np.isfinite(scores)
            ap = float(average_precision_score(y_dev[valid], scores[valid])) if valid.sum() and np.unique(y_dev[valid]).size == 2 else float("nan")
            all_oof.append({"repeat": repeat, "seed": seed, "config": cfg["name"], "family": cfg["family"], "status": config_status[cfg["name"]], "error": failure or "", "oof_ap": ap,
                            "n_dev_students": int(len(y_dev)), "n_dev_scored": int(valid.sum()), "dev_prevalence_scored": float(y_dev[valid].mean()) if valid.any() else float("nan")})

        # Select each model's threshold independently from its OOF development scores.
        for cfg in configs:
            name = cfg["name"]
            oof_scores = oof_scores_by_cfg[name]
            threshold, dev_op = select_threshold(y_dev, oof_scores, args.precision_floor, args.threshold_quantiles)
            tr = np.isin(row_sid, list(dev_ids)); te = np.isin(row_sid, list(test_ids))
            fit_error = ""
            if config_status[name] != "ok":
                fit_error = config_status[name]
                test_scores = np.full(len(test_order), np.nan, dtype=float)
            else:
                try:
                    fitted = fit_model(X.loc[tr].reset_index(drop=True), row_y[tr], row_sid[tr], cats, nums, cfg, seed + 9999)
                    row_test_scores = predict_model(fitted, X.loc[te].reset_index(drop=True))
                    test_scores = student_max_scores(cdf.loc[te].reset_index(drop=True), row_test_scores, test_order)
                except Exception as exc:
                    fit_error = f"{type(exc).__name__}: {exc}"
                    test_scores = np.full(len(test_order), np.nan, dtype=float)
            tm = metrics(y_test, test_scores, threshold)
            valid = np.isfinite(oof_scores)
            dev_ap = float(average_precision_score(y_dev[valid], oof_scores[valid])) if valid.sum() and np.unique(y_dev[valid]).size == 2 else float("nan")
            all_rows.append({"repeat": repeat, "seed": seed, "config": name, "family": cfg["family"], "status": "ok" if not fit_error else "failed", "error": fit_error,
                             "precision_floor": args.precision_floor, "selected_threshold_from_oof": threshold,
                             "development_threshold_feasible": dev_op["feasible"], "development_oof_precision_at_threshold": dev_op["precision"],
                             "development_oof_recall_at_threshold": dev_op["recall"], "development_oof_flagged": dev_op["flagged"],
                             "development_oof_ap": dev_ap, **{f"test_{k}": v for k, v in tm.items()},
                             "test_precision_floor_met": bool(tm["flagged"] > 0 and tm["precision"] + 1e-12 >= args.precision_floor)})
            for i, student in enumerate(test_order):
                all_test_preds.append({"repeat": repeat, "seed": seed, "config": name, "student_id": student,
                                       "target": int(y_test[i]), "score": float(test_scores[i]) if np.isfinite(test_scores[i]) else np.nan,
                                       "flag": int(np.isfinite(test_scores[i]) and test_scores[i] >= threshold), "threshold": threshold})

    results = pd.DataFrame(all_rows)
    oof_df = pd.DataFrame(all_oof)
    pred_df = pd.DataFrame(all_test_preds)
    results.to_csv(out / "stage_c_poly3_ensemble_test_metrics.csv", index=False)
    oof_df.to_csv(out / "stage_c_poly3_ensemble_oof_ap.csv", index=False)
    pred_df.to_csv(out / "stage_c_poly3_ensemble_test_predictions.csv", index=False)
    summary = results.groupby(["config", "family"]).agg(
        oof_ap_mean=("development_oof_ap", "mean"), oof_ap_std=("development_oof_ap", "std"),
        test_ap_mean=("test_ap", "mean"), test_precision_mean=("test_precision", "mean"),
        test_recall_mean=("test_recall", "mean"), test_flagged_mean=("test_flagged", "mean"),
        test_scored_mean=("test_n_scored", "mean"), precision_floor_pass_rate=("test_precision_floor_met", "mean"), repeats=("seed", "nunique")
    ).reset_index().sort_values("oof_ap_mean", ascending=False)
    summary.to_csv(out / "stage_c_poly3_ensemble_summary.csv", index=False)

    report = [
        "# Stage C: Degree-3 polynomial SVM with bagging and boosting", "",
        "## Objective", "Compare a degree-3 polynomial SVM with student-cluster bagging and AdaBoost, while preserving the existing cascade structure, Stage C snapshot data, target, and feature set.", "",
        "## Experiment design",
        f"- Outer split seeds: {args.outer_seeds}; each split holds out 20% of students.",
        f"- Inner validation: {args.inner_folds}-fold StratifiedGroupKFold over development students.",
        "- Stage C rows are the existing C2 rolling plus censored snapshots, labelled with the same eventual-dropout target used by the current kernel benchmark.",
        "- Feature set: compact + per-lesson/module features. Numeric scaling and imputation are fitted within each training fold.",
        "- Student-level evaluation: row-level scores are aggregated by the maximum score per student, matching the existing Stage C evaluation policy.",
        f"- Thresholds are selected from {args.threshold_quantiles} development OOF score quantiles only, with a {args.precision_floor:.0%} precision floor; test labels are not used for threshold selection. The more granular Stage-C-only threshold grid is used because this script selects one stage threshold rather than jointly optimizing the full A/B/C cascade.",
        f"- Bagging uses {args.bagging_estimators} student-cluster bootstrap models; each bootstrap samples students with replacement and keeps each sampled student's full set of C snapshots together.",
        f"- AdaBoost uses {args.adaboost_estimators} SAMME rounds with a degree-3 polynomial SVC base estimator. This is exploratory: a strong kernel SVM is not always an ideal weak learner for boosting.",
        "- SVM and ensemble decision margins are ranking scores, not calibrated probabilities.", "",
        "## Results summary", summary.to_markdown(index=False), "",
        "## Boosting diagnostic",
        "AdaBoost was included as an experimental combination, not assumed to work. With a degree-3 SVC base estimator, the base learner was worse than random under the current fold weighting in this run, so scikit-learn correctly refused to continue boosting. This configuration is retained in the output as a failed experiment rather than silently removed. A polynomial SVM is a relatively strong learner and is not necessarily a suitable weak learner for AdaBoost.",
        "",
        "## Interpretation notes",
        "- Compare development OOF AP and test AP, but also inspect precision, recall, number flagged, and precision-floor pass rate.",
        "- Outer splits may overlap in students and are robustness diagnostics rather than independent external validation.",
        "- If bagging helps, it suggests variance reduction across student-level resamples. If AdaBoost fails or does not improve, that is useful evidence; do not force an ensemble merely because it is more complex.",
        "- This script evaluates Stage C candidates only. It does not alter or reselect the Stage A/B models or change the cascade routing policy.",
        "- The dataset does not contain a true dropout timestamp, so results concern eventual labelled-dropout detection, not warning lead time.", ""
    ]
    # Record data-derived conclusions so the report stays aligned with the actual run.
    report += ["", "## Findings from this run"]
    valid_summary = summary[summary["oof_ap_mean"].notna()]
    if not valid_summary.empty:
        best_oof = valid_summary.sort_values("oof_ap_mean", ascending=False).iloc[0]
        report.append(f"- Highest mean development OOF AP was **{best_oof['config']}** at **{best_oof['oof_ap_mean']:.1%}**.")
    bag = summary[summary["family"] == "bagged_poly3_svm"]
    weighted_poly = summary[summary["config"] == "poly3_svm_class_weighted"]
    if not bag.empty and not weighted_poly.empty:
        b = bag.iloc[0]; w = weighted_poly.iloc[0]
        report.append(f"- Bagging OOF AP was {b['oof_ap_mean']:.1%}, versus {w['oof_ap_mean']:.1%} for the class-weighted polynomial SVM. However, mean held-out AP was {b['test_ap_mean']:.1%} for bagging versus {w['test_ap_mean']:.1%} for the class-weighted SVM.")
        report.append(f"- The class-weighted polynomial SVM met the selected precision floor on {w['precision_floor_pass_rate']:.0%} of outer splits; bagging did so on {b['precision_floor_pass_rate']:.0%}. This argues against declaring bagging a clear winner from three splits.")
    report.append("- Stage C AP is computed only on test students with available C snapshots; students without a C score are treated as not flagged for Stage-C-only precision/recall. This is not a complete A/B/C cascade metric.")
    failed = results[results["status"] != "ok"]
    if not failed.empty:
        report.append(f"- Failed model configurations: {', '.join(failed['config'].astype(str).unique())}. See the CSV error column for fold-level details.")
    report.append("")
    (out / "POLY3_KERNEL_ENSEMBLE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\nSummary:\n", summary.to_string(index=False))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

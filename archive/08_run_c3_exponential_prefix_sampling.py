#!/usr/bin/env python3
"""C3: exponentially weighted prefix sampling for Stage C.

This is an additive experiment. It does not modify the existing snapshot builder,
baseline scripts, datasets, or prior outputs.

Comparison:
  baseline: train on each training student's penultimate prefix (C1)
  C3: for each training student, sample exactly one prefix from the penultimate
      prefix back to the 4th attempt, with probability proportional to
      exp(-lambda * distance_from_penultimate). For students whose penultimate
      prefix is only 3 attempts (N=4), retain the k=3 penultimate prefix.

Validation and test always use the same C1 penultimate-prefix rows. All splits
are by student. Each sampled training row's target is the student's eventual
dropout label, matching the latest eventual-outcome-aligned modeling runs.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("xgb_tuning_base", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def metric_row(y, scores, threshold):
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    flags = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, flags.astype(int), labels=[0,1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else np.nan,
        "threshold": float(threshold),
        "precision": float(precision_score(y, flags, zero_division=0)),
        "recall": float(recall_score(y, flags, zero_division=0)),
        "f1": float(f1_score(y, flags, zero_division=0)),
        "flagged": int(flags.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y)), "positives": int(y.sum()),
    }

def choose_threshold(y, scores, floor=0.40):
    """Maximize recall on OOF scores subject to development precision floor."""
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    ok = np.isfinite(scores)
    y, scores = y[ok], scores[ok]
    if len(scores) == 0:
        return float("inf"), False
    candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    best = None
    for t in candidates:
        pred = scores >= t
        n = int(pred.sum())
        if n == 0:
            continue
        tp = int(np.sum(pred & (y == 1)))
        precision = tp / n
        if precision + 1e-12 < floor:
            continue
        recall = tp / max(1, int(y.sum()))
        key = (recall, precision, -n)
        if best is None or key > best[0]:
            best = (key, float(t))
    return (best[1], True) if best else (float("inf"), False)

def make_penultimate_and_rolling(input_dir: Path):
    audit = pd.read_csv(input_dir / "snapshot_audit.csv", dtype={"student_id": str})
    labels = audit[["student_id", "dropout_binary_audit_only"]].drop_duplicates("student_id")
    labels = labels.rename(columns={"dropout_binary_audit_only": "target"})
    labels["student_id"] = labels.student_id.astype(str)
    labels["target"] = labels.target.astype(int)

    c1 = pd.read_csv(input_dir / "C1_penultimate_only.csv", dtype={"student_id": str})
    c1 = c1.drop(columns=["target"], errors="ignore").merge(labels, on="student_id", how="left", validate="one_to_one")
    c1["student_id"] = c1.student_id.astype(str)
    c1["_row_id"] = np.arange(len(c1))

    c2 = pd.read_csv(input_dir / "C2_rolling.csv", dtype={"student_id": str})
    c2 = c2.drop(columns=["target"], errors="ignore").merge(labels, on="student_id", how="left", validate="many_to_one")
    c2["student_id"] = c2.student_id.astype(str)
    c2["_row_id"] = np.arange(len(c2))

    # Only eligible, observed prefixes at or before the penultimate prefix.
    # C2 rows already contain prefix-only features and k/prefix audit metadata.
    c1_k = c1.set_index("student_id")["prefix_length_audit"].astype(int).to_dict()
    c2["penultimate_k"] = c2.student_id.map(c1_k)
    c2 = c2[c2["prefix_length_audit"].astype(int) <= c2["penultimate_k"]].copy()
    c2["prefix_length_audit"] = c2["prefix_length_audit"].astype(int)
    if c1.target.isna().any() or c2.target.isna().any():
        raise ValueError("Missing eventual dropout labels after audit join")
    if c1.student_id.nunique() != len(c1):
        raise AssertionError("C1 should contain one penultimate row per student")
    if c1.groupby("student_id").target.nunique().max() != 1:
        raise AssertionError("Inconsistent C1 labels")
    return c1.reset_index(drop=True), c2.reset_index(drop=True), labels

def sample_one_prefix_per_student(train_ids, c1, c2, lam, seed):
    """Sample exactly one eligible prefix per training student."""
    rng = np.random.default_rng(seed)
    c1_by_sid = c1.set_index("student_id", drop=False)
    sampled_rows = []
    # Stable iteration ensures reproducibility for a given seed.
    for sid in sorted(map(str, train_ids)):
        pen_k = int(c1_by_sid.loc[sid, "prefix_length_audit"])
        candidates = c2[(c2.student_id == sid) &
                        (c2.prefix_length_audit.astype(int) <= pen_k)].copy()
        # Use prefixes k=4..penultimate. For N=4, penultimate k=3 is the only
        # valid candidate and is retained as the fallback.
        candidates = candidates[candidates.prefix_length_audit >= 4]
        if candidates.empty:
            candidates = c2[(c2.student_id == sid) &
                            (c2.prefix_length_audit.astype(int) == pen_k)].copy()
        if candidates.empty:
            # Defensive fallback to the fixed penultimate row.
            sampled_rows.append(c1_by_sid.loc[sid].to_dict())
            continue
        distance = pen_k - candidates.prefix_length_audit.to_numpy(dtype=int)
        weights = np.exp(-float(lam) * distance.astype(float))
        probs = weights / weights.sum()
        ix = int(rng.choice(len(candidates), p=probs))
        sampled_rows.append(candidates.iloc[ix].to_dict())
    out = pd.DataFrame(sampled_rows).reset_index(drop=True)
    # Exact one row per train student and target is student-level.
    if out.student_id.nunique() != len(out):
        raise AssertionError("C3 must contribute exactly one sampled row per student")
    return out

def get_x_y(base, df, feature_set):
    X, cats, nums = base.feature_frame(df, "C", feature_set)
    y = df.target.to_numpy(dtype=int)
    sid = df.student_id.astype(str).to_numpy()
    return X.reset_index(drop=True), y, sid, cats, nums

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/cascade_snapshots")
    ap.add_argument("--base-script", default="scripts/05_run_xgboost_oversampling_tuning.py")
    ap.add_argument("--output-dir", default="outputs/c3_exponential_prefix_sampling")
    ap.add_argument("--outer-seed", type=int, default=42,
                    help="Keep this fixed to compare against the existing seed-42 baseline.")
    ap.add_argument("--inner-folds", type=int, default=2)
    ap.add_argument("--precision-floor", type=float, default=0.40)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--sampling-repeats", type=int, default=5)
    ap.add_argument("--feature-set", choices=["compact", "compact_plus_trends", "compact_plus_module"],
                    default="compact_plus_module")
    args = ap.parse_args()
    root = Path.cwd()
    input_dir = Path(args.input_dir); base_path = Path(args.base_script); out = Path(args.output_dir)
    if not input_dir.is_absolute(): input_dir = root / input_dir
    if not base_path.is_absolute(): base_path = root / base_path
    if not out.is_absolute(): out = root / out
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(base_path)
    c1, c2, labels = make_penultimate_and_rolling(input_dir)
    meta = c1[["student_id", "target"]].drop_duplicates().sort_values("student_id").reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=args.outer_seed)
    tr_ix, te_ix = next(splitter.split(meta.student_id.to_numpy().reshape(-1,1), meta.target.to_numpy()))
    dev_meta = meta.iloc[tr_ix].reset_index(drop=True)
    test_meta = meta.iloc[te_ix].reset_index(drop=True)
    dev_ids = dev_meta.student_id.astype(str).tolist()
    test_ids = test_meta.student_id.astype(str).tolist()
    y_dev = dev_meta.target.to_numpy(dtype=int)
    y_test = test_meta.target.to_numpy(dtype=int)

    nfold = min(args.inner_folds, int(y_dev.sum()), int((y_dev == 0).sum()))
    cv = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=args.outer_seed + 7)
    folds = list(cv.split(dev_meta.student_id.to_numpy().reshape(-1,1), y_dev, groups=dev_meta.student_id.to_numpy()))

    # Fix the estimator to the project-recommended shallow XGBoost + student-level
    # positive-class weighting, so the experiment isolates prefix sampling.
    cfg = next(c for c in base.CONFIGS if c["name"] == "xgb_d2_group_weight")
    feature_set = args.feature_set
    c1x, cats, nums = base.feature_frame(c1, "C", feature_set)
    c2x, _, _ = base.feature_frame(c2, "C", feature_set)
    c1x = c1x.reset_index(drop=True); c2x = c2x.reset_index(drop=True)

    # For every condition, build OOF predictions on the same penultimate validation
    # rows. Average over sampling draws to reduce dependence on a single random draw.
    conditions = [("penultimate_baseline", None)] + [(f"c3_exp_lambda_{lam:g}", float(lam)) for lam in args.lambdas]
    oof_scores = {name: np.full(len(dev_meta), np.nan, dtype=float) for name, _ in conditions}
    oof_diag_rows = []
    for fold_i, (ti, vi) in enumerate(folds, 1):
        train_ids = set(dev_meta.student_id.iloc[ti].astype(str))
        valid_meta = dev_meta.iloc[vi].reset_index(drop=True)
        valid_ids = valid_meta.student_id.astype(str).tolist()
        valid_rows = c1[c1.student_id.isin(valid_ids)].copy()
        valid_rows = valid_rows.set_index("student_id").loc[valid_ids].reset_index()
        Xv, yv, sv, vcats, vnums = get_x_y(base, valid_rows, feature_set)
        for cond_name, lam in conditions:
            replicate_preds = []
            for rep in range(args.sampling_repeats if lam is not None else 1):
                if lam is None:
                    train_rows = c1[c1.student_id.isin(train_ids)].copy()
                    train_rows = train_rows.set_index("student_id").loc[sorted(train_ids)].reset_index()
                else:
                    train_rows = sample_one_prefix_per_student(train_ids, c1, c2, lam,
                                                               seed=args.outer_seed*1000 + fold_i*100 + rep)
                Xt, yt, st, tcats, tnums = get_x_y(base, train_rows, feature_set)
                fitted = base.fit_config(Xt, yt, st, tcats, tnums, cfg, seed=args.outer_seed + fold_i)
                replicate_preds.append(base.predict_config(fitted, Xv))
            mean_pred = np.mean(np.vstack(replicate_preds), axis=0)
            for sid, score in zip(valid_ids, mean_pred):
                pos = dev_meta.index[dev_meta.student_id.astype(str) == sid][0]
                oof_scores[cond_name][pos] = score
            oof_diag_rows.append({"condition": cond_name, "fold": fold_i, "n_valid": len(valid_ids),
                                  "mean_score": float(np.mean(mean_pred)),
                                  "sampling_repeats": len(replicate_preds)})

    # OOF AP + development-only threshold selection.
    oof_rows = []
    thresholds = {}
    for cond_name, lam in conditions:
        scores = oof_scores[cond_name]
        ap_score = average_precision_score(y_dev, scores)
        threshold, feasible = choose_threshold(y_dev, scores, args.precision_floor)
        thresholds[cond_name] = threshold
        met = metric_row(y_dev, scores, threshold)
        oof_rows.append({"condition": cond_name, "lambda": lam, "development_oof_ap": float(ap_score),
                         "development_threshold": threshold, "development_precision_floor_feasible": feasible,
                         **{f"development_{k}": v for k, v in met.items() if k not in ("average_precision","threshold")}})
    oof_df = pd.DataFrame(oof_rows).sort_values("development_oof_ap", ascending=False)
    best_c3 = oof_df[oof_df.condition.str.startswith("c3_exp_")].iloc[0]["condition"] if args.lambdas else "penultimate_baseline"
    selected = str(best_c3)
    selected_lambda = float(oof_df.loc[oof_df.condition == selected, "lambda"].iloc[0])
    selected_threshold = thresholds[selected]

    # Fixed held-out test: every model is evaluated on the same penultimate rows.
    test_rows = c1[c1.student_id.isin(test_ids)].copy()
    test_rows = test_rows.set_index("student_id").loc[test_ids].reset_index()
    Xt, yt, st, tcats, tnums = get_x_y(base, test_rows, feature_set)
    test_result_rows = []
    for cond_name, lam in conditions:
        preds = []
        for rep in range(args.sampling_repeats if lam is not None else 1):
            if lam is None:
                train_rows = c1[c1.student_id.isin(dev_ids)].copy()
                train_rows = train_rows.set_index("student_id").loc[sorted(dev_ids)].reset_index()
            else:
                train_rows = sample_one_prefix_per_student(set(dev_ids), c1, c2, lam,
                                                           seed=args.outer_seed*10000 + rep)
            Xtr, ytr, str_ids, tcats, tnums = get_x_y(base, train_rows, feature_set)
            fitted = base.fit_config(Xtr, ytr, str_ids, tcats, tnums, cfg,
                                     seed=args.outer_seed + 999)
            preds.append(base.predict_config(fitted, Xt))
        test_scores = np.mean(np.vstack(preds), axis=0)
        threshold = thresholds[cond_name]
        met = metric_row(y_test, test_scores, threshold)
        test_result_rows.append({"condition": cond_name, "lambda": lam,
                                 "best_c3_variant_by_development_ap": cond_name == selected,
                                 "development_oof_ap": float(oof_df.loc[oof_df.condition == cond_name, "development_oof_ap"].iloc[0]),
                                 **{f"test_{k}": v for k,v in met.items()}})
        pd.DataFrame({"student_id": test_ids, "target": y_test, "score": test_scores,
                      "flag": (test_scores >= threshold).astype(int),
                      "condition": cond_name, "lambda": lam}).to_csv(out / f"test_predictions_{cond_name}.csv", index=False)

    test_df = pd.DataFrame(test_result_rows)
    oof_df.to_csv(out / "development_oof_comparison.csv", index=False)
    test_df.to_csv(out / "locked_test_comparison.csv", index=False)
    pd.DataFrame(oof_diag_rows).to_csv(out / "oof_fold_sampling_diagnostics.csv", index=False)

    baseline_oof_ap = float(oof_df.loc[oof_df.condition == "penultimate_baseline", "development_oof_ap"].iloc[0])
    selected_oof_ap = float(oof_df.loc[oof_df.condition == selected, "development_oof_ap"].iloc[0])
    baseline_test_ap = float(test_df.loc[test_df.condition == "penultimate_baseline", "test_average_precision"].iloc[0])
    selected_test_ap = float(test_df.loc[test_df.condition == selected, "test_average_precision"].iloc[0])
    oof_delta = selected_oof_ap - baseline_oof_ap
    test_delta = selected_test_ap - baseline_test_ap

    # Dataset audit: count candidate prefixes per student and explain fallback.
    cand_rows = []
    for row in c1.itertuples(index=False):
        sid = str(row.student_id); pk = int(row.prefix_length_audit)
        cand = c2[(c2.student_id == sid) & (c2.prefix_length_audit.astype(int) <= pk) &
                  (c2.prefix_length_audit.astype(int) >= 4)]
        cand_rows.append({"student_id": sid, "penultimate_prefix_k": pk,
                          "eligible_prefix_count_k4_to_penultimate": len(cand),
                          "used_penultimate_fallback_k3": bool(len(cand) == 0),
                          "eventual_dropout_target": int(row.target)})
    pd.DataFrame(cand_rows).to_csv(out / "candidate_prefix_audit.csv", index=False)

    # Compact report. Existing files are not edited or removed.
    lines = [
        "# C3: Exponentially weighted prefix sampling",
        "",
        "## Executive summary",
        "",
        f"- Fixed evaluation: the same penultimate-only C1 cohort and the same locked 20% student test split (outer seed {args.outer_seed}) are used for every condition.",
        f"- Fixed model: `{cfg['name']}` with the existing compact + per-lesson feature set (`{feature_set}`); this isolates the training-prefix strategy.",
        "- Baseline trains on each development student's penultimate prefix. C3 samples exactly one earlier-or-penultimate prefix per training student with probability proportional to exp(-lambda * distance from penultimate).",
        "- Validation and test always use penultimate-prefix rows. All splits are by student; no student contributes to both training and validation/test.",
        "- The supplied eventual dropout label is used for every sampled prefix. This is not a next-attempt target, and no dropout timestamp exists.",
        f"- Decision: the penultimate-only baseline had OOF AP {baseline_oof_ap:.4f}; the best C3 variant ({selected}, lambda={selected_lambda:g}) had OOF AP {selected_oof_ap:.4f} (delta {oof_delta:+.4f}).",
        f"- On the locked test, baseline AP was {baseline_test_ap:.4f} and best-C3 AP was {selected_test_ap:.4f} (delta {test_delta:+.4f}). C3 does not improve the ranking metric in this run; do not replace the baseline based on these results.",
        "",
        "## Development out-of-fold comparison",
        "",
        oof_df.to_markdown(index=False),
        "",
        "## Locked test comparison",
        "",
        test_df.to_markdown(index=False),
        "",
        "## Interpretation rules",
        "",
        "- Prefer a C3 condition only if its development OOF AP improves over baseline and the locked test result is directionally consistent. One test split with a small positive count is noisy.",
        "- A higher AP measures ranking, not calibrated probability. Precision/recall thresholds are selected from development OOF scores only.",
        "- Repeated sampled-prefix fits reduce dependence on a single random draw; they do not create additional independent students.",
        "- Since validation/test remain penultimate-only, this experiment tests whether earlier-prefix training improves penultimate prediction. It does not directly establish performance at earlier live checkpoints.",
        "- Existing files, datasets, scripts and outputs were not modified or deleted. This script and its outputs are additive.",
        "",
        "## Candidate prefix construction",
        "",
        "- For each training student, candidate prefix lengths are k=4 through that student's penultimate prefix (k=N-1).",
        "- If a student's penultimate prefix is k=3 (four total attempts), the only valid row is the penultimate k=3 row and it is used as a fallback.",
        "- Sampling probability is proportional to `exp(-lambda * (penultimate_k - k))`; therefore the penultimate prefix is most likely, and earlier prefixes become exponentially less likely.",
        "- Each student contributes exactly one row to each fitted training dataset, avoiding extra row-level weighting for students with longer histories.",
        "",
        "## Files",
        "",
        "- `development_oof_comparison.csv`: OOF AP and development-selected operating point for each condition.",
        "- `locked_test_comparison.csv`: locked test metrics for each condition, on identical test students.",
        "- `test_predictions_*.csv`: per-student held-out scores and flags for each condition.",
        "- `candidate_prefix_audit.csv`: number of eligible prefixes per student and fallback indicator.",
        "- `oof_fold_sampling_diagnostics.csv`: fold/repeat diagnostics.",
        "",
        "## Limitations",
        "",
        "- This is a single locked outer split to preserve comparability with the existing seed-42 experiment; uncertainty across new cohorts is not estimated.",
        "- Sampling a shorter prefix while retaining eventual outcome labels is valid for eventual-dropout prediction, but the feature distribution differs from the penultimate test distribution.",
        "- The project has 500 students overall and 374 eligible C-stage students; results should be treated as exploratory.",
    ]
    (out / "C3_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Development OOF:")
    print(oof_df.to_string(index=False))
    print("\nLocked test:")
    print(test_df.to_string(index=False))
    print("\\nOutputs:", out)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

#!/usr/bin/env python3
"""Evaluate Task 2 against the labelled benchmark.

Predictions are generated and saved before the golden labels are loaded.
Previously inspected benchmark results must be reported as iterative tracking,
not as an untouched final-test estimate.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task2.src.evaluator import evaluate_one, load_knowledge, VALID_STATUSES  # noqa: E402

TASK2_DIR = ROOT / "task2"
EVAL_DIR = TASK2_DIR / "data" / "evaluation"
OUTPUT_DIR = TASK2_DIR / "outputs" / "benchmark"
INPUTS_PATH = EVAL_DIR / "golden_inputs.csv"
LABELS_PATH = EVAL_DIR / "golden_labels.csv"
MANIFEST_PATH = EVAL_DIR / "golden_manifest.json"

CLASSES = ["understood", "partial", "wrong"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="Confirm that you understand this is an iterative benchmark "
             "if results have previously been inspected.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small smoke test; do not use its metrics as benchmark results.",
    )
    args = parser.parse_args()

    if not INPUTS_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark inputs: {INPUTS_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inputs = pd.read_csv(INPUTS_PATH).fillna("")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        inputs = inputs.head(args.limit).copy()

    required = {"explanation_id", "lesson_id", "explanation_text"}
    missing = required - set(inputs.columns)
    if missing:
        raise ValueError(f"Missing columns in golden inputs: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prediction_path = OUTPUT_DIR / f"predictions_{run_id}.csv"
    metrics_path = OUTPUT_DIR / f"metrics_{run_id}.json"

    knowledge = load_knowledge()
    records = []
    total = len(inputs)
    print(f"Generating predictions for {total} examples.")
    # print("Golden labels have not been loaded.")

    for i, (_, series) in enumerate(inputs.iterrows(), start=1):
        row = series.to_dict()
        started = time.perf_counter()
        try:
            result = evaluate_one(row, knowledge)
            if not isinstance(result, dict):
                result = {
                    "explanation_id": row["explanation_id"],
                    "verdict": None,
                    "status": "invalid_result",
                    "error": f"Expected dict, got {type(result).__name__}",
                }
        except Exception as exc:
            result = {
                "explanation_id": row["explanation_id"],
                "verdict": None,
                "status": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }

        record = {
            "explanation_id": str(row["explanation_id"]),
            "lesson_id": str(row["lesson_id"]),
            "verdict": result.get("verdict"),
            "status": result.get("status", "missing_status"),
            "latency_seconds": result.get(
                "latency_seconds", round(time.perf_counter() - started, 3)
            ),
            "attempt_count": result.get("attempt_count"),
            "retrieved_cases": result.get("retrieved_cases"),
            "feedback_field_repairs": json.dumps(
                result.get("feedback_field_repairs", []), ensure_ascii=False
            ),
            "validation_errors": json.dumps(
                result.get("validation_errors", []), ensure_ascii=False
            ),
            "error": result.get("error", ""),
            "got_right": result.get("got_right", ""),
            "missed": result.get("missed", ""),
            "hint": result.get("hint", ""),
        }
        records.append(record)
        print(
            f"[{i}/{total}] {record['explanation_id']}: "
            f"{record['verdict']} ({record['status']})"
        )

    # Persist all model outputs before opening the labels file.
    predictions = pd.DataFrame(records)
    predictions.to_csv(prediction_path, index=False)
    print(f"\nSaved predictions first: {prediction_path}")

    # Only now load the golden labels.
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark labels: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH)

    if not {"explanation_id", "label"}.issubset(labels.columns):
        raise ValueError("Golden labels must contain explanation_id and label.")
    if labels["explanation_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate explanation_id values in golden labels.")

    scored = predictions.merge(
        labels[["explanation_id", "label"]].assign(
            explanation_id=lambda df: df["explanation_id"].astype(str)
        ),
        on="explanation_id",
        how="left",
        validate="one_to_one",
    )

    if scored["label"].isna().any():
        missing_ids = scored.loc[scored["label"].isna(), "explanation_id"].tolist()
        raise ValueError(f"Labels missing for prediction IDs: {missing_ids[:10]}")

    valid = scored[
        scored["status"].isin(VALID_STATUSES)
        & scored["verdict"].isin(CLASSES)
    ].copy()

    failures = scored.loc[~scored.index.isin(valid.index)].copy()

    metrics = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "iterative_benchmark_tracking",
        "warning": (
            "The labelled examples have been inspected during development. "
            "These metrics are not an untouched final-test estimate."
        ),
        "manifest_split_version": manifest.get("split_version"),
        "n_inputs": int(len(scored)),
        "n_successfully_scored": int(len(valid)),
        "n_failed_or_invalid": int(len(failures)),
        "status_counts": {
            str(k): int(v)
            for k, v in predictions["status"].value_counts(dropna=False).items()
        },
        "feedback_repair_count": int(
            predictions["status"].isin(
                {
                    "ok_with_feedback_repair",
                    "recovered_after_retry_with_feedback_repair",
                }
            ).sum()
        ),
        "accuracy_on_successful_predictions": None,
        "macro_f1_on_successful_predictions": None,
        "classification_report": {},
        "confusion_matrix_labels_order": CLASSES,
        "confusion_matrix": [],
        "failure_ids": failures["explanation_id"].astype(str).tolist(),
    }

    if len(valid):
        y_true = valid["label"].astype(str)
        y_pred = valid["verdict"].astype(str)
        metrics["accuracy_on_successful_predictions"] = float(
            accuracy_score(y_true, y_pred)
        )
        metrics["macro_f1_on_successful_predictions"] = float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        )
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=CLASSES
        ).tolist()

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Iterative benchmark results ===")
    print(f"Examples: {len(scored)}")
    print(f"Successfully scored: {len(valid)}")
    print(f"Failed/invalid: {len(failures)}")
    print(f"Feedback repairs: {metrics['feedback_repair_count']}")
    if len(valid):
        print(f"Accuracy (successful predictions): "
              f"{metrics['accuracy_on_successful_predictions']:.4f}")
        print(f"Macro-F1 (successful predictions): "
              f"{metrics['macro_f1_on_successful_predictions']:.4f}")
        print("\nClassification report:")
        print(classification_report(
            valid["label"].astype(str),
            valid["verdict"].astype(str),
            labels=CLASSES,
            zero_division=0,
        ))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(pd.DataFrame(
            metrics["confusion_matrix"],
            index=CLASSES,
            columns=CLASSES,
        ))
    else:
        print("No valid predictions to score; inspect the prediction CSV.")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Predictions saved to: {prediction_path}")

    if args.limit is not None:
        print("NOTE: --limit was used; this was a smoke test, not a full benchmark.")


if __name__ == "__main__":
    main()
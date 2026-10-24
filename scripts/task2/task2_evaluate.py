#!/usr/bin/env python3
"""Run the four Task 2 benchmark conditions and save predictions/metrics.

Conditions: risk/golden x memory off/on. Labels are deliberately loaded only
AFTER the raw prediction CSV has been written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task2.src.evaluator import VALID_STATUSES, VERDICTS, evaluate_one, load_knowledge  # noqa: E402

DATASETS = {
    "risk": ROOT / "task2/data/evaluation/risk_dataset.csv",
    "golden": ROOT / "task2/data/evaluation/golden_dataset.csv",
}
INPUT_COLUMNS = ["explanation_id", "lesson_id", "explanation_text"]
LABEL_COLUMN = "label"


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def evaluate_condition(dataset_name: str, memory_enabled: bool, out_dir: Path, max_retries: int) -> dict[str, Any]:
    dataset_path = DATASETS[dataset_name]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Intentionally do not read the label column during inference.
    inputs = pd.read_csv(dataset_path, usecols=INPUT_COLUMNS, dtype=str, keep_default_na=False)
    missing = set(INPUT_COLUMNS) - set(inputs.columns)
    if missing:
        raise ValueError(f"{dataset_path} missing required input columns: {sorted(missing)}")

    knowledge = load_knowledge()
    run_name = f"{dataset_name}__memory_{'on' if memory_enabled else 'off'}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    total = len(inputs)

    # The current POC memory is explicitly marked approved_for_poc. Permit that
    # source only in memory-on runs; the evaluator still excludes golden cases.
    if memory_enabled:
        os.environ["TASK2_ALLOW_PROVISIONAL_MEMORY"] = "1"

    print(f"\n=== {run_name}: {total} examples ===")
    for index, row in inputs.iterrows():
        payload = {column: str(row[column]) for column in INPUT_COLUMNS}
        try:
            result = evaluate_one(payload, knowledge, max_retries=max_retries, use_memory=memory_enabled)
        except Exception as exc:  # Keep the run complete and preserve failures.
            result = {
                "explanation_id": payload["explanation_id"],
                "verdict": None,
                "got_right": "",
                "missed": "",
                "hint": "",
                "status": "runner_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "retrieved_cases": 0,
                "latency_seconds": None,
            }
        record = {
            **payload,
            "predicted_verdict": result.get("verdict"),
            "got_right": result.get("got_right", ""),
            "missed": result.get("missed", ""),
            "hint": result.get("hint", ""),
            "status": result.get("status", "unknown"),
            "attempt_count": result.get("attempt_count"),
            "latency_seconds": result.get("latency_seconds"),
            "retrieved_cases": result.get("retrieved_cases", 0),
            "validation_errors": json.dumps(result.get("validation_errors", []), ensure_ascii=False),
            "error": result.get("error", ""),
        }
        predictions.append(record)
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"[{index + 1}/{total}] {payload['explanation_id']}: {record['predicted_verdict'] or 'INVALID'} ({record['status']})")

    predictions_df = pd.DataFrame(predictions)
    predictions_path = run_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    # Durable checkpoint: predictions are persisted before the labels are read.
    (run_dir / "prediction_manifest.json").write_text(json.dumps({
        "dataset": dataset_name,
        "memory_enabled": memory_enabled,
        "input_path": str(dataset_path),
        "prediction_path": str(predictions_path),
        "prediction_rows": len(predictions_df),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_column_loaded_for_inference": False,
        "memory_policy": "retrieval explicitly bypassed" if not memory_enabled else "canonical development memory; POC provisional cases permitted",
    }, indent=2) + "\n", encoding="utf-8")

    # Labels are first accessed only after predictions have been saved.
    labels_df = pd.read_csv(dataset_path, usecols=["explanation_id", LABEL_COLUMN], dtype=str, keep_default_na=False)
    scoring = predictions_df[["explanation_id", "predicted_verdict", "status"]].merge(
        labels_df, on="explanation_id", how="left", validate="one_to_one"
    )
    if scoring[LABEL_COLUMN].eq("").any() or scoring[LABEL_COLUMN].isna().any():
        raise ValueError(f"Missing labels while scoring {dataset_name}")

    valid_label_mask = scoring[LABEL_COLUMN].isin(VERDICTS)
    valid_pred_mask = scoring["predicted_verdict"].isin(VERDICTS) & scoring["status"].isin(VALID_STATUSES)
    scored = scoring[valid_label_mask & valid_pred_mask].copy()
    labels_order = ["understood", "partial", "wrong"]
    if len(scored):
        y_true = scored[LABEL_COLUMN].tolist()
        y_pred = scored["predicted_verdict"].tolist()
        report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_order, average="weighted", zero_division=0)),
            "confusion_matrix_labels_order": labels_order,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_order).tolist(),
            "classification_report": report,
        }
    else:
        metrics = {"accuracy": None, "macro_f1": None, "weighted_f1": None, "confusion_matrix_labels_order": labels_order, "confusion_matrix": [[0] * 3 for _ in range(3)], "classification_report": {}, "note": "No valid predictions available for scoring."}

    metrics.update({
        "dataset": dataset_name,
        "memory_enabled": memory_enabled,
        "total_examples": int(len(scoring)),
        "valid_predictions_scored": int(len(scored)),
        "invalid_or_failed_predictions": int(len(scoring) - len(scored)),
        "invalid_prediction_rate": float(1 - len(scored) / len(scoring)) if len(scoring) else None,
        "mean_latency_seconds": float(pd.to_numeric(predictions_df["latency_seconds"], errors="coerce").mean()) if len(predictions_df) else None,
        "mean_retrieved_cases": float(pd.to_numeric(predictions_df["retrieved_cases"], errors="coerce").fillna(0).mean()) if len(predictions_df) else 0.0,
        "predictions_path": str(predictions_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_safe) + "\n", encoding="utf-8")
    scoring.to_csv(run_dir / "scored_predictions.csv", index=False)
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Valid scored: {len(scored)}/{len(scoring)}; accuracy={metrics['accuracy']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["risk", "golden", "all"], default="all")
    parser.add_argument("--memory", choices=["off", "on", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "task2/outputs/benchmark")
    parser.add_argument("--run-id", default=None, help="Optional run folder name; defaults to a UTC timestamp.")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    datasets = ["risk", "golden"] if args.dataset == "all" else [args.dataset]
    memory_modes = [False, True] if args.memory == "both" else [args.memory == "on"]
    summary = []
    for dataset_name in datasets:
        for memory_enabled in memory_modes:
            summary.append(evaluate_condition(dataset_name, memory_enabled, run_root, args.max_retries))

    summary_path = run_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_safe) + "\n", encoding="utf-8")
    print(f"\nAll requested experiments complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()

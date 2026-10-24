
#!/usr/bin/env python3
"""Evaluate unlabeled explanations from Task 1 flagged at-risk students."""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "task2"))

from src.evaluator import load_knowledge, evaluate_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-file",
        default=(
            "outputs/eventual_dropout_aligned/"
            "test_predictions_precision_50.csv"
        ),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--output",
        default="task2/outputs/at_risk_examples.csv",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    prediction_path = ROOT / args.prediction_file
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Task 1 prediction file not found: {prediction_path}"
        )

    predictions = pd.read_csv(prediction_path)

    flag_col = next(
        (
            col for col in [
                "cascade_flag",
                "flag",
                "prediction_at_primary_precision_floor",
            ]
            if col in predictions.columns
        ),
        None,
    )
    if flag_col is None:
        raise ValueError(
            "No supported risk flag column found. "
            f"Available columns: {predictions.columns.tolist()}"
        )

    if "student_id" not in predictions.columns:
        raise ValueError("Prediction file must contain student_id.")

    flagged = predictions[
        pd.to_numeric(predictions[flag_col], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    ]
    flagged_ids = set(flagged["student_id"].astype(str))

    explanations = pd.read_csv(ROOT / "data" / "explanations.csv")
    required = {
        "explanation_id", "student_id", "lesson_id",
        "explanation_text", "label",
    }
    missing = required - set(explanations.columns)
    if missing:
        raise ValueError(
            f"Explanations file missing columns: {sorted(missing)}"
        )

    # Only unlabeled rows are eligible for this operational demo.
    unlabeled = explanations[
        explanations["label"].isna()
        | explanations["label"].astype(str).str.strip().eq("")
    ].copy()

    candidates = unlabeled[
        unlabeled["student_id"].astype(str).isin(flagged_ids)
    ].copy()

    # Explicit deterministic sampling; does not claim these are latest attempts.
    candidates = candidates.sort_values(
        ["student_id", "explanation_id"], kind="stable"
    ).drop_duplicates("student_id", keep="first").head(args.limit)

    if candidates.empty:
        raise RuntimeError(
            "No flagged students with unlabeled explanations were found. "
            "Check the prediction file, flag column, and student IDs."
        )

    knowledge = load_knowledge()
    results = []

    for _, row in candidates.iterrows():
        result = evaluate_one(row.to_dict(), knowledge)
        result["student_id"] = row["student_id"]
        result["explanation_id"] = row["explanation_id"]
        result["lesson_id"] = row["lesson_id"]
        result["explanation_text"] = row["explanation_text"]
        results.append(result)

        print(
            f"{row['student_id']} / {row['explanation_id']}: "
            f"{result.get('verdict')} ({result.get('status')})"
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output_path, index=False)
    output_path.with_suffix(".json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()

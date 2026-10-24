
#!/usr/bin/env python3
"""Prepare all 60 supplied labelled explanations as the Task 2 golden set.

No labelled row is designated as development data or used as episodic memory.
Inputs and labels are written separately so labels are only joined after inference.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "explanations.csv"
OUT = ROOT / "task2" / "data" / "evaluation"


def main():
    df = pd.read_csv(SOURCE)

    required = {
        "explanation_id", "student_id", "lesson_id",
        "explanation_text", "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    labelled = df[df["label"].ne("")].copy()

    allowed = {"understood", "partial", "wrong"}
    if len(labelled) != 60 or set(labelled["label"]) != allowed:
        raise SystemExit(
            f"Expected 60 labelled rows with labels {allowed}; "
            f"got {len(labelled)} rows and {set(labelled['label'])}"
        )

    if labelled["explanation_id"].duplicated().any():
        raise ValueError("Duplicate explanation_id found in labelled rows.")

    OUT.mkdir(parents=True, exist_ok=True)

    input_cols = [
        "explanation_id", "student_id",
        "lesson_id", "explanation_text",
    ]
    inputs = labelled[input_cols].sort_values("explanation_id")
    labels = labelled[["explanation_id", "label"]].sort_values(
        "explanation_id"
    )

    input_path = OUT / "golden_inputs.csv"
    label_path = OUT / "golden_labels.csv"

    inputs.to_csv(input_path, index=False)
    labels.to_csv(label_path, index=False)

    manifest = {
        "split_version": "all-labelled-v1",
        "method": (
            "All 60 provided human-labelled rows form the golden benchmark; "
            "no development split is created."
        ),
        "total_labelled_rows": len(labelled),
        "development_rows": 0,
        "golden_rows": len(labelled),
        "golden_labels_are_separate_file": True,
        "class_counts": labels["label"].value_counts().sort_index().to_dict(),
        "golden_inputs_sha256": hashlib.sha256(
            input_path.read_bytes()
        ).hexdigest(),
        "golden_labels_sha256": hashlib.sha256(
            label_path.read_bytes()
        ).hexdigest(),
        "memory_policy": (
            "No row or label from this set may be used as episodic memory."
        ),
        "evaluation_policy": (
            "Repeated evaluation after inspecting results is iterative "
            "benchmark tracking, not an untouched test estimate."
        ),
    }

    manifest_path = OUT / "golden_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

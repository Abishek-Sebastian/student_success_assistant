
#!/usr/bin/env python3
"""Generate golden, risk, and non-risk Task 2 datasets."""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE_CSV = ROOT / "data/task2/explanations_labelled_full.csv"
PREDICTIONS_CSV = (
    ROOT / "outputs/final_cascade_evaluation/"
    "outer_test_student_predictions.csv"
)
OUTPUT_ROOT = ROOT / "task2/data"

STUDENT_ID_COL = "student_id"
LABEL_SOURCE_COL = "label_source"
EXPLANATION_ID_COL = "explanation_id"

GOLDEN_SOURCE = "provided_existing_label"
PROVISIONAL_SOURCE = "rubric_based_provisional_label"

# Hardcoded end-to-end risk definition.
RISK_FLAG_COL = "flag_end_to_end"
RISK_FLAG_VALUE = "1"

VALID_LESSONS = {"L1", "L2", "L3"}
VALID_LABELS = {"understood", "partial", "wrong"}


def normalize_id(value):
    """Normalize student IDs consistently across both files."""
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value)).strip()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    return value.casefold()


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {PREDICTIONS_CSV}"
        )

    source = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(
        PREDICTIONS_CSV, dtype=str, keep_default_na=False
    )

    # Validate required source columns.
    required_source = {
        STUDENT_ID_COL,
        LABEL_SOURCE_COL,
        "explanation_text",
        "lesson_id",
        "label",
    }
    missing = required_source - set(source.columns)

    if missing:
        raise ValueError(
            f"Source CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {source.columns.tolist()}"
        )

    # Validate prediction columns.
    required_predictions = {STUDENT_ID_COL, RISK_FLAG_COL}
    missing = required_predictions - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Prediction CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {predictions.columns.tolist()}"
        )

    if source[STUDENT_ID_COL].map(normalize_id).eq("").any():
        raise ValueError("Source CSV contains blank student IDs.")

    if predictions[STUDENT_ID_COL].map(normalize_id).eq("").any():
        raise ValueError("Prediction CSV contains blank student IDs.")

    # 1. Identify golden examples using label_source.
    label_source = (
        source[LABEL_SOURCE_COL].astype(str).str.strip().str.casefold()
    )

    golden_mask = label_source.eq(GOLDEN_SOURCE.casefold())
    provisional_mask = label_source.eq(PROVISIONAL_SOURCE.casefold())

    unknown_mask = ~(golden_mask | provisional_mask)

    if unknown_mask.any():
        unknown_values = sorted(label_source[unknown_mask].unique())
        raise ValueError(
            "Unexpected label_source values. No datasets written. "
            f"Values: {unknown_values}"
        )

    golden = source.loc[golden_mask].copy()
    provisional = source.loc[provisional_mask].copy()

    if golden.empty:
        raise ValueError(
            f"No golden rows found for label_source={GOLDEN_SOURCE!r}."
        )

    if provisional.empty:
        raise ValueError(
            f"No development rows found for "
            f"label_source={PROVISIONAL_SOURCE!r}."
        )

    # 2. Extract at-risk student IDs from flag_end_to_end == 1.
    # A student is at risk if ANY of their prediction rows has flag 1.
    flags = (
        predictions[RISK_FLAG_COL]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    risk_rows = predictions.loc[flags.eq(RISK_FLAG_VALUE)].copy()

    if risk_rows.empty:
        observed = sorted(predictions[RISK_FLAG_COL].unique().tolist())
        raise ValueError(
            f"No rows have {RISK_FLAG_COL}={RISK_FLAG_VALUE}. "
            f"Observed values: {observed}"
        )

    risk_student_ids = set(
        risk_rows[STUDENT_ID_COL].map(normalize_id)
    )

    # 3. Match prediction student IDs against the full labelled dataset.
    provisional["_student_key"] = provisional[STUDENT_ID_COL].map(
        normalize_id
    )

    source_student_ids = set(provisional["_student_key"])
    prediction_student_ids = set(
        predictions[STUDENT_ID_COL].map(normalize_id)
    )

    missing_prediction_ids = source_student_ids - prediction_student_ids

    if missing_prediction_ids:
        print(
            f"{len(missing_prediction_ids)} provisional students are missing "
            "from the prediction file. The current prediction file does not "
            "cover the full development population. Generate predictions "
            "for all provisional students before creating risk/non-risk "
            "datasets, or explicitly define how unmatched students should "
            "be handled."
        )

    
    # Identify at-risk students: any prediction row with flag_end_to_end == 1.
    flags = (
        predictions[RISK_FLAG_COL]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    risk_student_ids = set(
        predictions.loc[
            flags.eq("1"), STUDENT_ID_COL
        ].map(normalize_id)
    )

    # Normalize provisional rows and validate using the actual label values.
    provisional["_student_key"] = provisional[STUDENT_ID_COL].map(
        normalize_id
    )
    provisional["_normalized_text"] = provisional[
        "explanation_text"
    ].map(normalize_text)
    provisional["_normalized_lesson"] = (
        provisional["lesson_id"].astype(str).str.strip().str.upper()
    )
    provisional["_normalized_label"] = (
        provisional["label"].astype(str).str.strip().str.casefold()
    )

    valid = (
        provisional["_normalized_text"].ne("")
        & provisional["_normalized_lesson"].isin(VALID_LESSONS)
        & provisional["_normalized_label"].isin(
            {"understood", "partial", "wrong"}
        )
    )

    provisional = provisional.loc[valid].copy()

    # Students absent from predictions are deliberately treated as non-risk.
    is_risk = provisional["_student_key"].isin(risk_student_ids)

    risk = provisional.loc[is_risk].copy()
    nonrisk = provisional.loc[~is_risk].copy()


    # 4. Validate development examples.
    # Do not apply development filtering to the golden evaluation set.
    def validate_development(df):
        valid = (
            df["_normalized_text"].ne("")
            & df["_normalized_lesson"].isin(VALID_LESSONS)
            & df["_normalized_label"].isin(VALID_LABELS)
        )
        return df.loc[valid].copy()

    risk_before_validation = len(risk)
    nonrisk_before_validation = len(nonrisk)

    risk = validate_development(risk)
    nonrisk = validate_development(nonrisk)

    # 5. Deduplicate within each segment separately.
    # Deduplicating separately prevents one segment from removing an
    # example belonging to the other segment.
    risk_before_dedup = len(risk)
    nonrisk_before_dedup = len(nonrisk)

    dedup_cols = ["_normalized_lesson", "_normalized_text"]

    risk = risk.drop_duplicates(subset=dedup_cols, keep="first").copy()
    nonrisk = nonrisk.drop_duplicates(
        subset=dedup_cols, keep="first"
    ).copy()

    internal_cols = [
        "_student_key",
        "_normalized_text",
        "_normalized_lesson",
        "_normalized_label",
    ]

    risk.drop(columns=internal_cols, inplace=True, errors="ignore")
    nonrisk.drop(columns=internal_cols, inplace=True, errors="ignore")

    # 6. Safety checks.
    risk_output_ids = set(risk[STUDENT_ID_COL].map(normalize_id))
    nonrisk_output_ids = set(nonrisk[STUDENT_ID_COL].map(normalize_id))

    if risk_output_ids & nonrisk_output_ids:
        raise RuntimeError(
            "A student appears in both risk and non-risk datasets."
        )

    unexpected_nonrisk = nonrisk_output_ids & risk_student_ids
    if unexpected_nonrisk:
        raise RuntimeError(
            "At-risk students leaked into non-risk dataset: "
            f"{sorted(unexpected_nonrisk)[:10]}"
        )

    # Ensure golden examples are not mixed with development by ID.
    if EXPLANATION_ID_COL in source.columns:
        golden_ids = set(
            golden[EXPLANATION_ID_COL].astype(str).str.strip()
        ) - {""}
        provisional_ids = set(
            provisional[EXPLANATION_ID_COL].astype(str).str.strip()
        ) - {""}

        overlap = golden_ids & provisional_ids
        if overlap:
            raise RuntimeError(
                f"{len(overlap)} explanation IDs occur in both golden "
                "and provisional data."
            )

    # 7. Write the three requested outputs.
    evaluation_dir = OUTPUT_ROOT / "evaluation"
    development_dir = OUTPUT_ROOT / "development"

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    development_dir.mkdir(parents=True, exist_ok=True)

    golden_path = evaluation_dir / "golden_dataset.csv"
    risk_path = evaluation_dir / "risk_dataset.csv"
    nonrisk_path = development_dir / "nonrisk_dataset.csv"

    golden.to_csv(golden_path, index=False)
    risk.to_csv(risk_path, index=False)
    nonrisk.to_csv(nonrisk_path, index=False)

    # 8. Write an audit report.
    audit = {
        "source_file": str(SOURCE_CSV),
        "predictions_file": str(PREDICTIONS_CSV),
        "risk_rule": f"{RISK_FLAG_COL} == {RISK_FLAG_VALUE}",
        "risk_definition": (
            "A student is at risk if any prediction row has "
            "flag_end_to_end == 1."
        ),
        "source_rows": int(len(source)),
        "source_unique_students": int(
            source[STUDENT_ID_COL].map(normalize_id).nunique()
        ),
        "golden_rows": int(len(golden)),
        "provisional_rows_before_split": int(len(provisional)),
        "prediction_rows": int(len(predictions)),
        "prediction_unique_students": int(
            predictions[STUDENT_ID_COL].map(normalize_id).nunique()
        ),
        "prediction_rows_flagged_at_risk": int(len(risk_rows)),
        "unique_at_risk_students": int(len(risk_student_ids)),
        "at_risk_students_matched_to_source": int(
            len(
                risk_student_ids
                & set(source[STUDENT_ID_COL].map(normalize_id))
            )
        ),
        "risk_rows_before_validation": int(risk_before_validation),
        "risk_rows_after_validation_and_dedup": int(len(risk)),
        "risk_duplicates_removed": int(
            risk_before_dedup - len(risk)
        ),
        "nonrisk_rows_before_validation": int(nonrisk_before_validation),
        "nonrisk_rows_after_validation_and_dedup": int(len(nonrisk)),
        "nonrisk_duplicates_removed": int(
            nonrisk_before_dedup - len(nonrisk)
        ),
        "risk_output_unique_students": int(len(risk_output_ids)),
        "nonrisk_output_unique_students": int(len(nonrisk_output_ids)),
        "at_risk_rows_in_nonrisk_output": int(
            nonrisk[STUDENT_ID_COL].map(normalize_id)
            .isin(risk_student_ids).sum()
        ),
        "golden_used_for_memory": False,
        "outputs": {
            "golden": str(golden_path),
            "risk": str(risk_path),
            "nonrisk": str(nonrisk_path),
        },
    }

    audit_path = OUTPUT_ROOT / "dataset_generation_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2))
    print(f"\nAudit saved to: {audit_path}")


if __name__ == "__main__":
    main()

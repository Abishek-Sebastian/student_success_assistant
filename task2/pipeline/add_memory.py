
"""Select informative Task 2 memory candidates without mutating canonical memory.

The pipeline:
- evaluates the risk-filtered candidate pool with the existing evaluator;
- compares predictions with candidate labels, treating disagreements as provisional;
- ranks for disagreement, novelty, and lesson coverage;
- writes proposals to task2/outputs/memory_pipeline/;
- never writes to approved_cases.jsonl or rebuilds Chroma.

Candidate labels must not be described as ground truth unless independently verified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task2.src.evaluator import evaluate_one, load_knowledge  # noqa: E402


CANDIDATES_PATH = (
    ROOT / "task2/memory/candidates/memory_candidates_nonrisk_only.csv"
)
CANONICAL_MEMORY_PATH = ROOT / "task2/memory/approved_cases.jsonl"
OUTPUT_DIR = ROOT / "task2/outputs/memory_pipeline"

VALID_LABELS = {"understood", "partial", "wrong"}
VALID_STATUSES = {
    "ok",
    "recovered_after_retry",
    "ok_with_feedback_repair",
    "recovered_after_retry_with_feedback_repair",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def load_canonical_memory() -> list[dict[str, Any]]:
    """Read existing canonical memory for duplicate/novelty checks only."""
    if not CANONICAL_MEMORY_PATH.exists():
        return []

    records = []
    for line_number, line in enumerate(
        CANONICAL_MEMORY_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid canonical memory JSONL at line {line_number}"
            ) from exc

        # Refuse to proceed if the canonical file contains golden data.
        if (
            record.get("source_dataset") == "provided_60_labelled_rows"
            or record.get("split") == "golden"
        ):
            raise ValueError(
                "Golden-set record found in canonical memory. "
                "Resolve this before running the pipeline."
            )
        records.append(record)

    return records


def max_similarity_to_texts(text: str, other_texts: list[str]) -> float:
    """Return TF-IDF cosine similarity to the nearest non-empty text."""
    text = str(text).strip()
    other_texts = [str(t).strip() for t in other_texts if str(t).strip()]
    if not text or not other_texts:
        return 0.0

    corpus = other_texts + [text]
    try:
        matrix = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True,
        ).fit_transform(corpus)
    except ValueError:
        return 0.0

    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    return float(scores.max()) if len(scores) else 0.0


def validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "explanation_id",
        "student_id",
        "lesson_id",
        "explanation_text",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Candidate CSV is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    for column in ["explanation_id", "student_id", "lesson_id",
                   "explanation_text", "label"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["label"] = df["label"].str.lower()
    df = df[
        df["explanation_text"].ne("")
        & df["lesson_id"].isin(["L1", "L2", "L3"])
        & df["label"].isin(VALID_LABELS)
    ].copy()

    if "label_status" in df.columns:
        # These files are expected to contain provisional non-golden labels.
        statuses = df["label_status"].fillna("").astype(str).str.lower()
        if statuses.str.contains("golden").any():
            raise ValueError("Golden-labelled rows are not eligible for memory.")
        df = df[~statuses.str.contains("golden")].copy()

    # Remove exact duplicate explanations within the same lesson.
    df["_normalized_text"] = df["explanation_text"].map(normalize_text)
    df = df.drop_duplicates(
        subset=["lesson_id", "_normalized_text"], keep="first"
    ).copy()

    if df.empty:
        raise ValueError("No valid candidates remain after validation.")

    return df.reset_index(drop=True)


def evaluate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    knowledge = load_knowledge()
    results: list[dict[str, Any]] = []

    total = len(df)
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        payload = {
            "explanation_id": str(row["explanation_id"]),
            "lesson_id": str(row["lesson_id"]),
            "explanation_text": str(row["explanation_text"]),
        }

        try:
            result = evaluate_one(payload, knowledge)
            if not isinstance(result, dict):
                raise TypeError(
                    f"evaluate_one returned {type(result).__name__}, not dict"
                )
        except Exception as exc:
            result = {
                "verdict": None,
                "status": "exception",
                "error": f"{type(exc).__name__}: {exc}",
            }

        predicted = str(result.get("verdict") or "").strip().lower()
        status = str(result.get("status") or "missing_status")
        valid_prediction = (
            predicted in VALID_LABELS and status in VALID_STATUSES
        )
        candidate_label = str(row["label"]).strip().lower()

        record = {
            "explanation_id": str(row["explanation_id"]),
            "student_id": str(row["student_id"]),
            "lesson_id": str(row["lesson_id"]),
            "explanation_text": str(row["explanation_text"]),
            "candidate_label": candidate_label,
            "label_status": str(row.get("label_status", "provisional")),
            "label_source": str(
                row.get("label_source", "unknown_candidate_label_source")
            ),
            "predicted_verdict": predicted or None,
            "evaluator_status": status,
            "valid_prediction": valid_prediction,
            "potential_disagreement": (
                bool(valid_prediction and predicted != candidate_label)
            ),
            "got_right": str(result.get("got_right", "")),
            "missed": str(result.get("missed", "")),
            "hint": str(result.get("hint", "")),
            "latency_seconds": result.get("latency_seconds"),
            "retrieved_cases": result.get("retrieved_cases"),
            "error": str(result.get("error", "")),
        }
        results.append(record)
        print(
            f"[{position}/{total}] {record['explanation_id']}: "
            f"pred={record['predicted_verdict']} "
            f"candidate={candidate_label} status={status}"
        )

    return pd.DataFrame(results)


def rank_candidates(
    evaluated: pd.DataFrame,
    canonical_records: list[dict[str, Any]],
    max_selected: int,
    max_per_lesson: int,
    duplicate_threshold: float,
) -> pd.DataFrame:
    if evaluated.empty:
        return evaluated.copy()

    approved_texts = [
        str(r.get("explanation_text", ""))
        for r in canonical_records
        if str(r.get("explanation_text", "")).strip()
    ]

    ranked = evaluated.copy()
    ranked["similarity_to_existing_memory"] = ranked[
        "explanation_text"
    ].map(lambda t: max_similarity_to_texts(t, approved_texts))
    ranked["novelty_score"] = (
        1.0 - ranked["similarity_to_existing_memory"]
    ).clip(0.0, 1.0)

    # Disagreements are useful review targets, but may be caused by noisy labels.
    # No model confidence is inferred because the evaluator does not return one.
    ranked["selection_score"] = (
        ranked["potential_disagreement"].astype(float) * 2.0
        + ranked["novelty_score"] * 1.0
    )
    ranked = ranked.sort_values(
        ["selection_score", "novelty_score", "explanation_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selected_indices: list[int] = []
    selected_texts_by_lesson: dict[str, list[str]] = {}
    selected_counts: dict[str, int] = {}
    all_seen_texts_by_lesson: dict[str, set[str]] = {}

    for idx, row in ranked.iterrows():
        lesson = str(row["lesson_id"])
        normalized = normalize_text(row["explanation_text"])

        # Never select an exact duplicate of canonical memory for the same lesson.
        existing_for_lesson = {
            normalize_text(r.get("explanation_text", ""))
            for r in canonical_records
            if str(r.get("lesson_id", "")) == lesson
        }
        if normalized in existing_for_lesson:
            continue

        if selected_counts.get(lesson, 0) >= max_per_lesson:
            continue

        # Greedy diversity check against already selected cases in this lesson.
        previous = selected_texts_by_lesson.get(lesson, [])
        if previous and max_similarity_to_texts(normalized, previous) >= duplicate_threshold:
            continue

        selected_indices.append(idx)
        selected_texts_by_lesson.setdefault(lesson, []).append(
            str(row["explanation_text"])
        )
        selected_counts[lesson] = selected_counts.get(lesson, 0) + 1

        if len(selected_indices) >= max_selected:
            break

    ranked["selected_for_proposal"] = ranked.index.isin(selected_indices)
    ranked["selection_reason"] = ranked.apply(
        lambda row: (
            "potential_disagreement_and_novelty"
            if row["selected_for_proposal"] and row["potential_disagreement"]
            else "novelty_and_lesson_coverage"
            if row["selected_for_proposal"]
            else "not_selected"
        ),
        axis=1,
    )
    return ranked


def write_proposals(selected: pd.DataFrame, output_path: Path) -> int:
    """Write a proposal-only JSONL. These records are not approved memory."""
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            if not bool(row["selected_for_proposal"]):
                continue

            case_id = f"proposal_{row['explanation_id']}"
            record = {
                "case_id": case_id,
                "explanation_id": str(row["explanation_id"]),
                "student_id": str(row["student_id"]),
                "lesson_id": str(row["lesson_id"]),
                "explanation_text": str(row["explanation_text"]),
                "final_verdict": str(row["candidate_label"]),
                "predicted_verdict": str(row["predicted_verdict"] or ""),
                "approval_status": "proposed_not_approved",
                "label_source": str(row["label_source"]),
                "label_status": str(row["label_status"]),
                "split": "development",
                "source_dataset": "non_golden_risk_filtered_candidate_pool",
                "memory_version": 1,
                "selection_score": round(float(row["selection_score"]), 4),
                "potential_disagreement": bool(row["potential_disagreement"]),
                "similarity_to_existing_memory": round(
                    float(row["similarity_to_existing_memory"]), 4
                ),
                "proposed_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": (
                    "Proposal only. Candidate label is not independently verified. "
                    "Do not append to approved_cases.jsonl."
                ),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and select Task 2 memory candidates safely."
    )
    parser.add_argument("--input", type=Path, default=CANDIDATES_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N candidates for a smoke test.")
    parser.add_argument("--max-selected", type=int, default=30)
    parser.add_argument("--max-per-lesson", type=int, default=12)
    parser.add_argument("--duplicate-threshold", type=float, default=0.85)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_selected < 1 or args.max_per_lesson < 1:
        parser.error("--max-selected and --max-per-lesson must be positive")
    if not 0.0 <= args.duplicate_threshold <= 1.0:
        parser.error("--duplicate-threshold must be between 0 and 1")
    if not args.input.exists():
        raise FileNotFoundError(
            f"Candidate CSV not found: {args.input}\n"
            "Check that memory_candidates_nonrisk_only.csv is in "
            "task2/memory/candidates/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    candidates = validate_candidates(raw)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    canonical = load_canonical_memory()

    print(f"Candidates to evaluate: {len(candidates)}")
    print("Calling the configured evaluator API. This may take some time.")
    evaluated = evaluate_candidates(candidates)

    ranked = rank_candidates(
        evaluated=evaluated,
        canonical_records=canonical,
        max_selected=args.max_selected,
        max_per_lesson=args.max_per_lesson,
        duplicate_threshold=args.duplicate_threshold,
    )

    predictions_path = OUTPUT_DIR / "candidate_evaluations.csv"
    ranked.to_csv(predictions_path, index=False)

    proposals_path = OUTPUT_DIR / "proposed_memory_cases.jsonl"
    selected_count = write_proposals(ranked, proposals_path)

    valid = ranked[ranked["valid_prediction"]]
    disagreements = ranked[ranked["potential_disagreement"]]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "candidate_rows_after_validation": int(len(candidates)),
        "evaluator_valid_predictions": int(len(valid)),
        "evaluator_failures_or_invalid": int(len(ranked) - len(valid)),
        "potential_disagreements": int(len(disagreements)),
        "proposed_cases": int(selected_count),
        "lesson_counts_input": {
            str(k): int(v)
            for k, v in candidates["lesson_id"].value_counts().to_dict().items()
        },
        "lesson_counts_proposed": {
            str(k): int(v)
            for k, v in ranked.loc[
                ranked["selected_for_proposal"], "lesson_id"
            ].value_counts().to_dict().items()
        },
        "candidate_label_counts": {
            str(k): int(v)
            for k, v in candidates["label"].value_counts().to_dict().items()
        },
        "predicted_verdict_counts": {
            str(k): int(v)
            for k, v in ranked["predicted_verdict"].fillna(
                "invalid_or_failed"
            ).value_counts().to_dict().items()
        },
        "canonical_memory_cases_read_only": int(len(canonical)),
        "golden_set_added_to_memory": False,
        "canonical_memory_modified": False,
        "chroma_rebuilt": False,
        "label_warning": (
            "Candidate labels are provisional. Disagreements are selection signals, "
            "not confirmed evaluator errors. Proposed cases are not approved memory."
        ),
        "outputs": {
            "predictions_csv": str(predictions_path),
            "proposals_jsonl": str(proposals_path),
        },
    }

    report_path = OUTPUT_DIR / "memory_pipeline_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Memory candidate pipeline complete ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nNo canonical memory records were added and Chroma was not rebuilt.")


if __name__ == "__main__":
    main()

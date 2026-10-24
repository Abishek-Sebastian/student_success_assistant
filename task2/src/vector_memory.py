
"""Persistent local semantic memory for approved Task 2 cases."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = ROOT / "task2/memory/approved_cases.jsonl"
CHROMA_PATH = ROOT / "task2/memory/chroma_db"
COLLECTION_NAME = "approved_task2_cases"

EMBEDDING_MODEL = os.getenv(
    "TASK2_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
ALLOWED_LABEL_SOURCES = {
    "manual_review",
    "larger_llm_then_human_verified",
    "poc_assumed_provisional",
    "independently_verified",
    "human_verified",
    "provided_existing_label",
}
ALLOWED_APPROVAL_STATUSES = {"approved", "approved_for_poc"}
TRUSTED_APPROVAL_STATUSES = {"approved"}
VALID_VERDICTS = {"understood", "partial", "wrong"}


def _load_approved_records() -> list[dict[str, Any]]:
    """Load approved development cases, including explicitly POC-approved records."""
    if not MEMORY_PATH.exists():
        return []

    records = []
    seen_ids = set()
    seen_texts = set()

    for line_number, line in enumerate(
        MEMORY_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at line {line_number}"
            ) from exc

        if (
            record.get("approval_status") not in ALLOWED_APPROVAL_STATUSES
            or record.get("label_source") not in ALLOWED_LABEL_SOURCES
        ):
            raise ValueError(
                f"Record at line {line_number} is not independently approved"
            )

        if (
            record.get("source_dataset") == "provided_60_labelled_rows"
            or record.get("split") == "golden"
        ):
            raise ValueError("Protected golden-set record found in memory")

        if record.get("split") != "development":
            raise ValueError(
                f"Record at line {line_number} must belong to development"
            )

        case_id = str(record.get("case_id", "")).strip()
        lesson_id = str(record.get("lesson_id", "")).strip()
        explanation = str(record.get("explanation_text", "")).strip()
        verdict = str(record.get("correct_verdict", record.get("final_verdict", record.get("verified_verdict", "")))).strip().lower()

        if not case_id or not lesson_id or not explanation:
            raise ValueError(
                f"Record at line {line_number} lacks a required identifier, "
                "lesson_id, or explanation_text"
            )

        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"Record {case_id} has invalid final_verdict: {verdict!r}"
            )

        normalized = " ".join(explanation.lower().split())
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        if (lesson_id, normalized) in seen_texts:
            raise ValueError(f"Duplicate explanation in lesson: {case_id}")

        seen_ids.add(case_id)
        seen_texts.add((lesson_id, normalized))
        records.append(record)

    return records


def _dependencies():
    """Import optional dependencies only when vector memory is used."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install chromadb and sentence-transformers to use vector memory."
        ) from exc

    return chromadb, SentenceTransformer


def rebuild_index() -> int:
    """Rebuild Chroma from the canonical approved JSONL file."""
    chromadb, SentenceTransformer = _dependencies()
    records = _load_approved_records()

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Rebuilding replaces only this project's collection.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if not records:
        print("Approved cases: 0. Created an empty Chroma collection.")
        return 0

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [r["explanation_text"].strip() for r in records]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    collection.add(
        ids=[str(r["case_id"]) for r in records],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "lesson_id": str(r["lesson_id"]),
                "final_verdict": str(r.get("correct_verdict", r.get("final_verdict", r.get("verified_verdict", "")))).lower(),
                "review_status": str(r["approval_status"]),
                "label_status": str(r.get("label_status", "unknown")),
                "ground_truth_available": str(bool(r.get("ground_truth_available", False))).lower(),
                "source_split": str(r["split"]),
                "source_dataset": str(r.get("source_dataset", "independent")),
                "memory_version": str(r.get("memory_version", "1")),
                "label_source": str(r["label_source"]),
                "case_type": str(r.get("case_type", "learning_case")),
                "predicted_verdict": str(r.get("predicted_verdict", "")),
                "error_type": str(r.get("error_type", "")),
                "correction_principle": str(r.get("correction_principle", "")),
                "error_description": str(r.get("error_description", "")),
                "rubric_concepts_present": json.dumps(r.get("rubric_concepts_present", []), ensure_ascii=False),
                "rubric_concepts_missing": json.dumps(r.get("rubric_concepts_missing", []), ensure_ascii=False),
                "retrieval_tags": json.dumps(r.get("retrieval_tags", []), ensure_ascii=False),
            }
            for r in records
        ],
    )

    print(f"Indexed {collection.count()} approved cases.")
    return collection.count()


def retrieve_vector_cases(
    lesson_id: str,
    explanation: str,
    k: int = 2,
    exclude_explanation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve similar approved cases from the same lesson."""
    if k <= 0 or not explanation.strip() or not CHROMA_PATH.exists():
        return []

    chromadb, SentenceTransformer = _dependencies()
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(
        [explanation],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Oversample because we may exclude the current case or exact duplicates.
    # POC memory built from accepted candidate labels is retrievable by default.
    statuses = ["approved", "approved_for_poc"]
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(collection.count(), max(k * 4, k)),
        where={
            "$and": [
                {"lesson_id": str(lesson_id)},
                {"review_status": {"$in": statuses}},
                {"source_split": "development"},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )

    cases = []
    query_norm = " ".join(explanation.lower().split())

    for case_id, document, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if case_id == exclude_explanation_id:
            continue
        if " ".join(document.lower().split()) == query_norm:
            continue

        cases.append({
            "case_id": case_id,
            "lesson_id": metadata["lesson_id"],
            "explanation_text": document,
            "final_verdict": metadata["final_verdict"],
            "correct_verdict": metadata["final_verdict"],
            "approval_status": metadata["review_status"],
            "label_status": metadata.get("label_status", "unknown"),
            "ground_truth_available": metadata.get("ground_truth_available", "false") == "true",
            "label_source": metadata["label_source"],
            "case_type": metadata.get("case_type", "learning_case"),
            "predicted_verdict": metadata.get("predicted_verdict", ""),
            "error_type": metadata.get("error_type", ""),
            "correction_principle": metadata.get("correction_principle", ""),
            "error_description": metadata.get("error_description", ""),
            "rubric_concepts_present": json.loads(metadata.get("rubric_concepts_present", "[]")),
            "rubric_concepts_missing": json.loads(metadata.get("rubric_concepts_missing", "[]")),
            "retrieval_tags": json.loads(metadata.get("retrieval_tags", "[]")),
            "similarity": round(1.0 - float(distance), 4),
        })

        if len(cases) >= k:
            break

    return cases

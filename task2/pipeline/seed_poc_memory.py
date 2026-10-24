#!/usr/bin/env python3
"""Seed the Task 2 proof-of-concept CBR memory from the three reviewed proposals.

Assumption: provisional labels are treated as correct for this assignment POC.
This script refuses golden-set sources, is idempotent by case_id, and does not
silently overwrite existing canonical records. Run from repository root.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROPOSALS = ROOT / "task2/outputs/memory_pipeline/proposed_memory_cases.jsonl"
EVALUATIONS = ROOT / "task2/outputs/memory_pipeline/candidate_evaluations.csv"
MEMORY = ROOT / "task2/memory/approved_cases.jsonl"
VALID = {"understood", "partial", "wrong"}

CONCEPTS = {
    "E0005": {"present": ["sunlight", "water", "sugar"], "missing": ["carbon dioxide", "oxygen"], "tags": ["photosynthesis", "missing_concepts", "overgrading"]},
    "E0010": {"present": ["chloroplasts", "light energy", "carbon dioxide", "water", "glucose", "oxygen"], "missing": ["chlorophyll"], "tags": ["photosynthesis", "complete_process", "positive_example"]},
    "E0013": {"present": ["supply", "demand", "price", "equilibrium", "quantity supplied", "quantity demanded"], "missing": [], "tags": ["supply_demand", "price_changes", "equilibrium", "positive_example"]},
}

def norm(s): return re.sub(r"\s+", " ", str(s).lower()).strip()

def main():
    if not PROPOSALS.exists() or not EVALUATIONS.exists():
        raise SystemExit("Missing pipeline outputs. Run add_memory.py --limit 5 --max-selected 3 first.")
    proposals = [json.loads(x) for x in PROPOSALS.read_text(encoding="utf-8").splitlines() if x.strip()]
    eval_rows = {}
    import pandas as pd
    df = pd.read_csv(EVALUATIONS).fillna("")
    for row in df.to_dict(orient="records"):
        eval_rows[str(row.get("explanation_id", ""))] = row
    existing = []
    if MEMORY.exists():
        for line_no, line in enumerate(MEMORY.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            record = json.loads(line)
            if record.get("split") == "golden" or record.get("source_dataset") == "provided_60_labelled_rows":
                raise SystemExit(f"Protected golden-set record detected in canonical memory at line {line_no}.")
            existing.append(record)
    ids = {str(r.get("case_id")) for r in existing}
    known_texts = {(str(r.get("lesson_id")), norm(r.get("explanation_text", ""))) for r in existing}
    additions = []
    for p in proposals:
        eid = str(p.get("explanation_id", ""))
        row = eval_rows.get(eid)
        if not row or str(row.get("valid_prediction", "")).lower() not in {"true", "1"}:
            raise SystemExit(f"Refusing to seed {eid}: no valid evaluator prediction.")
        if p.get("source_dataset") == "provided_60_labelled_rows" or p.get("split") == "golden":
            raise SystemExit(f"Refusing golden-set proposal {eid}.")
        correct = str(p.get("final_verdict", p.get("candidate_label", ""))).lower()
        predicted = str(p.get("predicted_verdict", row.get("predicted_verdict", ""))).lower()
        if correct not in VALID or predicted not in VALID:
            raise SystemExit(f"Invalid verdict for {eid}: {correct}/{predicted}")
        case_id = f"poc_{eid}"
        if case_id in ids:
            print(f"Already present; skipping {case_id}")
            continue
        key = (str(p["lesson_id"]), norm(p["explanation_text"]))
        if key in known_texts:
            print(f"Duplicate explanation; skipping {eid}")
            continue
        is_error = predicted != correct
        concept = CONCEPTS.get(eid, {"present": [], "missing": [], "tags": []})
        error_type = ""
        error_description = ""
        correction = ""
        if is_error:
            # Verdict ordinal: wrong < partial < understood
            rank = {"wrong": 0, "partial": 1, "understood": 2}
            error_type = "overgrading" if rank[predicted] > rank[correct] else "undergrading"
            error_description = str(row.get("missed", ""))
            correction = "Apply the lesson rubric to required concepts; do not infer mastery from partial evidence."
        record = {
            "schema_version": 1,
            "case_id": case_id,
            "case_type": "evaluator_error" if is_error else "learning_case",
            "lesson_id": str(p["lesson_id"]),
            "explanation_id": eid,
            "student_id": str(p.get("student_id", row.get("student_id", ""))),
            "explanation_text": str(p["explanation_text"]),
            "predicted_verdict": predicted,
            "correct_verdict": correct,
            "final_verdict": correct,
            "label_source": "poc_assumed_provisional",
            "label_status": "assumed_for_poc",
            "error_type": error_type,
            "error_description": error_description,
            "correction_principle": correction,
            "rubric_concepts_present": concept["present"],
            "rubric_concepts_missing": concept["missing"],
            "retrieval_tags": concept["tags"] + ([error_type] if error_type else []),
            "split": "development",
            "approval_status": "approved_for_poc",
            "memory_version": 1,
            "source_dataset": "non_golden_risk_filtered_candidate_pool",
            "similarity_metric": {"semantic_weight": 0.65, "concept_weight": 0.20, "lexical_weight": 0.15},
            "assumption": "Provisional candidate label treated as correct for assignment proof of concept; not independently verified.",
        }
        additions.append(record)
        ids.add(case_id); known_texts.add(key)
    if additions:
        MEMORY.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY.open("a", encoding="utf-8") as f:
            for r in additions: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"added": len(additions), "canonical_memory_count": len(existing)+len(additions), "case_ids": [r["case_id"] for r in additions], "golden_set_used": False}, indent=2))
    if additions:
        sys.path.insert(0, str(ROOT / "task2"))
        from src.vector_memory import rebuild_index
        count = rebuild_index()
        print(f"Chroma index rebuilt with {count} approved POC/verified cases.")

if __name__ == "__main__": main()

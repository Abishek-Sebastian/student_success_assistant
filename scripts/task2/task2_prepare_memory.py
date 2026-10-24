#!/usr/bin/env python3
"""Build versioned episodic CBR memory from human-labelled development cases only.

Never imports golden inputs or labels. Run after preparing the fixed split; regenerate
only when the development memory is deliberately versioned and re-tested.
"""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "task2/data/development/development_cases.csv"
MEMORY_DIR = ROOT / "task2/memory"
OUT = MEMORY_DIR / "approved_cases.jsonl"
MANIFEST = MEMORY_DIR / "case_memory_manifest.json"


def main():
    if not DEV.exists():
        raise SystemExit(f"Missing development cases: {DEV}")
    df = pd.read_csv(DEV).fillna("")
    allowed = {"understood", "partial", "wrong"}
    if not set(df.label.astype(str).str.lower()).issubset(allowed):
        raise SystemExit("Development memory contains an invalid label")
    required = {"explanation_id", "lesson_id", "explanation_text", "label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Development data missing columns: {sorted(required-set(df.columns))}")
    records = []
    for row in df.to_dict(orient="records"):
        records.append({
            "case_id": str(row["explanation_id"]),
            "lesson_id": str(row["lesson_id"]),
            "explanation_text": str(row["explanation_text"]),
            "verified_verdict": str(row["label"]).strip().lower(),
            "label_source": "provided_human_label",
            "approval_status": "approved",
            "split": "development",
            "memory_version": 1,
        })
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    OUT.write_text(payload, encoding="utf-8")
    manifest = {
        "memory_version": 1,
        "source": "task2/data/development/development_cases.csv",
        "source_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
        "case_count": len(records),
        "label_source": "provided_human_label",
        "allowed_splits": ["development"],
        "golden_data_used": False,
        "model_generated_cases_allowed": False,
        "retention_policy": "Only human-labelled development cases; no automatic model self-retention.",
        "memory_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

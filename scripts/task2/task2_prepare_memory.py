
#!/usr/bin/env python3
"""Validate the approved episodic memory without importing dataset examples."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = ROOT / "task2" / "memory"
OUT = MEMORY_DIR / "approved_cases.jsonl"
MANIFEST = MEMORY_DIR / "case_memory_manifest.json"

ALLOWED_LABEL_SOURCES = {
    "manual_review",
    "larger_llm_then_human_verified",
}


def main():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if not OUT.exists():
        OUT.write_text("", encoding="utf-8")

    records = []

    for line_number, line in enumerate(
        OUT.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON on memory line {line_number}: {exc}"
            ) from exc

        if (
            record.get("source_dataset") == "provided_60_labelled_rows"
            or record.get("split") == "golden"
        ):
            raise SystemExit(
                "Refusing memory containing a row from the 60-row golden set."
            )

        if record.get("approval_status") != "approved":
            raise SystemExit(
                f"Memory line {line_number} is not explicitly approved."
            )

        if record.get("label_source") not in ALLOWED_LABEL_SOURCES:
            raise SystemExit(
                f"Memory line {line_number} lacks an acceptable independent "
                "verification source."
            )

        required = {
            "case_id", "lesson_id", "explanation",
            "verdict", "approval_status", "label_source",
        }
        missing = required - set(record)
        if missing:
            raise SystemExit(
                f"Memory line {line_number} missing fields: {sorted(missing)}"
            )

        if record["verdict"] not in {"wrong", "partial", "understood"}:
            raise SystemExit(
                f"Invalid verdict on memory line {line_number}."
            )

        records.append(record)

    manifest = {
        "memory_version": 2,
        "source": (
            "Independently verified cases only; never import the supplied "
            "60 labelled rows."
        ),
        "case_count": len(records),
        "label_sources_allowed": sorted(ALLOWED_LABEL_SOURCES),
        "golden_data_used": False,
        "provided_labelled_rows_allowed": False,
        "model_generated_cases_allowed_without_human_verification": False,
        "retention_policy": (
            "No automatic retention. Each case must be independently "
            "verified and manually approved."
        ),
        "memory_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

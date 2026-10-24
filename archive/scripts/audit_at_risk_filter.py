#!/usr/bin/env python3
"""Audit whether candidate-pool exclusions match the at-risk student list.

Run from repo root, for example:
  python scripts/task2/audit_at_risk_filter.py \
    --source-csv path/to/all_non_golden_conversations.csv \
    --candidates-csv task2/memory/candidates/memory_candidates_nonrisk_only.csv \
    --risk-csv path/to/at_risk_students.csv \
    --risk-id-col student_id

The risk CSV must contain student IDs. If it contains all students plus a risk
flag, also pass --risk-flag-col and --risk-flag-value (e.g. --risk-flag-col risk_label
--risk-flag-value at_risk). Script is read-only and writes audit outputs only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd

ID_COL = "student_id"


def norm_ids(series: pd.Series) -> pd.Series:
    # Preserve IDs as strings (read CSV with dtype=str) and trim whitespace.
    return series.fillna("").astype(str).str.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-csv", required=True, help="All non-golden conversations before at-risk filtering")
    ap.add_argument("--candidates-csv", default="task2/memory/candidates/memory_candidates_nonrisk_only.csv")
    ap.add_argument("--risk-csv", required=True, help="CSV containing at-risk student IDs or all students with a risk flag")
    ap.add_argument("--source-id-col", default=ID_COL)
    ap.add_argument("--candidate-id-col", default=ID_COL)
    ap.add_argument("--risk-id-col", default=ID_COL)
    ap.add_argument("--risk-flag-col", default=None, help="Optional flag column in risk CSV")
    ap.add_argument("--risk-flag-value", default="at_risk", help="Value indicating at-risk when --risk-flag-col is used")
    ap.add_argument("--output-dir", default="task2/outputs/at_risk_filter_audit")
    args = ap.parse_args()

    paths = [Path(args.source_csv), Path(args.candidates_csv), Path(args.risk_csv)]
    for p in paths:
        if not p.exists():
            ap.error(f"File not found: {p}")

    source = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    candidates = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    risk = pd.read_csv(paths[2], dtype=str, keep_default_na=False)

    for df, col, label in [(source,args.source_id_col,"source"),(candidates,args.candidate_id_col,"candidates"),(risk,args.risk_id_col,"risk")]:
        if col not in df.columns:
            ap.error(f"Column '{col}' not found in {label} CSV. Available columns: {list(df.columns)}")

    source["_sid"] = norm_ids(source[args.source_id_col])
    candidates["_sid"] = norm_ids(candidates[args.candidate_id_col])
    risk["_sid"] = norm_ids(risk[args.risk_id_col])

    if args.risk_flag_col:
        if args.risk_flag_col not in risk.columns:
            ap.error(f"Risk flag column '{args.risk_flag_col}' not found. Available columns: {list(risk.columns)}")
        flag = risk[args.risk_flag_col].fillna("").astype(str).str.strip().str.lower()
        risk_rows = risk.loc[flag == args.risk_flag_value.strip().lower()].copy()
    else:
        risk_rows = risk.copy()

    source = source.loc[source["_sid"].ne("")].copy()
    candidates = candidates.loc[candidates["_sid"].ne("")].copy()
    risk_rows = risk_rows.loc[risk_rows["_sid"].ne("")].copy()

    source_ids = set(source["_sid"])
    candidate_ids = set(candidates["_sid"])
    risk_ids = set(risk_rows["_sid"])
    source_unique = source_ids
    removed_rows = source.loc[~source["_sid"].isin(candidate_ids)].copy()
    removed_ids = set(removed_rows["_sid"])
    retained_rows = source.loc[source["_sid"].isin(candidate_ids)].copy()

    # At-risk student IDs represented in source, and how they appear in candidates.
    risk_in_source = risk_ids & source_unique
    risk_retained = risk_in_source & candidate_ids
    risk_removed = risk_in_source - candidate_ids
    nonrisk_removed = removed_ids - risk_ids
    risk_not_in_source = risk_ids - source_unique

    # Row-level table: every source conversation gets retained/removed and risk status.
    source_audit = source.copy()
    source_audit["in_candidate_pool"] = source_audit["_sid"].isin(candidate_ids)
    source_audit["student_marked_at_risk"] = source_audit["_sid"].isin(risk_ids)
    source_audit["audit_result"] = source_audit.apply(
        lambda r: "AT_RISK_RETAINED" if r["student_marked_at_risk"] and r["in_candidate_pool"]
        else "AT_RISK_REMOVED" if r["student_marked_at_risk"] and not r["in_candidate_pool"]
        else "NONRISK_RETAINED" if r["in_candidate_pool"]
        else "NONRISK_REMOVED", axis=1
    )
    source_audit.drop(columns=["_sid"], errors="ignore").to_csv("/tmp/unused.csv", index=False) if False else None

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    source_audit.drop(columns=["_sid"], errors="ignore").to_csv(outdir / "source_row_audit.csv", index=False)
    removed_rows.drop(columns=["_sid"], errors="ignore").to_csv(outdir / "removed_source_conversations.csv", index=False)
    retained_rows.drop(columns=["_sid"], errors="ignore").to_csv(outdir / "retained_source_conversations.csv", index=False)
    pd.DataFrame({"student_id": sorted(risk_removed)}).to_csv(outdir / "at_risk_students_removed.csv", index=False)
    pd.DataFrame({"student_id": sorted(risk_retained)}).to_csv(outdir / "at_risk_students_retained_unexpected.csv", index=False)
    pd.DataFrame({"student_id": sorted(nonrisk_removed)}).to_csv(outdir / "nonrisk_students_removed_unexpected.csv", index=False)
    pd.DataFrame({"student_id": sorted(risk_not_in_source)}).to_csv(outdir / "risk_students_not_in_source.csv", index=False)

    report = {
        "source_file": str(paths[0]), "candidate_file": str(paths[1]), "risk_file": str(paths[2]),
        "source_conversation_rows": int(len(source)), "candidate_conversation_rows": int(len(candidates)),
        "source_unique_students": int(len(source_unique)), "candidate_unique_students": int(len(candidate_ids)),
        "source_students_missing_from_candidate_pool": int(len(source_unique - candidate_ids)),
        "source_conversation_rows_retained_by_student_membership": int(source["_sid"].isin(candidate_ids).sum()),
        "source_conversation_rows_removed_by_student_membership": int((~source["_sid"].isin(candidate_ids)).sum()),
        "risk_file_rows_after_flag_filter": int(len(risk_rows)), "unique_at_risk_students_in_risk_file": int(len(risk_ids)),
        "at_risk_students_also_in_source": int(len(risk_in_source)),
        "at_risk_students_retained_in_candidates_unexpected": int(len(risk_retained)),
        "at_risk_students_removed_from_candidates": int(len(risk_removed)),
        "nonrisk_students_removed_from_candidates_unexpected": int(len(nonrisk_removed)),
        "risk_students_not_found_in_source": int(len(risk_not_in_source)),
        "note": "Membership is student-level: if a student has any row in the candidate file, all their source rows are counted as retained for the student-membership comparison. Inspect source_row_audit.csv for row-level status. Candidate rows may be a deduplicated subset, so conversation counts can differ from student membership counts."
    }
    (outdir / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nDetailed audit files saved to: {outdir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Back up, verify, and reset the Task 2 Chroma database."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = ROOT / "task2" / "memory" / "chroma_db"
BACKUP_ROOT = ROOT / "task2" / "outputs" / "chroma_backups"


def inventory(path: Path) -> tuple[int, int]:
    """Return file count and total bytes."""
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back up and reset the Task 2 Chroma index."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying files.",
    )
    args = parser.parse_args()

    print(f"Repository: {ROOT}")
    print(f"Chroma:    {CHROMA_PATH}")
    print(f"Backups:   {BACKUP_ROOT}")

    if not CHROMA_PATH.exists():
        print("Chroma directory does not exist; nothing to reset.")
        return 0

    if not CHROMA_PATH.is_dir():
        print("ERROR: Chroma path exists but is not a directory.")
        return 1

    if args.dry_run:
        print("\nDRY RUN: no changes made.")
        print("Would back up and verify the Chroma directory,")
        print("then delete the original directory.")
        return 0

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_ROOT / f"chroma_db_backup_{timestamp}"

    print(f"\nCreating backup: {backup_path}")

    try:
        shutil.copytree(CHROMA_PATH, backup_path)

        source_inventory = inventory(CHROMA_PATH)
        backup_inventory = inventory(backup_path)

        if source_inventory != backup_inventory:
            raise RuntimeError(
                f"Backup verification failed: source={source_inventory}, "
                f"backup={backup_inventory}"
            )

        print(
            f"Backup verified: {backup_inventory[0]} files, "
            f"{backup_inventory[1]} bytes."
        )

    except Exception as exc:
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)

        print(
            f"ERROR: backup failed. Original database preserved. {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        shutil.rmtree(CHROMA_PATH)
    except Exception as exc:
        print(
            f"ERROR: backup is safe, but original deletion failed: {exc}",
            file=sys.stderr,
        )
        print(f"Backup location: {backup_path}")
        return 3

    print("\nOld Chroma database deleted successfully.")
    print(f"Backup retained at: {backup_path}")
    print("The memory builder can now create a fresh index.")
    print("Canonical approved_cases.jsonl was not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

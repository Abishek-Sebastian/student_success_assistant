
#!/usr/bin/env bash
set -Eeuo pipefail

# Reset explicitly listed Task 2 generated artifacts.
# Always run with --dry-run first.
# Usage:
#   bash scripts/task2/reset_task2_generated.sh --dry-run
#   bash scripts/task2/reset_task2_generated.sh

DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

# IMPORTANT:
# Verify these paths against your refactored repository before use.
# Only list paths confirmed to contain regenerable artifacts.
CANDIDATES=(
    "task2/data/evaluation/golden_inputs.csv"
    "task2/data/evaluation/golden_labels.csv"
    "task2/data/split_manifest.json"
    # "task2/data/development/development_cases.csv"
    "task2/memory/approved_cases.jsonl"
    "task2/memory/case_memory_manifest.json"
    "task2/outputs/development"
    "task2/outputs/golden"
    "outputs/task2"
    "outputs/task_2"
    "artifacts/task2"
)

for rel in "${CANDIDATES[@]}"; do
    [[ -e "$rel" || -L "$rel" ]] || continue

    # Refuse paths that escape the repository.
    target="$(python3 - "$ROOT" "$rel" <<'PY'
import os, sys
root = os.path.realpath(sys.argv[1])
target = os.path.abspath(os.path.join(root, sys.argv[2]))
if os.path.commonpath([root, os.path.realpath(target)]) != root:
    raise SystemExit("Refusing path outside repository: " + target)
print(target)
PY
)"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "WOULD REMOVE: ${target#"$ROOT"/}"
    else
        echo "Removing: ${target#"$ROOT"/}"
        rm -rf -- "$target"
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete. Nothing was deleted."
else
    echo "Reset complete."
fi

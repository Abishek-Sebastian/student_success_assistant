# Task 2 JSON reliability + CBR memory patch

This patch is intended to be copied into the existing repository root, preserving `.env` and unrelated local work.

## Files included

- `task2/src/evaluator.py`: robust JSON parsing, application-owned IDs, bounded schema retries, explicit feedback repair flags, approved-memory-only retrieval.
- `scripts/task2/task2_evaluate.py`: development default, golden-evaluation confirmation gate, input/label hash checks, predictions written before golden labels are read, failure/repair metrics.
- `scripts/task2/task2_prepare_memory.py`: builds versioned episodic memory from development human-labelled rows only.
- `task2/memory/approved_cases.jsonl` and `case_memory_manifest.json`: current approved development case memory and provenance hashes.
- `docs/TASK2_MEMORY_POLICY.md`, `docs/TASK2_GUIDE.md`: documented CBR policy and golden-set procedure.
- `tests/test_task2.py`: JSON robustness, ID override, memory isolation, retrieval, and split tests.

## Apply

From the repository root, back up your current files, extract this ZIP, and copy the contents into the same relative paths. Do not replace your `.env` file.

Run:

```bash
python -m pytest -q tests/test_task2.py
python scripts/task2/task2_evaluate.py --split development
```

If the development split is deliberately rebuilt, regenerate the approved episodic memory before evaluating:

```bash
python scripts/task2/task2_prepare_memory.py
```

Do not run the golden set until model, prompt, retrieval, and repair policy are frozen. The final command requires explicit confirmation:

```bash
python scripts/task2/task2_evaluate.py --split golden --confirm-final-evaluation
```

The patch does not run the LLM or consume the golden set. It also does not alter `.env` or `.env.example`.

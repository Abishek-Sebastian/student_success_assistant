# Task 2 memory policy

This is a small, auditable CBR system with explicit memory boundaries.

- **Semantic memory:** `knowledge/lesson_rubric.json` contains lesson summaries, key points, and common misconceptions.
- **Episodic memory:** `approved_cases.jsonl` contains only provided human-labelled examples from the development split. Rebuild with `python scripts/task2/task2_prepare_memory.py`; review the generated `case_memory_manifest.json` and hashes.
- **Procedural memory:** the prompt, retrieval, validation, and retry/repair rules in `src/evaluator.py`.
- **Working memory:** one inference request's lesson rubric, student explanation, and retrieved examples; it is not persisted.

## CBR cycle

1. **Retrieve:** filter approved cases to the same lesson and rank with TF-IDF similarity. Exclude the current ID and exact normalized-text matches.
2. **Reuse:** provide up to two human-labelled cases as contextual examples; the model must still assess the current explanation against the lesson rubric.
3. **Revise:** validate JSON/schema, retry only schema failures, and deterministically fill blank feedback fields with visible repair markers. These fallbacks do not change the verdict.
4. **Retain:** no automatic retention. A future case can be added only from a human-verified label, after explicit review and memory versioning. Model predictions, golden labels, and golden inputs are never automatically added.

The golden set is isolated from memory. Development evaluation uses leave-one-out retrieval and excludes exact duplicate text. Golden evaluation can retrieve from this approved development memory, but never from the golden set itself. If prompts or memory are changed after examining golden results, the golden set is considered consumed and a new held-out set is needed for a fresh final claim.

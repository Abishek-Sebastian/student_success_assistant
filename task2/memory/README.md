# Task 2 memory policy

The 60 provided labelled explanations are the golden evaluation dataset. **No row, label, duplicate, or derived case from those 60 examples may be stored in episodic memory or included in a grading prompt.**

- **Semantic memory:** `../knowledge/lesson_rubric.json` contains lesson summaries, key points, and common misconceptions. This is the default grading context.
- **Episodic memory:** `approved_cases.jsonl` is empty by default. It may contain only examples outside the 60-row dataset that have been independently verified by manual review or by a larger LLM followed by human verification. Each case requires explicit approval and provenance.
- **Procedural memory:** the prompt, retrieval, validation, and retry/repair rules in `../src/evaluator.py`.
- **Working memory:** the current request's rubric and student explanation; it is not persisted.

## Grading and improvement cycle

1. Run zero-shot grading using the lesson rubric and no episodic examples.
2. Compare predictions with the supplied labels only after predictions have been saved.
3. Inspect disagreements. Use this chat or a stronger model to propose an explanation of the disagreement, but treat that as a hypothesis, not ground truth.
4. Human-verify any new example intended for memory. It must come from outside the 60-row golden dataset; never retain golden examples or labels.
5. Version prompt, model, settings, and external verified memory; rerun all 60 examples to track changes.

The 60 rows can support iterative comparison across versions, but once results are inspected they are no longer an untouched test set. Report the results as benchmark agreement / iterative evaluation, not as an unbiased final held-out estimate. A fresh independently labelled set would be needed for that stronger claim.

## Retrieval

If external verified memory is later populated, retrieval filters by lesson and ranks examples with TF-IDF word/bi-gram vectors and cosine similarity. The default empty memory means no retrieval and a zero-shot baseline. No automatic model self-retention is allowed.

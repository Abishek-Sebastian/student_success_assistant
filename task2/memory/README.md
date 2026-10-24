# Task 2 memory policy

This is a deliberately small, auditable case-based reasoning (CBR) memory.

- **Semantic memory:** `knowledge/lesson_rubric.json` contains the lesson summaries, key points, and common misconceptions.
- **Episodic memory:** only the 18 labelled development examples in `data/development/development_cases.csv` are eligible for retrieval.
- **Procedural memory:** the prompt and validation/retry logic in `src/evaluator.py` define how cases are graded.
- **Working memory:** a single request's lesson rubric, student explanation, and retrieved development examples; it is not persisted.

Golden evaluation inputs and labels are isolated under `data/evaluation/`. Neither golden inputs nor golden labels may be added to retrieval memory. The model must never write to memory. Only human-provided labels are used as verified labels. Any later additions require a new versioned dataset and a fresh held-out evaluation set.

# Task 2: Student explanation grader

This module grades explanations against `knowledge/lesson_rubric.json` and returns structured JSON with `explanation_id`, `verdict`, `got_right`, `missed`, and `hint`.

## Golden dataset and leakage boundary

The 60 labelled rows supplied in `data/explanations.csv` are **all** the golden evaluation dataset. `scripts/task2/task2_prepare_split.py` writes all 60 inputs to `data/evaluation/golden_inputs.csv` and their labels separately to `golden_labels.csv`. The evaluator does not send labels to the model and joins them only after all predictions are saved.

None of the 60 rows, labels, duplicates, or derived cases may be used as episodic memory. The default `memory/approved_cases.jsonl` is empty, so grading is zero-shot with respect to examples and uses only the lesson rubric. Independently verified examples from outside the 60 rows may be added to memory in a later version, with provenance and explicit approval.

We can rerun the model on these 60 rows after changing prompts or settings to track improvement. However, once results are inspected, this is iterative benchmark tracking, not an untouched final test. Report agreement with the provided labels, confusion matrix, per-class precision/recall/F1, coverage, latency, and feedback quality. Do not claim that repeated tuning on this set is leakage-free held-out evaluation. A new independently labelled set is required for an untouched final estimate.

The supplied labels are evaluation references, not guaranteed infallible truth. Disagreements can be inspected, and this chat or a stronger LLM can propose reasons, but suspected label errors should remain flagged until independently human-verified. Do not silently replace original labels.

## Memory policy

- **Semantic:** lesson rubric, key points, and common misconceptions in `knowledge/lesson_rubric.json`.
- **Episodic:** empty by default. Only examples outside the 60-row golden set, independently manually verified or reviewed by a larger LLM and then human-verified, may be added.
- **Procedural:** prompt, TF-IDF/cosine retrieval code (used only if approved external memory exists), schema validation, retry and feedback-repair policy.
- **Working:** current request only; not persisted.

No model-generated case is automatically retained. See `memory/README.md` and `memory/case_memory_manifest.json`.

## Configure and run

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in your environment. Keep secrets out of Git. For example:

```bash
export LLM_BASE_URL="http://127.0.0.1:8888/v1"
export LLM_MODEL="openresearchtools/Qwen3.5-4B-Instruct-GGUF"
# Set LLM_API_KEY as required by your local server.
python scripts/task2/task2_prepare_split.py
python scripts/task2/task2_prepare_memory.py
python scripts/task2/task2_evaluate.py --split golden --confirm-final-evaluation
```

The evaluator writes predictions before reading labels for scoring. Use a fresh `--output-dir` for a new run, or explicitly pass `--overwrite-final` to replace existing artifacts. `--limit` is for smoke tests only and is rejected for full benchmark runs.

## At-risk examples

`scripts/task2/task2_run_at_risk.py` uses Task 1 predictions and evaluates eligible unlabelled explanations for the five-example demo. It excludes the 60 labelled rows.

## Limitations

Only 60 supplied labels are available. Accuracy measures agreement with those labels, not necessarily objective grading correctness. JSON validation improves output structure, not conceptual correctness. Any model/prompt changes after viewing the 60-row results should be described as iterative improvement on this benchmark.

# Task 2: Evidence-grounded LLM evaluation

This is an independent module inside the existing Student Success Assistant repository. It does not change Task 1 model code or outputs.

## What is implemented

- Structured semantic memory for all three lessons (`knowledge/lesson_rubric.json`).
- An auditable CBR retrieval step that retrieves up to two same-lesson examples from a versioned, explicitly approved episodic memory built only from provided human-labelled development examples using TF-IDF similarity.
- Prompt protections treating student text as untrusted data, including attempts to override the grading instruction.
- Robust JSON parsing/schema validation (including markdown fences, preambles, Qwen thinking blocks, balanced-brace extraction, and API content arrays), bounded schema-only retries, explicit feedback-repair accounting, and per-example latency tracking. The application attaches explanation IDs rather than asking the model to generate them.
- A held-out evaluation path that generates all predictions first and only then joins the separate human labels for scoring.
- Metrics: accuracy, per-class precision/recall/F1, confusion matrix, valid output rate, latency, and row-level errors.

## Golden-set protocol (v1)

The provided dataset has 60 labelled rows, balanced across `wrong`, `partial`, and `understood`. The deterministic split reserves 18 development rows (6 per class) and 42 golden rows (14 per class). Exact normalized duplicate explanations are kept within a single split, so duplicate text cannot cross from development into the golden set. The golden inputs and labels are separate files. The evaluator never sends labels to the model.

**Do not open or use `data/evaluation/golden_labels.csv` while developing prompts, retrieval, or model settings.** The final evaluator joins it only after writing predictions. Because this is a small dataset, results have high uncertainty and should be accompanied by the per-class report and error analysis. If the prompt/model is tuned after looking at golden errors, v1 is no longer an untouched final set; use a new held-out golden set for the next final claim.

## API configuration

The evaluator calls an OpenAI-compatible `/v1/chat/completions` endpoint. Configure environment variables for your Unsloth server:

```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="local-unsloth"
export LLM_MODEL="unsloth/Qwen3.5-4B-Instruct"
```

These are defaults, not a claim that the chosen checkpoint is installed or that the API server is already running. Set `LLM_MODEL` to the exact model identifier exposed by your running Unsloth endpoint. Keep API keys out of source control.

## Run

From repository root, install the project requirements (scikit-learn is already included):

```bash
pip install -r requirements.txt
python scripts/task2/task2_prepare_memory.py  # run after preparing/refreshing development split
python scripts/task2/task2_evaluate.py --split development
# Only after freezing the model, prompt, memory, and retry/repair policy:
python scripts/task2/task2_evaluate.py --split golden --confirm-final-evaluation
python scripts/task2_demo.py --lesson-id L2 --explanation "An object keeps moving unless an unbalanced force acts. This tendency is inertia."
streamlit run task2/app.py
python scripts/task2_run_at_risk.py --prediction-file outputs/eventual_dropout_aligned/test_predictions_precision_50.csv --limit 5
```

The golden command is the official evaluation run. Do not use `--limit` for final metrics; it is only for smoke tests. Results are written under `task2/outputs/{development,golden}/`. The at-risk demo writes five eligible unlabeled explanation outputs to `task2/outputs/at_risk_examples.csv` when the configured API is running.

## Memory / CBR policy

- Semantic memory is the lesson rubric.
- Episodic memory is `memory/approved_cases.jsonl`, generated from the labelled development set by `scripts/task2/task2_prepare_memory.py`; the manifest records the source and SHA-256 hashes.
- Procedural memory is the evaluator prompt, retrieval, validation, and retry policy.
- Working memory is the current request only and is not persisted.
- The model never updates memory or self-approves examples. `approval_status=approved`, `label_source=provided_human_label`, and `split=development` are required for retrieval.
- Golden inputs/labels must never be retrieved as examples. Retrieval code reads only the approved episodic memory file.
- The evaluator defaults to development. Golden evaluation requires `--confirm-final-evaluation`, forbids `--limit`, verifies the golden-input hash before inference, writes predictions before reading golden labels, then verifies the label hash before scoring. Existing golden prediction artifacts are not overwritten unless explicitly requested.

## JSON reliability and limitations

- The model returns four fields (`verdict`, `got_right`, `missed`, `hint`); the application attaches `explanation_id` to avoid ID-generation mismatches.
- The parser accepts a single JSON object even when surrounded by a markdown fence or short preamble, removes Qwen `<think>` blocks, and uses balanced-brace extraction rather than a greedy regex. It still rejects invalid verdicts and missing required keys.
- Blank feedback strings are repaired deterministically and reported in `feedback_field_repairs`; these fallbacks count as repaired feedback, not as fully model-authored feedback. Inspect repair rate and quality before final evaluation.
- Schema retries are bounded. A generated verdict of `wrong` on API failure is never silently counted as a valid prediction: failure remains `failed_after_retries`, and metrics report coverage separately. Empty inputs use an explicit fallback status and are not evidence of model understanding.
- These checks improve format reliability, not factual correctness. The golden evaluation is still required to estimate held-out classification performance.


## Task 1 integration

`scripts/task2_run_at_risk.py` reads an existing Task 1 prediction file, selects flagged students with an unlabeled explanation, and evaluates up to five examples. It defaults to the Task 1 eventual-dropout-aligned 50% precision-floor predictions. It deliberately excludes all labelled explanations, so the demo cannot accidentally showcase golden evaluation examples. The supplied data has no reliable explanation timestamp, so the script selects one available row per student and does not claim it is the latest.

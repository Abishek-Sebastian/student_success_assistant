# Task 2: Evidence-grounded LLM evaluation

This is an independent module inside the existing Student Success Assistant repository. It does not change Task 1 model code or outputs.

## What is implemented

- Structured semantic memory for all three lessons (`knowledge/lesson_rubric.json`).
- An auditable CBR retrieval step that retrieves up to two same-lesson **development-only** labelled examples using TF-IDF similarity.
- Prompt protections treating student text as untrusted data, including attempts to override the grading instruction.
- Strict JSON parsing/schema validation, bounded retries, explicit failure status, and per-example latency tracking.
- A held-out evaluation path that generates all predictions first and only then joins the separate human labels for scoring.
- Metrics: accuracy, per-class precision/recall/F1, confusion matrix, valid output rate, latency, and row-level errors.

## Golden-set protocol (v1)

The provided dataset has 60 labelled rows, balanced across `wrong`, `partial`, and `understood`. The deterministic split reserves 18 development rows (6 per class) and 42 golden rows (14 per class). Exact normalized duplicate explanations are kept within a single split, so duplicate text cannot cross from development into the golden set. The golden inputs and labels are separate files. The evaluator never sends labels to the model.

**Do not open or use `data/evaluation/golden_labels.csv` while developing prompts, retrieval, or model settings.** The final evaluator joins it only after writing predictions. Because this is a small dataset, results have high uncertainty and should be accompanied by the per-class report and error analysis. If the prompt/model is tuned after looking at golden errors, v1 is no longer an untouched final set; use a new held-out golden set for the next final claim.

## API configuration

Task 2 loads environment variables from the repository-root `.env` file automatically using `python-dotenv`. Create it from the template:

```bash
cp .env.example .env
```

Edit `.env` and set the values to match your running Unsloth server. Example for the local server used during development:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8888/v1
LLM_API_KEY=replace-with-your-local-api-key
LLM_MODEL=replace-with-exact-model-id-from-unsloth
```

`LLM_MODEL` must exactly match the `id` returned by `GET /v1/models`; do not assume the model name or quantization suffix. Never commit `.env` or share its API key. `.env.example` is the safe-to-commit template.

To export `.env` variables into the current Bash/Zsh shell as well, run from the repository root:

```bash
set -a
source .env
set +a
```

Verify the server with:

```bash
curl -s "$LLM_BASE_URL/models" -H "Authorization: Bearer $LLM_API_KEY"
```

## Run

From repository root, install the project requirements (scikit-learn is already included):

```bash
pip install -r requirements.txt
python scripts/task2_evaluate.py --split development
python scripts/task2_evaluate.py --split golden
python scripts/task2_demo.py --lesson-id L2 --explanation "An object keeps moving unless an unbalanced force acts. This tendency is inertia."
streamlit run task2/app.py
python scripts/task2_run_at_risk.py --prediction-file outputs/eventual_dropout_aligned/test_predictions_precision_50.csv --limit 5
```

The golden command is the official evaluation run. Do not use `--limit` for final metrics; it is only for smoke tests. Results are written under `task2/outputs/{development,golden}/`. The at-risk demo writes five eligible unlabeled explanation outputs to `task2/outputs/at_risk_examples.csv` when the configured API is running.

## Memory / CBR policy

- Semantic memory is the lesson rubric.
- Episodic memory is restricted to the labelled development set.
- Procedural memory is the evaluator prompt, retrieval, validation, and retry policy.
- Working memory is the current request only and is not persisted.
- The model never updates memory or self-approves examples.
- Golden inputs/labels must never be retrieved as examples.

## Important limitation

A generated verdict of `wrong` on API failure is not silently counted as a valid model prediction: its `status` remains `failed_after_retries`, and metrics report the invalid/failure rate separately. For a production system, failures should be surfaced for retry or a safe non-grading fallback, not presented as confidently graded wrong. Empty inputs use an explicit fallback status and are not evidence of model understanding.


## Task 1 integration

`scripts/task2_run_at_risk.py` reads an existing Task 1 prediction file, selects flagged students with an unlabeled explanation, and evaluates up to five examples. It defaults to the Task 1 eventual-dropout-aligned 50% precision-floor predictions. It deliberately excludes all labelled explanations, so the demo cannot accidentally showcase golden evaluation examples. The supplied data has no reliable explanation timestamp, so the script selects one available row per student and does not claim it is the latest.

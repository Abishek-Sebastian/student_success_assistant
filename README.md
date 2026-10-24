# Student Success Assistant

A small student-risk and learning-feedback project built for the Student Success Assistant task. The repository contains **Task 1 (dropout-risk modelling)** and **Task 2 (LLM-based explanation feedback)**, plus supporting experiments and a research memo.

> **Scope note:** This cleanup reorganizes files and documentation. It does not intentionally change model logic, feature definitions, prompts, evaluation rules, or existing result artifacts.

## Project status

- **Task 1:** EDA, feature engineering, snapshot construction, baseline and benchmark experiments, and cascade evaluation are present. The current experiment outputs are retained under `outputs/`.
- **Task 2:** The LLM evaluator, Streamlit demo, lesson rubric, golden evaluation inputs/labels, evaluation outputs, and memory safeguards are present. This task is still in progress.
- **Task 3:** The research memo is part of the submission requirements; see `docs/` for research and implementation notes. Confirm the memo is complete before final submission.

## Repository layout

```text
.
├── README.md                    # Single project README
├── data/                        # Source data and Task 1 snapshot datasets
├── notebooks/                   # Task 1 EDA and feature-discovery notebooks
├── scripts/
│   ├── task1/                   # Task 1 dataset/model/benchmark scripts
│   └── task2/                   # Task 2 preparation, evaluation, and demo scripts
├── task2/
│   ├── app.py                   # Streamlit explanation-feedback demo
│   ├── src/                     # LLM evaluator
│   ├── knowledge/               # Lesson rubric
│   ├── memory/                  # Approved episodic memory and manifest
│   ├── data/                    # Task 2 development and golden evaluation data
│   └── outputs/                 # Task 2 predictions and metrics
├── outputs/                     # Task 1 experiment results and reports
├── docs/                        # Supporting guides, reports, and experiment notes
├── archive/                     # Historical experiments, reports, and artifacts
├── tests/                       # Tests
└── requirements.txt
```

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run notebook cells from the repository root or follow each notebook's data-path instructions. Some notebooks are exploratory and may rely on outputs produced by earlier steps.

## Task 1: dropout-risk modelling

### Main workflow

1. Explore and audit the source data in `data/` using the notebooks in `notebooks/`.
2. Build the snapshot datasets and their QA artifacts:

   ```bash
   python scripts/task1/01_build_cascade_snapshots.py
   ```

3. Run a selected experiment from the repository root. Examples:

   ```bash
   python scripts/task1/02_run_baseline_models.py
   python scripts/task1/07_run_kernel_svm_benchmark.py --outer-seeds 17 42 73 --inner-folds 2 --precision-floors 0.4 --top-k 3 --feature-sets compact compact_plus_module
   python scripts/task1/08_run_poly3_kernel_ensemble_benchmark.py --outer-seeds 17 42 73 --inner-folds 2 --precision-floor 0.4
   python scripts/task1/09_run_final_cascade_evaluation.py
   ```

4. Inspect the corresponding experiment folder under `outputs/`. Do not compare results from different target definitions or evaluation protocols as though they were the same experiment.

The scripts use repository-root-relative default paths. Run them from the repository root unless you explicitly provide the relevant `--input-dir`, `--output-dir`, or other path arguments.

### Current interpretation

The checked-in reports document multiple exploratory comparisons, not a guarantee of production performance. The kernel experiment notes report that RBF and degree-2 polynomial kernels were near-tied candidates for Stage C, without establishing a reliable end-to-end improvement. The degree-3 polynomial ensemble experiment is also exploratory. Review the relevant reports and evaluation artifacts before presenting a single model as the final recommendation.

## Task 2: LLM explanation feedback

Task 2 grades a student's explanation against the lesson rubric and returns structured feedback containing `explanation_id`, `verdict`, `got_right`, `missed`, and `hint`.

### Configure the LLM endpoint

Copy the example configuration and edit it locally:

```bash
cp task2/.env.example task2/.env
```

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` as required by your OpenAI-compatible endpoint. The evaluator reads environment variables; to load values from the file into a Bash/Zsh shell, run from the repository root:

```bash
set -a
source task2/.env
set +a
```

Never commit `.env` or expose API keys. The actual local `.env` file is intentionally not included in this project package.

### Prepare and evaluate

Run these commands from the repository root:

```bash
python scripts/task2/task2_prepare_split.py
python scripts/task2/task2_prepare_memory.py
python scripts/task2/task2_evaluate.py --split golden --confirm-final-evaluation
```

The evaluator saves predictions before joining labels for scoring. Existing golden prediction artifacts are protected against accidental overwrite; use a fresh `--output-dir` for a new run. Do not use `--limit` for a full benchmark run.

Launch the interactive demo with:

```bash
streamlit run task2/app.py
```

To generate examples for Task 1 flagged students, first ensure the expected Task 1 prediction file exists, then run:

```bash
python scripts/task2/task2_run_at_risk.py --limit 5
```

The script's default prediction path is `outputs/eventual_dropout_aligned/test_predictions_precision_50.csv`; supply `--prediction-file` if using a different artifact.

### Evaluation caveat and memory boundary

The provided 60 labelled explanations are the available labelled benchmark. The labels are kept separate from the inference inputs, and the evaluator does not pass them to the model. Do not use any of those 60 rows, duplicates, or derived cases as episodic memory. The approved memory file is empty by default; only independently verified and explicitly approved examples from outside the golden set may be added.

Once results on the 60 rows have been inspected, further prompt/model changes and reruns are **iterative benchmark tracking**, not an untouched final-test estimate. Report this limitation honestly. The labels are reference annotations and may themselves contain errors; flag suspected issues rather than silently changing them.


## Task 2: LLM Grading

### Evaluation protocol

The 60 explanations with supplied labels are the Task 2 golden evaluation set. All 60 rows are evaluated; no labelled row, label, or derived case may be used as grading memory.

The grader uses the lesson rubric and the student's explanation to predict a verdict (`understood`, `partial`, or `wrong`) and generate structured feedback. Golden labels are loaded only after predictions have been generated and saved.

Because we inspect the golden-set results to improve the system, subsequent scores represent **iterative benchmark performance**, not an untouched final test estimate.

### Reset generated Task 2 artifacts

First inspect the reset script's targets:

```bash
bash scripts/task2/reset_task2_generated.sh --dry-run
```

If all listed paths are disposable generated artifacts, reset them:

```bash
bash scripts/task2/reset_task2_generated.sh
```

The reset removes only explicitly listed paths. It must not remove the original datasets, lesson knowledge, source code, or environment configuration.

### Prepare the evaluation data and memory

Run these commands from the repository root:

```bash
python scripts/task2/task2_prepare_split.py
python scripts/task2/task2_prepare_memory.py
```

The preparation script writes:
- `task2/data/evaluation/golden_inputs.csv`: all 60 labelled explanations, without the label column.
- `task2/data/evaluation/golden_labels.csv`: reference labels kept separate from inference inputs.
- `task2/data/split_manifest.json`: row counts, class counts, and file hashes.

The memory preparation script must not import the supplied labelled dataset. Memory cases, if added in the future, must come from outside the golden set and be independently verified and explicitly approved.

### Run the golden evaluation

Use the configured LLM endpoint and model from the environment. Run:

```bash
python scripts/task2/task2_evaluate.py --confirm-final-evaluation
```

The command generates and saves predictions before joining reference labels for scoring. It refuses to overwrite existing golden prediction artifacts by default. To preserve previous runs, use a new output directory for each experiment.

Expected output files under `task2/outputs/golden/` include:

- `predictions.csv` and `predictions.jsonl`: model outputs.
- `scored_predictions.csv`: predictions joined with reference labels for analysis.
- `metrics.json`: accuracy, per-class metrics, confusion matrix, coverage, latency, and validation statistics.

Review failures, invalid outputs, feedback quality, and per-class errors alongside accuracy. Keep each experiment's model, prompt, and configuration identifiable.

### Important data policy

- Never add any of the 60 golden rows or their labels to memory.
- Never expose golden labels to the model during inference.
- Do not silently overwrite previous experiment results.
- Preserve original labels; document any independently verified corrections.
- Do not report iteratively tuned golden-set scores as an untouched test result.

## Task 2 Automated Case Memory Update

This bundle updates three project files:

- `task2/pipeline/add_memory.py`
- `task2/src/evaluator.py`
- `task2/src/vector_memory.py`

### Behavior

1. Selects candidate explanations using semantic, lexical, rubric-concept and coverage proxies. SentenceTransformer cosine similarity is used for semantic similarity when available; TF-IDF is the documented fallback.
2. Selects diverse candidates while favoring lesson coverage.
3. Automatically adds selected candidates whose evaluator verdict agrees with a provisional candidate label as `learning_case` records with `approval_status=approved_for_poc`, `label_source=poc_assumed_provisional`, and explicit provisional provenance. These are pseudo-labelled examples, not ground truth.
4. Automatically adds verified matches as `learning_case` records when source and label status indicate trusted ground truth.
5. Calls the configured API LLM to produce `error_type`, `error_description`, `correction_principle`, and `supporting_evidence` only when a trusted ground-truth label disagrees with the evaluator. If the label is provisional, a disagreement is not called an evaluator error and is not auto-added to canonical memory.
6. Never promotes golden-set examples. Chroma rebuilding is optional and explicitly requested.
7. Provisional POC records are excluded from evaluator retrieval by default. Set `TASK2_ALLOW_PROVISIONAL_MEMORY=1` only for development experiments that intentionally use pseudo-labelled cases. Keep it unset for final evaluation.

### Install

From the project root, extract the zip and copy the three files to their matching paths, replacing the current versions after backing them up.

### Recommended first run

A five-row smoke test will auto-add matching provisional examples to canonical memory, but will not create an evaluator-error case from an unverified disagreement:

```bash
python task2/pipeline/add_memory.py --limit 5 --max-selected 3
```

To also rebuild the Chroma index after adding cases (requires `chromadb` and `sentence-transformers`):

```bash
python task2/pipeline/add_memory.py --limit 5 --max-selected 3 --rebuild-chroma
```

To prevent provisional cases from being added:

```bash
python task2/pipeline/add_memory.py --limit 5 --max-selected 3 --no-provisional-learning-cases
```

For a memory-assisted POC run using provisional cases with the TF-IDF backend:

```bash
export TASK2_ALLOW_PROVISIONAL_MEMORY=1
export TASK2_RETRIEVAL_BACKEND=tfidf
python task2/pipeline/add_memory.py --limit 5 --max-selected 3
```

The API configuration remains controlled by `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`; do not hard-code credentials.

### Outputs

- `task2/outputs/memory_pipeline/candidate_evaluations.csv`
- `task2/outputs/memory_pipeline/proposed_memory_cases.jsonl`
- `task2/outputs/memory_pipeline/memory_pipeline_report.json`
- Canonical records appended to `task2/memory/approved_cases.jsonl` only under the promotion rules above.

### Important interpretation

A match to a provisional label is a pseudo-label agreement signal, not independent verification. `approved_for_poc` records should not be used in final benchmark evaluation unless the experiment explicitly calls for provisional-memory ablation. Error-correction records require trusted reference labels.


## Outputs, documentation, and archive

- `outputs/` contains Task 1 experiment metrics, predictions, and reports.
- `task2/outputs/` contains Task 2 predictions and evaluation metrics.
- `docs/` contains supporting implementation guides, evaluation protocols, findings, and experiment reports. This root README is the single README entry point.
- `archive/` contains older or superseded experiments and their artifacts. Archived material is retained for traceability and is not necessarily part of the recommended current workflow.

## Dependencies and security

Install dependencies with `pip install -r requirements.txt`. API credentials must be supplied through environment variables and must not be committed. Treat student data and evaluation artifacts according to the sharing permissions for the assignment; do not publish data unless sharing is explicitly authorized.

## Submission checklist

- [ ] Confirm the notebooks and scripts run from a clean environment.
- [ ] Recheck current Task 1 metrics and clearly state the target/evaluation protocol.
- [ ] Recheck Task 2 metrics, error cases, feedback examples, and the golden-set limitation.
- [ ] Complete the research memo comparing at least three LLM options and explaining when small ML models are preferable.
- [ ] Add the required note describing AI tools used and what they were used for.
- [ ] Verify no API keys, private environment files, or unauthorized student data are committed.

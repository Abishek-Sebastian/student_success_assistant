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

## Task 2: LLM explanation feedback and memory

Task 2 grades a student's explanation against `task2/knowledge/lesson_rubric.json` and returns a verdict (`understood`, `partial`, or `wrong`) with structured feedback. The current workflow uses one labelled dataset format per segment and keeps the label out of the inference payload.

### Configure the LLM endpoint

Copy the example configuration and set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in your shell or local `task2/.env` file. Never commit API keys or `.env`.

```bash
set -a
source task2/.env
set +a
```

### Current data layout

The dataset-generation script writes these canonical files:

- `task2/data/development/nonrisk_dataset.csv`: non-risk development examples eligible for memory construction.
- `task2/data/evaluation/risk_dataset.csv`: risk-segment examples, used for a separate evaluation.
- `task2/data/evaluation/golden_dataset.csv`: held-out golden examples, used for a separate evaluation.
- `task2/data/dataset_generation_audit.json`: counts and checks from dataset generation.

The datasets retain their source columns, including `explanation_id`, `student_id`, `lesson_id`, `explanation_text`, `label`, and available provenance columns. Evaluation code must pass only `explanation_id`, `lesson_id`, and `explanation_text` to the evaluator. The `label` column is read only after predictions have been saved, for scoring. Do not add golden rows or derived copies to memory.

### Step 1: Generate the datasets

Run from the repository root:

```bash
python scripts/task2/create_task2_datasets.py
```

The script reads `data/task2/explanations_labelled_full.csv` and `outputs/final_cascade_evaluation/outer_test_student_predictions.csv`. A student is considered at risk if any prediction row has `flag_end_to_end == 1`. Provisional students absent from the prediction file are treated as non-risk. It validates and deduplicates development examples, keeps the golden segment separate, and writes an audit report.

### Step 2: Reset the Chroma index

This backs up and verifies the current Chroma directory before removing the index. It does not edit `task2/memory/approved_cases.jsonl`.

```bash
python scripts/task2/reset_chroma_memory.py --dry-run
python scripts/task2/reset_chroma_memory.py
```

### Step 3: Build memory from non-risk development data

```bash
python task2/pipeline/add_memory.py
```

The default candidate input is `task2/data/development/nonrisk_dataset.csv`. The script validates the expected columns, evaluates/ranks candidates, writes pipeline outputs under `task2/outputs/memory_pipeline/`, appends eligible records to `task2/memory/approved_cases.jsonl`, and rebuilds Chroma unless `--skip-chroma` is passed. Its current POC policy accepts eligible non-golden candidate labels as reference labels; this is a provisional assumption, not independent ground truth. Keep this limitation explicit when interpreting results. Do not seed memory from either evaluation dataset.

The default build considers all eligible candidates; there are only three lessons in this dataset, so no artificial total or per-lesson selection cap is applied. For a smoke test only, use `--limit 5 --skip-chroma`; this limits evaluation to the first five input rows and does not represent a complete memory build.

### Step 4: Run the demo

```bash
python scripts/task2/task2_demo.py --lesson-id L1 --explanation "Your example explanation here"
```

The demo uses the same `task2/src/evaluator.py` and memory backend as the benchmark scripts.

### Step 5: Run the four benchmark experiments

The current benchmark script runs four conditions with one command:

```bash
python scripts/task2/task2_evaluate.py --dataset all --memory both
```

The four conditions are:

1. `risk__memory_off`
2. `risk__memory_on`
3. `golden__memory_off`
4. `golden__memory_on`

To run one condition or one dataset, use `--dataset risk|golden|all` and `--memory off|on|both`. Each invocation creates a timestamped folder under `task2/outputs/benchmark/` (or use `--run-id` to name it). Each condition saves `predictions.csv`, `prediction_manifest.json`, `scored_predictions.csv`, and `metrics.json`; the run root also receives `benchmark_summary.json`.

The evaluator receives only `explanation_id`, `lesson_id`, and `explanation_text`. The script writes raw predictions to disk before it separately loads the `label` column for scoring. The no-memory condition explicitly bypasses retrieval in `evaluate_one`; it does not depend on an empty index. The memory-on condition uses the canonical development memory. Since the current POC memory builder explicitly accepts provisional development labels as reference labels, the benchmark permits records marked `approved_for_poc` in the memory-on condition and records that policy in the prediction manifest. Golden examples/labels are never eligible for memory.

Metrics include accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix, valid/failed prediction counts, mean latency, and mean retrieved-case count. Invalid/failed outputs are excluded from classification scores but included in the invalid prediction rate. Compare memory-on/off runs on the same dataset and model configuration; preserve each run folder rather than overwriting prior results.

`task2/pipeline/seed_poc_memory.py` is retained in the pipeline directory for explicit/manual POC seeding only. It is not part of the default workflow and should not be run before the automated memory build unless deliberately intended.

### Scripts being retired from the active workflow

The old split-file workflow is superseded by the single labelled dataset files above. Do not use these legacy scripts/files in the new workflow:

- `scripts/task2/task2_prepare_split.py`
- `scripts/task2/task2_prepare_memory.py`
- `scripts/task2/task2_run_at_risk.py` (the new risk evaluation uses `risk_dataset.csv` directly)
- The previous `scripts/task2/task2_evaluate.py` implementation is replaced by the four-condition benchmark runner in this bundle.
- `task2/data/development/development_cases.csv`
- `task2/data/evaluation/golden_inputs.csv`
- `task2/data/evaluation/golden_labels.csv`
- `task2/data/evaluation/golden_manifest.json`
- `task2/data/split_manifest.json`

Delete the listed local files only after applying and checking the replacement workflow. The standalone `task2_build_memory_index.py` is optional because the memory builder rebuilds Chroma by default. Keep it only if you want a separate manual index-rebuild command. Update tests to the current labelled dataset layout before running them.

### Evaluation caveat

If the golden results have already been inspected and used to tune prompts, model choice, or memory, report them as iterative benchmark tracking rather than an untouched final-test estimate. Preserve labels and document any independently verified corrections instead of silently changing them.


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

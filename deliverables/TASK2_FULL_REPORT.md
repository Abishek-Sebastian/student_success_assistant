# Task 2: Student Explanation Evaluation and Feedback

## 1. Overview

Task 2 extends the Student Success Assistant by evaluating an individual student's written explanation of a lesson concept. Given a lesson identifier and an explanation, the system assesses the conceptual understanding demonstrated in the text and generates structured feedback to help the student improve.

The system combines a lesson-specific rubric, a locally served large language model (LLM), structured-output validation, and an optional retrieval-augmented episodic memory of previous explanation cases. The objective is to provide lesson-grounded assessments while maintaining a clear separation between development examples and the held-out evaluation dataset.

## 2. Problem Statement

Students may provide explanations that are conceptually correct, partially correct, incomplete, or based on misconceptions. A useful educational feedback system must distinguish between these cases and provide actionable feedback rather than merely assigning a classification label.

Task 2 formulates explanation assessment as a three-class classification and feedback-generation problem.

### 2.1 Input

For each explanation, the system receives:

- `explanation_id`: Unique identifier for the explanation.
- `lesson_id`: Identifier of the lesson whose concepts are being assessed.
- `explanation_text`: The student's written explanation.

### 2.2 Output

The evaluator produces a structured response containing:

- `verdict`: The predicted understanding category.
- `got_right`: A correct idea explicitly demonstrated in the explanation.
- `missed`: An important omission or misconception.
- `hint`: A guiding question or next step to help the student improve.
- `explanation_id`: The identifier attached by the application.
- Operational metadata, including response status, latency, retry count, retrieved-case count, and validation errors.

### 2.3 Verdict definitions

The evaluator uses three possible verdicts:

| Verdict | Definition |
|---|---|
| `understood` | Essential concepts are demonstrated correctly without a major contradiction. Minor omissions may be acceptable when the central concept is clear. |
| `partial` | Some relevant understanding is demonstrated, but an important concept is omitted or a meaningful misconception remains. |
| `wrong` | The explanation demonstrates a fundamental misconception, contradicts the central concept, or contains no meaningful correct explanation. |

The evaluator should judge the evidence present in the student's explanation. It should not infer knowledge that the student has not expressed or require exact wording from the lesson material.

## 3. Available Data

Task 2 uses student explanation records, lesson-specific knowledge, Task 1 risk predictions, and a separate labelled evaluation dataset.

### 3.1 Student explanations

The source records contain fields such as explanation identifier, student identifier, lesson identifier, explanation text, and a label where available.

The supported labels are `understood`, `partial`, and `wrong`.

Development-data preparation validates the explanation text, lesson identifier, and label. Text normalization and duplicate detection are used to remove duplicate explanations within each risk segment.

### 3.2 Lesson knowledge and rubric

The evaluator loads lesson-specific knowledge from:

`task2/knowledge/lesson_rubric.json`

The lesson rubric supplies contextual information such as:

- Lesson title and summary.
- Key concepts that a student explanation should demonstrate.
- Common misconceptions that should be identified during assessment.

The current demo supports lesson identifiers `L1`, `L2`, and `L3`.

This lesson-specific context grounds the evaluator in the intended learning objectives rather than relying exclusively on the model's general knowledge.

### 3.3 Task 1 risk predictions

The dataset-generation workflow uses Task 1 predictions to distinguish risk-associated explanations from non-risk development examples.

A student is considered at risk if any available prediction row for that student has `flag_end_to_end == 1`.

The resulting datasets are:

- **Risk dataset:** Explanations associated with students who have at least one risk flag.
- **Non-risk development dataset:** Explanations associated with students who have no risk flag across their available prediction rows.
- **Golden evaluation dataset:** A separate labelled benchmark used for scoring.

Students absent from the prediction file have unknown risk status. They should not automatically be treated as confirmed non-risk students when constructing the risk-filtered memory candidate pool.

### 3.4 Dataset files

| File | Purpose |
|---|---|
| `task2/data/development/nonrisk_dataset.csv` | Non-risk development explanations used as candidate examples for memory construction. |
| `task2/data/evaluation/risk_dataset.csv` | Explanations associated with at-risk students, used in evaluation scenarios. |
| `task2/data/evaluation/golden_dataset.csv` | Separate labelled benchmark used for evaluation and scoring. |

The generated dataset snapshot used during development contained 179 non-risk development rows, 99 risk evaluation rows, and 60 golden rows. These counts describe that particular generated snapshot; the dataset-generation audit is the source to consult after regenerating the datasets.

### 3.5 Golden-set isolation

The golden evaluation data is kept separate from memory construction and inference.

During inference, the benchmark uses only the fields required to assess the explanation:

- `explanation_id`
- `lesson_id`
- `explanation_text`

Predictions are persisted to disk and a prediction manifest is written before labels are loaded for scoring.

No golden example or golden label should be inserted into episodic memory or exposed to the model during inference. This separation helps prevent direct leakage from the evaluation labels into the system being evaluated.

## 4. System Architecture

The Task 2 pipeline consists of the following components:

1. **Dataset preparation:** Validates records, checks risk/non-risk separation, removes duplicates, and maintains golden-set isolation.
2. **Lesson knowledge store:** Supplies lesson summaries, key points, and common misconceptions.
3. **Explanation evaluator:** Constructs the prompt, optionally retrieves relevant memory cases, calls the model API, and parses the response.
4. **Output validation and recovery:** Validates the response schema and verdict, repairs empty feedback fields where configured, and retries failed requests.
5. **Memory construction pipeline:** Evaluates eligible development candidates, ranks them, checks for duplicates, and writes eligible cases.
6. **Vector index:** Stores sentence-transformer embeddings in Chroma and retrieves semantically similar cases.
7. **Benchmark runner:** Executes memory-on and memory-off conditions, saves predictions before scoring, and calculates classification and operational metrics.

## 5. Explanation Evaluation Methodology

### 5.1 Prompt construction

The evaluator constructs a prompt using the selected lesson's title, summary, key points, common misconceptions, and retrieved memory examples when memory is enabled.

The model is instructed to:

- Grade conceptual understanding against the supplied lesson rubric.
- Use only evidence from the student's explanation and the supplied lesson facts.
- Avoid requiring exact wording when the concept is expressed correctly.
- Identify one correct idea, one important omission or misconception, and a useful hint.
- Return a single JSON object containing the required fields.

The student's explanation is treated as untrusted input. Any instructions embedded in the explanation must not override the evaluator's instructions or the lesson rubric.

The application, rather than the model, attaches the explanation identifier to the final response.

### 5.2 Model-serving interface

The evaluator communicates with a model through an OpenAI-compatible `/chat/completions` endpoint.

The connection is configured through environment variables:

- `LLM_BASE_URL`: Base URL of the model server, including `/v1`.
- `LLM_API_KEY`: API key expected by the server.
- `LLM_MODEL`: Identifier of the model served by the endpoint.

The current local development setup uses an Unsloth-served Qwen model.

Credentials are supplied through environment variables and should not be committed to source control or included in the documentation.

### 5.3 Structured-output validation

The evaluator validates the model's response before returning it to the caller.

The validation process checks:

1. Whether the response contains parseable JSON.
2. Whether the required fields are present.
3. Whether the verdict belongs to the allowed label set.
4. Whether the feedback values have the expected string type.
5. Whether required feedback fields are non-empty after configured repairs.

Empty feedback fields may be repaired using explicit fallback messages. The evaluator records which fields were repaired so that these cases remain distinguishable from responses that passed validation without repairs.

If the API call or response validation fails, the evaluator records the error and follows its retry policy. If all attempts fail, it returns a failure status instead of fabricating a valid assessment.

A structurally valid response does not by itself guarantee that the educational judgement is correct.

## 6. Retrieval-Augmented Episodic Memory

### 6.1 Purpose

The memory component supplies similar prior explanations from the same lesson to provide additional context for the evaluator.

The intent is to improve calibration and consistency by making relevant examples available during assessment. Retrieved cases supplement the lesson rubric; they do not override it.

### 6.2 Canonical memory store

The canonical memory file is:

`task2/memory/approved_cases.jsonl`

Each memory record may contain:

- Case identifier.
- Lesson identifier.
- Explanation text.
- Reference or accepted verdict.
- Approval status and label source.
- Dataset split and provenance.
- Case type.
- Predicted verdict, where relevant.
- Error type and error description.
- Correction principle.
- Rubric concepts and retrieval tags.

The memory loader validates required fields, allowed verdicts, development provenance, and duplicate identifiers or explanation text. It also checks that golden-set examples have not entered the canonical memory.

### 6.3 Candidate selection

The memory-building pipeline begins with the non-risk development candidate pool.

The general workflow is:

1. Load and validate candidate explanations.
2. Evaluate candidates using the existing evaluator.
3. Calculate candidate-selection signals, including available lexical or semantic similarity, novelty, rubric-concept coverage, and lesson coverage.
4. Rank eligible candidates and apply duplicate checks and configured selection limits.
5. Construct memory records with explicit provenance.
6. Append eligible records to the canonical JSONL memory.
7. Rebuild the Chroma index when configured.
8. Write a report describing candidate processing, selection, memory updates, and index status.

The pipeline writes intermediate artifacts under `task2/outputs/memory_pipeline/`, including candidate evaluations, proposed cases, and a machine-readable report.

### 6.4 Provisional proof-of-concept records

The current proof-of-concept workflow may accept eligible non-golden development labels as provisional reference labels.

Such records are explicitly marked with provenance such as:

- `approval_status = approved_for_poc`
- `label_source = poc_assumed_provisional`

These are pseudo-labelled examples, not independently verified ground truth.

Agreement between the evaluator and a provisional label is an agreement signal under the proof-of-concept assumption; it is not independent confirmation of correctness. Likewise, disagreement with a provisional label must not be described as a confirmed evaluator error.

The distinction between provisional and independently verified cases must be preserved in the memory store, index metadata, and interpretation of benchmark outcomes.

### 6.5 Retrieval backends

The evaluator supports two retrieval backends, selected by `TASK2_RETRIEVAL_BACKEND`.

#### TF-IDF retrieval

The TF-IDF backend:

- Reads eligible cases from the canonical JSONL memory.
- Restricts cases to the same lesson and development split.
- Applies approval, provenance, and label-validity filters.
- Excludes exact normalized text matches.
- Represents explanations using TF-IDF unigram and bigram features.
- Ranks cases using cosine similarity and returns up to the requested number of positive-similarity cases.

#### Chroma vector retrieval

The Chroma backend is implemented in `task2/src/vector_memory.py`.

It uses:

- Persistent Chroma storage.
- A collection named `approved_task2_cases`.
- Sentence-transformer embeddings.
- Same-lesson, approval-status, and development-split filters.
- Exclusion of the same case identifier and exact normalized text matches.

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

Retrieved cases include their identifiers, lesson, verdict, case type, provenance, and similarity information. The similarity score is derived from the vector distance and represents text proximity in the embedding space, not correctness of the stored verdict.

If `TASK2_RETRIEVAL_BACKEND` is unset, the evaluator defaults to TF-IDF. Set it to `chroma` to use vector retrieval.

Provisional memory access is controlled separately by `TASK2_ALLOW_PROVISIONAL_MEMORY`. Enabling this variable permits retrieval of explicitly provisional cases for development experiments; it does not increase the evidential reliability of their labels.

### 6.6 Index maintenance

The canonical JSONL file is the source of truth, while Chroma is a derived retrieval index.

When canonical memory changes, the index should be rebuilt and checked. The pipeline report records the requested rebuild, resulting collection count when available, and any rebuild error.

A successful index build confirms that records were indexed. It does not independently verify their labels or establish that memory improves assessment quality.

## 7. Benchmark Design

The benchmark supports four core conditions.

| Dataset | Memory condition | Purpose |
|---|---|---|
| Risk | Off | Baseline assessment for risk-associated explanations without retrieved cases. |
| Risk | On | Assessment of the same risk dataset with retrieval enabled. |
| Golden | Off | Baseline assessment on the isolated labelled benchmark. |
| Golden | On | Assessment of the golden benchmark with retrieval enabled while preserving golden-set isolation. |

For each condition, the runner records the predicted verdict, feedback fields, execution status, attempt count, latency, retrieved-case count, validation errors, and runtime errors.

Predictions are saved to `predictions.csv`, and a prediction manifest is written before labels are read for scoring. Scored predictions and aggregate metrics are saved separately.

Invalid predictions should be identified explicitly rather than silently coerced into valid labels.

### 7.1 Classification metrics

The benchmark can use the following metrics:

- **Precision:** Among examples predicted to belong to a class, the proportion that truly belongs to that class.
- **Recall:** Among examples belonging to a class, the proportion correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** The distribution of true versus predicted verdicts.
- **Operational metrics:** Valid-response rate, failure rate, latency, retry count, and number of retrieved cases.

Because this is a three-class task, the averaging method for aggregate metrics, such as macro or weighted averaging, should be stated explicitly. Per-class metrics are also useful because performance can differ across `understood`, `partial`, and `wrong`.

Memory-on and memory-off comparisons should use the same inputs and scoring policy.

## 8. Reproducibility and Operating Procedure

Run commands from the repository root.

### 8.1 Generate Task 2 datasets

```bash
python scripts/task2/create_task2_datasets.py
```

Inspect the dataset-generation audit. Verify that risk and non-risk students do not overlap and that golden examples remain isolated.

### 8.2 Configure the local model endpoint

```bash
export LLM_BASE_URL="http://127.0.0.1:8888/v1"
export LLM_API_KEY="<configured-local-api-key>"
export LLM_MODEL="openresearchtools/Qwen3.5-4B-Instruct-GGUF"
```

These values reflect the local development setup and should match the server actually running. Never commit the API key.

### 8.3 Configure retrieval

For Chroma retrieval with provisional POC memory:

```bash
export TASK2_RETRIEVAL_BACKEND=chroma
export TASK2_ALLOW_PROVISIONAL_MEMORY=1
```

For TF-IDF retrieval:

```bash
export TASK2_RETRIEVAL_BACKEND=tfidf
```

Provisional memory should be enabled only for experiments that explicitly intend to use provisional cases.

### 8.4 Build memory

The full memory builder can be invoked using:

```bash
python task2/pipeline/add_memory.py
```

A limited smoke test can be run first:

```bash
python task2/pipeline/add_memory.py --limit 5 --max-selected 3 --max-per-lesson 2
```

Inspect the pipeline report at:

`task2/outputs/memory_pipeline/memory_pipeline_report.json`

Check candidate counts, selected cases, provenance, golden-set exclusion, and any Chroma rebuild error. The builder is append-oriented, so canonical memory should not be deleted merely to repeat a run.

### 8.5 Smoke-test the evaluator

```bash
python scripts/task2/task2_demo.py \
  --lesson-id L3 \
  --explanation "Describe a student's explanation here." \
  --explanation-id DEMO_SMOKE_001
```

Use an explanation appropriate to the selected lesson. A smoke test checks API connectivity, output structure, and retrieval behavior; it is not a substitute for benchmark evaluation.

### 8.6 Run benchmark conditions

```bash
python scripts/task2/task2_evaluate.py --dataset all --memory both
```

Use the script's `--help` output to confirm the available options in the checked-out version. Inspect each run's prediction manifest and output files before interpreting aggregate metrics.

## 9. Project File Map

| Path | Responsibility |
|---|---|
| `task2/knowledge/lesson_rubric.json` | Lesson summaries, key points, and common misconceptions. |
| `task2/src/evaluator.py` | Prompt construction, retrieval dispatch, API calls, parsing, validation, retries, and output fields. |
| `task2/src/vector_memory.py` | Chroma index construction and vector retrieval. |
| `task2/pipeline/add_memory.py` | Candidate evaluation, selection, memory-record construction, and pipeline reporting. |
| `task2/memory/approved_cases.jsonl` | Canonical episodic memory records. |
| `task2/memory/chroma_db/` | Persistent Chroma vector index. |
| `task2/data/development/nonrisk_dataset.csv` | Non-risk development candidate pool. |
| `task2/data/evaluation/risk_dataset.csv` | Risk evaluation data. |
| `task2/data/evaluation/golden_dataset.csv` | Isolated labelled benchmark. |
| `task2/outputs/memory_pipeline/` | Candidate evaluations, proposals, and memory-build report. |
| `scripts/task2/task2_demo.py` | Single-example evaluator smoke test. |
| `scripts/task2/task2_evaluate.py` | Benchmark runner. |

## 10. Safeguards and Limitations

The implementation incorporates several safeguards:

- Lesson-specific rubric grounding.
- Explicit treatment of student explanations as untrusted input.
- A fixed verdict vocabulary and structured-output validation.
- Transparent recording of repaired fields and failed requests.
- Same-lesson retrieval and exact-text exclusion.
- Separation of golden data from memory construction and inference.
- Explicit provenance for provisional memory records.
- Persistence of predictions before scoring labels are loaded.

The system remains a prototype. A language model can produce plausible but incorrect judgements, and schema validation does not guarantee pedagogical correctness. Similarity measures text proximity rather than correctness or conceptual equivalence. Retrieval behavior depends on the candidate pool, filtering rules, text representation, and quality of stored labels.

Provisional POC records are not independently verified ground truth and must be interpreted accordingly. End-to-end behavior also depends on API availability, model latency, response parsing, validation, and retrieval configuration.

## 11. Summary

Task 2 implements a rubric-grounded evaluator for student explanations that returns a three-class verdict and targeted feedback. It combines lesson-specific knowledge, an OpenAI-compatible local LLM endpoint, structured-output validation, and optional TF-IDF or Chroma retrieval of prior development cases.

The memory workflow preserves case provenance and separates the golden benchmark from inference and memory construction. Together, these components provide a reproducible framework for assessing explanation quality while making the system's assumptions, safeguards, and limitations explicit.

# Student Success Assistant — Task 2 Implementation Guide

## 1. Purpose and scope

Task 2 uses an LLM to evaluate a student's explanation against lesson-specific key points and return:
- `verdict`: `understood`, `partial`, or `wrong`;
- `got_right`: a specific correct idea demonstrated by the student;
- `missed`: an essential omitted concept or misconception;
- `hint`: a prompt that helps the student improve without simply giving the full answer.

The model is an evaluator and feedback writer. It should not invent what the student said, use student-supplied instructions as grading instructions, or consult the human answer label at inference time.

This document separates:
- **Design intent:** how the pipeline is intended to work based on the implementation history.
- **Evidence from the uploaded output bundle:** what the metrics and predictions actually demonstrate.
- **Items to verify in source:** behavior that cannot be proven from output files alone.

## 2. End-to-end request flow

1. Load environment configuration (`.env` for local development; `.env.example` documents expected variable names).
2. Receive `lesson_id`, `explanation_id`, and `explanation_text`.
3. Load the relevant lesson rubric from semantic memory.
4. Retrieve up to two approved, similar examples from the same lesson from episodic memory.
5. Build a prompt with lesson facts, grading definitions, retrieved examples, and the student explanation clearly delimited as untrusted data.
6. Call the configured OpenAI-compatible chat completions endpoint (the local Unsloth API in the tested runs).
7. Parse and validate the model response.
8. Retry only on configured structural failures, with a bounded attempt count.
9. Validate feedback fields separately from verdict/schema validity.
10. Save prediction, latency, retrieval metadata, validation events, and final status.
11. In evaluation mode, join predictions to human labels only after predictions have been produced, then compute metrics.

**Important:** output JSON validity, verdict correctness, and feedback quality are three separate quality dimensions. A record can have valid JSON and still have an invalid feedback field such as the literal string `None`.

## 3. The four memory types

### 3.1 Semantic memory — lesson knowledge

Contains authoritative lesson information: core concepts, key points, definitions, acceptable paraphrases, and common misconceptions. In the current task there are three lessons:
- L1: photosynthesis;
- L2: Newton's First Law and inertia;
- L3: supply and demand.

Recommended organization: maintain one rubric file per lesson, or one manifest with three separate lesson-keyed entries. For example:

```text
task2/memory/semantic/
  L1_photosynthesis.json
  L2_newtons_first_law.json
  L3_supply_demand.json
```

**Should each lesson have its own database?** With only three small lesson plans, separate files or separate namespaces in one JSON/YAML store are simpler and safer than three separate database servers. Retrieval must always filter by `lesson_id`. Separate physical databases are justified if lessons have different access controls, update owners, retention rules, or very large content. The important isolation boundary is the query/filter, not necessarily a separate database process.

A lesson rubric should specify:
- required concepts;
- optional concepts;
- misconceptions;
- what `understood`, `partial`, and `wrong` mean for that lesson;
- acceptable paraphrases;
- feedback guidance.

### 3.2 Episodic memory — reviewed cases

Stores past examples with human-verified labels and evidence. A case can contain:
- case ID and lesson ID;
- explanation text or normalized text;
- verified verdict;
- evidence for correct concepts;
- missed concepts/misconceptions;
- source/provenance;
- review status and version.

Only approved human-labelled development examples should be retrievable. A model's own predictions must not automatically become approved cases. Golden examples and labels must never be in the case bank.

Suggested organization:

```text
task2/memory/episodic/
  approved_cases_L1.jsonl
  approved_cases_L2.jsonl
  approved_cases_L3.jsonl
  manifest.json
```

One shared file with a required `lesson_id` field also works. Per-lesson files reduce accidental cross-lesson retrieval and make auditing straightforward.

### 3.3 Procedural memory — how the system behaves

Stores instructions and policy rather than factual content:
- verdict rubric;
- prompt template and version;
- retrieval algorithm and top-k;
- response schema;
- retry policy;
- feedback validation/repair policy;
- retention and human-review rules;
- evaluation protocol.

Procedural memory should be versioned. A change to the prompt or repair policy is a new pipeline version and must be reflected in experiment records.

### 3.4 Working memory — current request

Temporary context for one inference: the current lesson, student explanation, retrieved examples, and model response. It should not persist automatically after the request, and it must not be mixed into long-term memory.

## 4. Case-Based Reasoning (CBR)

CBR has four stages:

### Retrieve
- Filter cases to the current `lesson_id`.
- Exclude the current explanation ID.
- Exclude exact duplicate text and, ideally, near-duplicates where evaluation leakage is a concern.
- Retrieve a small number of most similar approved examples (currently the observed output files commonly report 2 retrieved cases).
- Record the retrieved case IDs and similarity scores.

### Reuse
Pass retrieved examples as labelled precedents, not as rules that override the current explanation. The prompt must say the case is an example and the current student's response must be judged independently.

### Revise
- Parse the generated JSON.
- Validate exact allowed verdict.
- Validate required feedback fields semantically.
- If structural JSON is broken, perform a bounded structural retry.
- If a field is blank or a sentinel such as `None`, attempt a targeted feedback repair or mark feedback invalid.
- Do not silently replace bad content and count it as model-generated feedback.
- Never allow a repair process to change the human label or read the golden label.

### Retain
Do not retain model outputs automatically. A human reviewer can approve a corrected case, record the corrected evidence, and add it to a new version of development memory. Preserve provenance and version history. Golden examples remain excluded.

## 5. JSON and output validation

Validation has multiple layers:

1. **Transport:** API response exists and contains message content.
2. **Parse:** content is valid JSON (strip known fences/preambles only when safe; use balanced-brace extraction cautiously).
3. **Schema:** required keys exist; verdict belongs to the allowed enum; fields have the expected types.
4. **Semantic field checks:** `got_right`, `missed`, and `hint` are not blank or sentinel strings such as `None`, `null`, `n/a`, or `nan`; content is relevant and non-generic.
5. **Pedagogical checks:** feedback is supported by the lesson, accurately describes the student's answer, and the hint doesn't simply reveal the full answer.

The uploaded golden metrics report zero JSON/schema failures and zero retries. However, `got_right` contains the literal string `None` for 12 rows while the repair count remains zero. This proves schema checks alone are insufficient and semantic field validation needs to be added or verified.

Recommended separate metrics:
- valid verdict rate;
- valid JSON/schema rate;
- feedback-field validity rate;
- feedback repair rate;
- human-audited feedback quality;
- classification accuracy on valid predictions;
- coverage;
- end-to-end correct predictions over all examples;
- mean/median latency.

## 6. Evaluation split discipline

### Development
Used to debug, compare retrieval strategies, adjust prompts, and inspect errors. All changes should be recorded. Avoid leakage: a development example cannot retrieve itself or an exact duplicate with its human label.

### Golden
A held-out, human-labelled dataset that the model has not seen. No human intervention during final predictions. Predictions must be generated and saved before labels are joined for scoring. The same golden set must not be reused as “unseen” after tuning based on its errors.

The uploaded 42-row golden result is **consumed** for the evaluated model/configuration. If changes are made in response to its errors, use a new untouched human-labelled golden set for a new final claim.

### Metrics
Always report accuracy alongside:
- support per class;
- precision, recall, F1;
- confusion matrix;
- scorable coverage;
- JSON/schema failures;
- feedback-field validity/repair rate;
- latency;
- model ID and configuration.

With a 42-row golden set, one example changes accuracy by about 2.38 percentage points. Avoid overclaiming generalization.

## 7. Environment configuration

Use `.env.example` as the template and keep `.env` out of Git. Typical values:

```dotenv
LLM_BASE_URL=http://127.0.0.1:8888/v1
LLM_API_KEY=your-local-api-key
LLM_MODEL=exact-model-id-from-v1-models
```

Load in a shell if desired:

```bash
set -a
source .env
set +a
```

Never commit `.env` or print the API key in logs. Confirm the exact model ID from the API's `/v1/models` response. Python can also load `.env` via `python-dotenv` if the evaluator initializes it.

## 8. Current evidence from the uploaded outputs

### Golden
- 42 examples, balanced with 14 per class.
- 40 correct, 2 errors: accuracy 95.24%.
- 100% valid verdict coverage.
- Zero JSON/schema failures and zero retries in the metrics.
- Mean latency about 8.98 seconds; median about 8.70 seconds.
- Both classification errors are `partial` labelled examples predicted as `understood`.
- `got_right` is the literal string `None` in 12 predictions, despite a reported repair count of zero.

### Development artifacts in the uploaded ZIP
- 18 rows.
- 15 valid outputs and 3 failed outputs.
- 83.3% coverage.
- 100% accuracy among the 15 valid outputs.
- Mean latency about 24.21 seconds.

These development metrics differ from the later terminal run previously shared in the conversation. Confirm which run the files represent before combining results in a report.

### Limits of the uploaded output artifacts
The prediction files do not include original explanation text, lesson IDs, retrieved case IDs, or prompt/memory version hashes. Therefore they do not independently prove which cases were retrieved or why the two golden errors occurred. Add those fields to future evaluation artifacts (with appropriate privacy controls).

## 9. Recommended next steps

1. Fix semantic validation for literal `None` and other sentinel feedback values.
2. Add tests for all missing/invalid fields, invalid verdicts, malformed JSON, prompt injection, current-example retrieval exclusion, duplicate exclusion, and golden-memory exclusion.
3. Store `lesson_id`, retrieved case IDs, similarity, prompt version, model ID, and memory hash in every prediction.
4. Review the two golden errors after the fact; do not tune and reuse that set as an untouched test.
5. Compare CBR against a lesson-rubric-only baseline on development data using identical examples and configuration apart from retrieval.
6. Freeze a version before creating/running a new held-out golden set.
7. Conduct a separate feedback-quality audit. Accuracy alone does not measure whether hints are useful or whether feedback is grounded.

## 10. Glossary

- **Semantic memory:** authoritative lesson knowledge.
- **Episodic memory:** reviewed past examples.
- **Procedural memory:** rules for prompting, retrieval, validation, and retention.
- **Working memory:** temporary context for the current prediction.
- **CBR:** retrieve, reuse, revise, retain.
- **Golden set:** untouched human-labelled final evaluation examples.
- **Coverage:** proportion of examples with a valid scorable prediction.
- **Feedback repair:** code or a second model action that fixes incomplete feedback; report separately from model-generated content.

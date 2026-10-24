# Student Success Assistant Task 2: Output Audit

## Scope and caveats

This report analyzes the uploaded `outputs(2).zip`, which contains development and golden metrics plus prediction CSV/JSONL and scored-prediction files. It does not contain the current source tree, original explanation texts, lesson IDs on each prediction, or retrieved case IDs. Therefore, it can assess aggregate metrics and output quality, but cannot independently prove retrieval provenance or explain the semantic cause of each error.

## Executive summary

- **Golden set:** 42 examples, 42 valid outputs, 95.24% accuracy, 100% scorable coverage.
- **Golden errors:** 2 / 42 predictions. Both errors are human-labelled `partial` but predicted `understood`.
- **JSON syntax/schema metrics:** the metrics file reports zero schema failures and zero retries. However, the prediction CSV contains the literal string `None` in `got_right` for **12 predictions**. That is a semantic/output-quality defect not counted by the current repair metrics.
- **Latency:** mean 8.98s, median 8.70s, minimum 3.79s, maximum 14.99s.
- **Development artifact is not comparable to the latest terminal run unless confirmed:** the uploaded development metrics show 15/18 valid outputs, 3 failures, 24.21s mean latency and 100.00% accuracy on valid outputs. This differs from the previously pasted development run, so treat the uploaded development files as a separate/stale run until timestamps/configuration are checked.

## Golden classification results

The golden set is balanced: 14 human labels per class.

| Human label | Predicted wrong | Predicted partial | Predicted understood | Total |
|---|---:|---:|---:|---:|
| wrong | 14 | 0 | 0 | 14 |
| partial | 0 | 12 | 2 | 14 |
| understood | 0 | 0 | 14 | 14 |

- Accuracy: **95.24% (40/42)**.
- Macro F1: **95.21%**.
- `wrong`: precision 1.00, recall 1.00, F1 1.00 (14 cases).
- `partial`: precision 1.00, recall 0.857, F1 0.923 (14 cases).
- `understood`: precision 0.875, recall 1.00, F1 0.933 (14 cases).

### Main classification insight: partial-to-understood overgrading

The only two errors are false acceptances of partial understanding as full understanding. This suggests the rubric or prompt should make the distinction between `partial` and `understood` explicit: an answer should not be `understood` merely because it contains one correct idea if an essential lesson point is omitted. Because the golden set has now been used to identify this pattern, do not tune and then reuse it as an untouched final test. Use these errors as post-evaluation observations; future tuning requires a new untouched golden set for a new final claim.

## Critical feedback-quality defect

The golden `predictions.csv` has `got_right` equal to the literal string `None` for **12 rows**. The affected IDs are:

E0006, E0252, E0275, E0306, E0335, E0429, E0438, E0537, E0538, E0553, E0644, E0713

The metrics report zero repaired feedback fields, so the current validation apparently treats `"None"` as a non-empty string rather than an invalid feedback value. This means *valid JSON does not equal valid feedback*. Add semantic validation for each required feedback field:
- trim whitespace;
- reject empty values and case-insensitive sentinel strings such as `none`, `null`, `n/a`, `nan`;
- require each field to be meaningful for the verdict and lesson;
- if it fails, run a targeted repair or mark the feedback invalid and count it separately;
- never replace a missing field with generic text and then count it as model-generated feedback.

For `wrong` answers, `got_right` need not invent a correct idea. Better output could explicitly say what part, if anything, was directionally correct, or say in a clear and consistent policy that no correct key idea was demonstrated. It must not return the literal `"None"`.

## Retrieval / CBR observations

The golden prediction file reports 2 retrieved cases for 27/42 examples and 0 retrieved cases for 15/42. The files do **not** include retrieved case IDs, so we cannot verify which cases were used, whether they came from the correct lesson, or whether any duplicate/near-duplicate example was retrieved.

Add to each prediction record:
- `lesson_id`;
- `retrieved_case_ids`;
- similarity score and retrieval method;
- `memory_version` / case-bank hash;
- `prompt_version` and model ID;
- `attempt_count`, raw validation failures, field-level validation and repairs.

Golden examples must never enter memory. Memory retrieval must be development-only and filtered by lesson. The current uploaded outputs alone do not prove those safeguards, so verify them in code and with automated tests.

## Development output caution

The uploaded development files report 100.00% accuracy on 15 valid predictions but only 83.33% coverage, with 3 failed outputs. The accuracy is conditional on the successful subset and should not be presented alone. Report both accuracy and coverage.

The development prediction file also has `retrieved_cases` counts but no case IDs, explanation text, or lesson IDs. That limits diagnosis and reproducibility.

## Prioritized actions

1. **Fix semantic field validation:** reject `"None"`/`"null"`/blank strings; add tests that the current output would fail.
2. **Improve output artifacts:** include input lesson ID, explanation hash or text in a protected artifact, retrieved case IDs, memory/prompt versions, and per-field validation diagnostics.
3. **Audit the two classification errors** against the original explanations and lesson rubric. The uploaded output bundle doesn't contain their source texts, so the precise reasoning error cannot be determined from this ZIP alone.
4. **Verify memory isolation and retrieval filters** in the implementation; add tests excluding current ID, duplicate text, golden IDs, and cross-lesson cases.
5. **Freeze current version and document this golden result as consumed.** No tuning followed by re-reporting this set as unseen.
6. **If tuning based on these observations, create a new human-labelled, untouched golden set** before making a new final performance claim.
7. **Test feedback usefulness separately from verdict accuracy.** Use a predefined rubric for correctness, specificity, pedagogical value, hint-not-answer, and consistency with the verdict.

## What this run supports

It supports the statement: *On this 42-row golden set, the current run classified 40 rows correctly (95.24%) and returned a valid verdict on every row, with mean latency around 8.98 seconds.*

It does **not** establish multilingual performance, production-scale throughput, CBR benefit over a no-retrieval baseline, or that all feedback fields are useful and valid.

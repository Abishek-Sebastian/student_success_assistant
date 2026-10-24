Good. We'll proceed with local embeddings + persistent Chroma + LLM-assisted annotation with human approval, while preserving the existing evaluator and its golden-set safeguards.

The first implementation milestone should be a reliable data and memory pipeline, not a new prompt. Your existing evaluator already has lesson-rubric-first grading, structured feedback validation, and a retrieval interface. The approved memory is empty by default, and the 60 labelled golden examples must remain excluded from memory.&#x20;

evaluator.py

README(1).md



## 1. Agreed architecture

Source explanations + Task 1 risk predictions

Keep source records and student IDs traceable

Split registry and annotation workflow

Golden set / at-risk evaluation / development corpus

LLM-assisted annotation

Candidate verdict, rationale and feedback → human review → approved record

Canonical records + local Chroma index

Verified labels remain authoritative; embeddings support retrieval

Existing evaluator + controlled experiments

Rubric + retrieved examples → verdict and feedback → measured evaluation

## 2. Dataset policy we will enforce

| Dataset                                    | Permitted use                                          |
| ------------------------------------------ | ------------------------------------------------------ |
| 60 existing labelled rows                  | Protected benchmark; never retrieval memory            |
| At-risk explanations selected using Task 1 | Separately labelled operational evaluation set         |
| Remaining eligible explanations            | Annotation, development, and approved retrieval memory |
| Model predictions without verified labels  | Candidate annotations only, never trusted memory       |

A critical safeguard: we must check for duplicate explanations and student-level overlap across splits. If an explanation from an evaluation set is copied into the development corpus, it must not become eligible for retrieval. We will retain stable IDs, source provenance, and split membership.

The at-risk set should also be frozen before Task 2 predictions are examined. If it is subsequently used to tune prompts or retrieval, its results become development results too.

## 3. Implementation plan

I recommend doing this in six small milestones. Each should have tests and a clear completion criterion.

1. Milestone 1: Freeze data splits and build the annotation workflow

   Extend the current data-preparation pipeline to produce:

   - A manifest of golden, at-risk evaluation, development, and excluded records.
   - A deduplicated pool of eligible explanations for annotation.
   - An annotation file with candidate labels, rationales, reviewer decisions, and provenance.
   - Explicit protection against golden-set rows and duplicates entering memory.

   The existing split script already separates the 60 labels from inference inputs, so preserve that behavior.&#x20;

   task2_prepare_split.py

   Done when: Every candidate has a known source and split, and protected records cannot be approved into memory.

2. Milestone 2: Add LLM-assisted annotation with human approval

   Generate proposed annotations containing:

   - `proposed_verdict`
   - `proposed_rationale`
   - `proposed_got_right`
   - `proposed_missed`
   - `proposed_hint`

   Add reviewer fields such as `final_verdict`, `review_status`, `reviewer_notes`, and `reviewed_at`.

   The LLM proposal is not authoritative. Only records explicitly approved by a human enter the canonical training corpus. A CSV-based review workflow is sufficient initially; we can integrate it into Streamlit later.

   Done when: Pending or rejected annotations cannot enter retrieval memory.

3. Milestone 3: Build persistent local Chroma

   Use a local embedding model and persistent Chroma collection, with lesson-aware retrieval.

   Store the stable record ID and useful metadata in Chroma, but keep the canonical annotation records in JSONL or SQLite. This lets us rebuild the vector index if the embedding model changes.

   Recommended metadata includes `lesson_id`, `final_verdict`, `review_status`, `source_split`, `source_dataset`, and `memory_version`.

   Done when: The same approved records remain retrievable after restarting the application, and no protected evaluation example can be retrieved.

4. Milestone 4: Integrate retrieval into the existing evaluator

   Refactor `retrieve_cases()` behind a retrieval interface rather than rewriting the entire `evaluate_one()` pipeline.

   Preserve the current prompt rules, output schema, JSON parsing, retries, and feedback validation. Expand the retrieved payload to include verified rationales and feedback examples when useful.

   Done when: Existing evaluator tests still pass, retrieved records are lesson-relevant, and a zero-memory configuration continues to work.

5. Milestone 5: Run controlled experiments

   Compare the existing baseline against:

   1. Lesson rubric only.
   2. Rubric plus fixed verified examples.
   3. Rubric plus Chroma retrieval.
   4. Rubric plus retrieval and verified correction examples.

   Track accuracy, macro-F1, per-class recall, feedback quality, latency, and coverage. Use development data for selecting settings. The 60-row golden score should be reported as iterative benchmark tracking because it has already been inspected.

6. Milestone 6: Add controlled memory updates

   Record prediction errors and reviewer corrections in a correction registry. Use similarity to surface relevant corrections, not to decide whether a prediction was wrong.

   Update the canonical records and rebuild or update Chroma only after approval. Every experiment should capture the prompt version, rubric version, model, embedding model, and memory version.

   Done when: Every memory change is traceable and reproducible.

## 4. Technology choices

| Component             | Initial choice                                     | Reason                                         |
| --------------------- | -------------------------------------------------- | ---------------------------------------------- |
| LLM                   | Existing configured Qwen endpoint                  | Preserve the working baseline                  |
| Embeddings            | Local sentence-embedding model                     | Avoid sending student text to an embedding API |
| Vector store          | Chroma with local persistence                      | Semantic search across restarts                |
| Canonical annotations | JSONL initially, SQLite if review complexity grows | Easy inspection and versioning                 |
| Annotation review     | CSV workflow initially                             | Fastest path to a defensible dataset           |
| Evaluation            | Existing scripts, extended with retrieval metrics  | Preserve golden-set safeguards                 |

For embeddings, we should select a compact model that runs comfortably on your machine, then benchmark retrieval quality on the actual lesson explanations. The exact embedding model and Chroma version should be pinned in the project's dependencies and configuration.

## 5. What I would implement first

Before changing `evaluator.py`, I would build Milestone 1 and Milestone 2 together. The current `task2_prepare_memory.py` deliberately refuses unapproved records and creates an empty memory file. We need to extend that workflow carefully, rather than bypass those protections.&#x20;

task2_prepare_memory.py



One additional issue to fix in the existing at-risk script: it selects the first eligible explanation per student, not necessarily the most recent one.&#x20;

task2_run_at_risk.py

&#x20;Before creating the operational evaluation set, we should establish the intended sampling rule and inspect the actual data columns.



I haven't modified your files yet. The next implementation step is to establish the actual split counts and overlap constraints from `data/explanations.csv` and the Task 1 prediction artifact, then build the annotation pipeline around those verified counts. That avoids choosing a split policy based on assumptions about how many eligible examples remain.
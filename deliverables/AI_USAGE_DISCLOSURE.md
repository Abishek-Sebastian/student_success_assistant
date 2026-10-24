# AI Usage Disclosure
## Student Success Assistant

### Purpose
This note discloses how AI tools were used while developing the Student Success Assistant submission, including both Task 1 (student dropout-risk prediction) and Task 2 (student-explanation evaluation and feedback). The purpose is to make the development process and the role of AI assistance transparent.

### How AI was used

AI assistance was used as a development aid for brainstorming, coding support, debugging, experiment design, evaluation design, and documentation. In particular, it helped with the following work.

#### Task 1: Student dropout-risk prediction
- Discussing candidate features for predicting eventual student dropout from demographic and learning-attempt data.
- Comparing candidate models and evaluation approaches, including logistic regression, gradient boosting/XGBoost, class weighting, student-level oversampling, and kernel SVM experiments.
- Drafting and refining scripts, diagnosing code errors, and organizing experiment outputs.
- Reviewing validation design, including student-level grouping, leakage risks, threshold selection on development data, and the distinction between ranking metrics and thresholded classification metrics.
- Explaining results and helping prepare reports, README material, and experiment documentation.

#### Task 2: Explanation evaluation and feedback
- Helping plan the LLM-based evaluator that assesses a student's explanation against lesson-specific summaries, key points, and common misconceptions.
- Assisting with prompt design and output validation so the evaluator returns a structured JSON response containing a verdict (`understood`, `partial`, or `wrong`), what the student got right, what was missed, and a learning hint.
- Supporting implementation and debugging of the evaluator's API integration, response parsing, validation, retry behavior, and feedback-field repairs.
- Helping design a retrieval-augmented memory workflow that stores eligible development examples and retrieves similar cases for the same lesson. The implementation supports TF-IDF retrieval and optional Chroma vector retrieval using sentence-transformer embeddings.
- Supporting the memory-build and diagnostic workflow, including checks of lesson filters, metadata, retrieval behavior, exact-text exclusion, and the separation of the golden evaluation set from memory construction.
- Helping structure benchmark runs that compare risk and golden evaluation datasets with memory disabled and enabled, while keeping labels out of inference and using them only for scoring.
- Assisting with documentation of limitations and safeguards, including the distinction between provisional proof-of-concept labels and independently verified labels.

The Task 1 and 2 implementation uses an OpenAI-compatible API interface configured to connect to a locally served Unsloth/Qwen model. AI assistance helped with implementation and troubleshooting; it should not be interpreted as a claim that every output or proposed memory case was independently validated.


### My contributions
I directed the project and made the substantive project decisions, including:
- Interpreting the task requirements and prioritizing a working, explainable submission under a limited time budget.
- Defining and retaining the A/B/C cascade approach and the eventual-dropout prediction objective, given that the dataset does not provide a reliable dropout timestamp.
- Deciding which experiments were worth pursuing, reviewing their results, and choosing when to stop exploring new approaches.
- Developing the project-specific modeling hypotheses, including richer attempt-history and per-lesson features, imbalance handling, kernel methods, and the C3 prefix-sampling idea.
- Reviewing the implications of student-level splits, validation thresholds, class imbalance, and the risk of overfitting on a small dataset.
- Deciding to keep the existing cascade and dataset structure stable while comparing models within individual stages.
- Directing the final evaluation to report stage-wise and end-to-end precision, recall, F1, and related metrics.
- Defining the intended Task 2 workflow, reviewing the evaluator and memory architecture, running smoke tests, inspecting diagnostics, and deciding how to interpret the provisional-memory limitation.
- Integrating the selected changes into the project, checking that the code runs, interpreting results, and taking responsibility for explaining the design and its limitations.

### Responsibility
I am responsible for the final submitted work, including the decisions made, the correctness of the reported results, the handling of data, and explaining the implementation. AI assistance is disclosed to make the development process transparent. The final submission should accurately distinguish between independently verified results and proof-of-concept behavior, and should not describe provisional memory labels as independently verified ground truth.

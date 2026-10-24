# Student Success Assistant

Initial milestone: data audit and exploratory data analysis.

## Current status
- `notebooks/01_eda.ipynb`: audit, data-quality checks, four exploratory analyses, and preliminary cleaning/model-readiness notes.
- `data/`: supplied synthetic task data copied locally for reproducibility.
- `outputs/metrics/`: CSV audit tables generated when the notebook is executed from the project root.

## Setup
Recommended Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab
```

Open and run `notebooks/01_eda.ipynb` from the project root.

## Data handling
The supplied data is synthetic. Keep data files out of any public repository unless sharing is explicitly authorized. API credentials must be stored in environment variables and never committed.

## Important modeling caveat
The 14-day prediction cutoff is a working assumption to be validated. Do not train on events occurring after that cutoff. Refund status is examined for EDA but excluded from the primary early-warning feature set until its timing is established.

## Kernel SVM experiment (current modeling work)
- `scripts/05_run_kernel_svm_benchmark.py`: compares linear and RBF SVMs with existing model controls while preserving the A/B/C cascade and current snapshot datasets.
- `outputs/kernel_svm_benchmark/`: grouped-validation AP tables, selected configurations, test predictions, and cascade metrics.
- `docs/KERNEL_SVM_EXPERIMENT.md`: experiment findings and limitations.

Run the benchmark from the project root with:

```bash
python scripts/05_run_kernel_svm_benchmark.py --outer-seeds 17 42 73 --inner-folds 2 --precision-floors 0.4 --top-k 3 --feature-sets compact compact_plus_module
```

The RBF and degree-2 polynomial kernels are near-tied candidates for Stage C only; the polynomial-inclusive run did not establish a reliable end-to-end gain. Stage B remains logistic-regression-led, and Stage A has weak signal. SVM decision margins are ranking scores, not calibrated probabilities. The RBF-only run is preserved under `archive/outputs/kernel_svm_benchmark_rbf_only/`.

## Stage C: Degree-3 polynomial SVM with bagging/boosting
- `scripts/08_run_poly3_kernel_ensemble_benchmark.py` compares the degree-3 polynomial SVM, student-cluster bagging of polynomial SVMs, an AdaBoost/SVM attempt, and logistic regression / Gradient Boosting controls. It is Stage C only; A/B models, snapshot data, labels, and cascade routing are unchanged.
- Bagging bootstraps whole students and carries each sampled student's C-prefix rows together; numeric preprocessing is fitted within each training fold. Validation and test are grouped by student.
- Outputs are written to `outputs/poly3_kernel_ensemble/`, including OOF AP, per-split test metrics, student-level test predictions, and a Markdown report. AdaBoost failures are recorded instead of stopping the run.

Run from the project root:

```bash
python scripts/08_run_poly3_kernel_ensemble_benchmark.py --outer-seeds 17 42 73 --inner-folds 2 --precision-floor 0.4
```

In the initial three-split run, student-cluster bagging had the highest mean development OOF AP, but it did not consistently meet the precision floor on held-out splits. The class-weighted degree-3 SVM met the floor on all three splits. AdaBoost could not be fit because its SVM base learner was worse than random under the fold weighting. Treat this as an exploratory Stage C comparison, not a proven end-to-end cascade improvement.

## Task 2: LLM explanation evaluation

Task 2 lives in `task2/` and is designed to run against an OpenAI-compatible Unsloth API. See [`task2/README.md`](task2/README.md) for setup, the held-out golden evaluation protocol, memory boundaries, and commands. Task 2 is isolated from the existing Task 1 models and outputs.

## Local LLM configuration (Task 2)

Create your private local configuration from the committed template:

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY and LLM_MODEL to the values shown by Unsloth.
```

The Task 2 Python code loads `.env` automatically using `python-dotenv`. If you also want to export those variables into your current shell (for commands or tools that only read shell environment variables), run this from the repository root in Bash or Zsh:

```bash
set -a
source .env
set +a
```

Confirm the configured endpoint/model without printing your API key:

```bash
printf 'LLM_BASE_URL=%s\nLLM_MODEL=%s\n' "$LLM_BASE_URL" "$LLM_MODEL"
curl -s "$LLM_BASE_URL/models" -H "Authorization: Bearer $LLM_API_KEY"
```

Keep `.env` private; `.env.example` contains placeholders only and is safe to commit.

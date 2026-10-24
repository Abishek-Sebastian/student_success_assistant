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

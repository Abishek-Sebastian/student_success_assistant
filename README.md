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

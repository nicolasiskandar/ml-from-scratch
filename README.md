# ML from scratch

From scratch implementations of classic ML algorithms using only Python and `numpy`.
Each algorithm lives in its own folder, ships with a Jupyter sandbox, and has a
short demo video showing the results.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run the Jupyter notebooks:

```bash
pip install -e ".[notebooks]"
```

To run tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Repository layout

- `linear-regression/` — `LinearRegression` with feature standardization,
  gradient descent, L2 regularization, and supporting notebooks.
- `logistic-regression/` — `LogisticRegression` (L2) with probability-based
  training, cost history tracking, and notebooks.
- `k-means-clustering/` — `KMeansClustering` that either accepts a fixed `k`
  or uses an elbow heuristic, plus demo notebooks.
- `tests/` — pytest unit tests covering happy paths, edge cases, and error
  conditions for all algorithms.
- `utils.py` — shared utilities (`compute_mu_sigma`, `standardize`) used by
  the regression models.
- `pyproject.toml` — project metadata and dependency management.

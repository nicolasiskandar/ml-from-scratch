# ML from scratch

From scratch implementations of classic ML algorithms using only Python and `numpy`.
Each algorithm lives in its own folder, ships with a Jupyter
sandbox, and has a short demo video showing the results.

## Repository layout
- `linear-regression/`: `LinearRegression` with feature standardization,
  gradient descent, L2 regularization, and supporting notebooks.
- `logistic-regression/`: `LogisticRegression` (L2) with probability-based
  training, cost history tracking, and notebooks.
- `k-means-clustering/`: `KMeansClustering` that either accepts a fixed `k`
  or uses an elbow heuristic, plus demo notebooks.

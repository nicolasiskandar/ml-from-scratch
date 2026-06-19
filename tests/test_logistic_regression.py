import numpy as np
import pytest
from logistic_regression import LogisticRegression


class TestLogisticRegression:
    def test_fit_predict_synthetic(self):
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=5000)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean(preds == y) > 0.9

    def test_predict_before_fit_raises(self):
        model = LogisticRegression()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_predict_returns_binary(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=2000)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_bounds(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=2000)
        model.fit(X, y)
        x_norm = (X - model.mu) / model.sigma
        probas = model._predict_proba(x_norm)
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0)

    def test_cost_history_decreases(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=500)
        model.fit(X, y)
        assert len(model.cost_history) == 500
        assert model.cost_history[-1] < model.cost_history[0]

    def test_l2_regularization(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.01, nb_of_iterations=3000, lambda_=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean(preds == y) > 0.8

    def test_single_feature(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=2000)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean(preds == y) > 0.9

    def test_perfect_separation(self):
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [10.0], [11.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=5000, lambda_=0.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.array_equal(preds, y)

    def test_single_sample(self):
        np.random.seed(42)
        X = np.random.randn(1, 3)
        y = np.array([1])
        model = LogisticRegression(learning_rate=0.01, nb_of_iterations=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (1,)

    def test_verbose_output(self, capsys):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=100, verbose=True)
        model.fit(X, y)
        captured = capsys.readouterr()
        assert "Iteration" in captured.out

    def test_one_class(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.ones(50, dtype=int)
        model = LogisticRegression(learning_rate=0.1, nb_of_iterations=500)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds == 1)

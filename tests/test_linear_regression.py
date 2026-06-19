import numpy as np
import pytest
from linear_regression import LinearRegression


class TestLinearRegression:
    def test_fit_predict_synthetic(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=5000)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean((preds - y)**2) < 0.05

    def test_predict_before_fit_raises(self):
        model = LinearRegression()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_cost_history_decreases(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + 0.5*X[:, 1] + np.random.randn(50)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=500)
        model.fit(X, y)
        assert len(model.cost_history) == 500
        assert model.cost_history[-1] < model.cost_history[0]

    def test_l2_regularization(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        true_w = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
        y = X @ true_w + np.random.randn(50)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=3000, lambda_=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean((preds - y)**2) < 0.5

    def test_single_feature(self):
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 3*X[:, 0] + np.random.randn(50)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=2000)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean((preds - y)**2) < 0.05

    def test_single_sample(self):
        np.random.seed(42)
        X = np.random.randn(1, 3)
        y = np.array([2.0])
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=100)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (1,)

    def test_constant_feature(self):
        np.random.seed(42)
        n = 100
        X = np.ones((n, 2))
        X[:, 0] = np.random.randn(n)
        y = 2*X[:, 0] + np.random.randn(n)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=3000)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.mean((preds - y)**2) < 0.1

    def test_early_stopping(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100)*0.1
        model = LinearRegression(learning_rate=0.1, nb_of_iterations=10000, epsilon=1e-2)
        model.fit(X, y)
        assert len(model.cost_history) < 500

    def test_verbose_output(self, capsys):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + np.random.randn(30)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=100, verbose=True)
        model.fit(X, y)
        captured = capsys.readouterr()
        assert "Iteration" in captured.out

    def test_zero_learning_rate(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + np.random.randn(50)*0.1
        model = LinearRegression(learning_rate=0.0, nb_of_iterations=100)
        model.fit(X, y)
        before = model.cost_history[0]
        after = model.cost_history[-1]
        assert abs(before - after) < 1e-6

    def test_weights_and_bias_updated(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + 0.5*X[:, 1] + np.random.randn(50)*0.1
        model = LinearRegression(learning_rate=0.01, nb_of_iterations=1000)
        model.fit(X, y)
        assert model.w is not None
        assert model.b is not None
        assert np.any(model.w != 0) or model.b != 0

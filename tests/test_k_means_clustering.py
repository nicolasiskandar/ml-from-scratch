import numpy as np
import pytest
from k_means_clustering import KMeansClustering


class TestKMeansClustering:
    def test_fit_predict_fixed_k(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        model = KMeansClustering(k=3)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (100,)
        assert len(np.unique(labels)) == 3

    def test_predict_before_fit_raises(self):
        model = KMeansClustering(k=3)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_k_greater_than_samples_raises(self):
        X = np.random.randn(5, 2)
        model = KMeansClustering(k=10)
        with pytest.raises(ValueError, match="cannot be greater"):
            model.fit(X)

    def test_k_equals_one(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        model = KMeansClustering(k=1)
        model.fit(X)
        labels = model.predict(X)
        assert np.all(labels == 0)
        assert len(model.centroids) == 1

    def test_k_equals_n_samples(self):
        np.random.seed(42)
        X = np.random.randn(5, 2)
        model = KMeansClustering(k=5)
        model.fit(X)
        labels = model.predict(X)
        assert len(np.unique(labels)) == 5

    def test_auto_k_elbow(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        model = KMeansClustering(k=None, max_k=8)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (50,)
        assert model.k is not None and 1 <= model.k <= 8

    def test_single_feature(self):
        np.random.seed(42)
        X = np.random.randn(50, 1)
        model = KMeansClustering(k=3)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (50,)

    def test_centroids_stored(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        model = KMeansClustering(k=3)
        model.fit(X)
        assert len(model.centroids) == 3
        for c in model.centroids:
            assert c.shape == (2,)

    def test_inertia_computed(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        model = KMeansClustering(k=3)
        model.fit(X)
        inertia = model._compute_inertia(X)
        assert isinstance(inertia, float)
        assert inertia >= 0

    def test_predict_with_different_x(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 2)
        X_test = np.random.randn(10, 2)
        model = KMeansClustering(k=3)
        model.fit(X_train)
        labels = model.predict(X_test)
        assert labels.shape == (10,)

    def test_assign_indices_returns_correct_shape(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansClustering(k=4)
        model.fit(X)
        indices = model._assign_indices(X)
        assert indices.shape == (30,)
        assert np.all(indices >= 0) and np.all(indices < 4)

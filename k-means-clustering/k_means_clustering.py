import numpy as np


class KMeansClustering:
    def __init__(self, k=None, max_k=10, max_iterations=300, tol=1e-4):
        self.k = k
        self.max_k = max_k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = []
        self.pt_to_centroid_idx = []

    def predict(self, x):
        if not self.centroids:
            raise ValueError("Model not fitted yet.")
        return self._assign_indices(x)

    def fit(self, x):
        if self.k is None:
            self.k, _ = self._find_best_k(x)

        m = x.shape[0]

        if self.k > m:
            raise ValueError("k cannot be greater than number of data points")

        self.pt_to_centroid_idx = np.empty(m)
        self.centroids = [None] * self.k

        indices = np.random.choice(m, self.k, replace=False)
        self.centroids = [x[i].astype(float) for i in indices]

        for _ in range(self.max_iterations):
            self.pt_to_centroid_idx = self._assign_indices(x)
            prev = np.stack(self.centroids)

            for i in range(len(self.centroids)):
                points = x[self.pt_to_centroid_idx == i]
                if len(points) == 0:
                    self.centroids[i] = x[np.random.randint(m)]
                else:
                    self.centroids[i] = np.mean(points, axis=0).astype(float)

            shift = np.max(np.linalg.norm(
                prev - np.stack(self.centroids), axis=1))
            if shift <= self.tol:
                break

    def _assign_indices(self, x):
        labels = np.empty(len(x), dtype=int)
        for i in range(len(x)):
            labels[i] = self._get_centroid_index_having_min_distance(x[i])
        return labels

    def _get_centroid_index_having_min_distance(self, point):
        min_distance = float('inf')
        min_idx = 0

        for i in range(len(self.centroids)):
            distance = self._calculate_distance_between(
                point, self.centroids[i])
            if distance < min_distance:
                min_distance = distance
                min_idx = i

        return min_idx

    def _calculate_distance_between(self, x, y):
        return np.linalg.norm(x - y)

    def _find_best_k(self, x):
        inertias = []

        for k in range(1, self.max_k + 1):
            model = KMeansClustering(k=k,
                                     max_iterations=self.max_iterations,
                                     tol=self.tol)
            model.fit(x)
            inertias.append(model._compute_inertia(x))

        # elbow method
        # how much is the diff
        deltas = np.diff(inertias)

        # how much that diff is changing (shrinking)
        second_deltas = np.diff(deltas)

        # get the min change (the elbow)
        # +2 to convert array index back to k value (approx)
        best_k = np.argmin(second_deltas) + 2

        return best_k, inertias

    def _compute_inertia(self, x):
        inertia = 0.0
        for i, point in enumerate(x):
            centroid = self.centroids[self.pt_to_centroid_idx[i]]
            inertia += np.linalg.norm(point - centroid) ** 2
        return inertia

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=1e-2, nb_of_iterations=10_000, epsilon=1e-8, verbose=False, lambda_=0.1):
        self.learning_rate = learning_rate
        self.nb_of_iterations = nb_of_iterations
        self.epsilon = epsilon
        self.verbose = verbose
        self.lambda_ = lambda_
        self.w = None
        self.b = 0
        self.cost_history = []
        self.mu = None
        self.sigma = None

    def predict(self, x):
        if self.mu is None or self.sigma is None:
            raise ValueError("Model not fitted yet.")

        x = self._normalize(x)
        return np.dot(x, self.w) + self.b

    def fit(self, x, y):
        m = x.shape[0]
        self.w = np.zeros(x.shape[1])
        prev_cost = 0

        # Normalize
        self.mu = np.mean(x, axis=0)
        # added 1e-8 to prevent /0 if std=0
        self.sigma = np.std(x, axis=0) + 1e-8
        x = self._normalize(x)

        for i in range(self.nb_of_iterations):
            predictions = np.dot(x, self.w) + self.b
            errors = predictions - y

            dJ_dw = (1 / m) * np.dot(x.T, errors) + (self.lambda_ / m) * self.w
            dJ_db = (1 / m) * np.sum(errors)

            self.w = self.w - self.learning_rate * dJ_dw
            self.b = self.b - self.learning_rate * dJ_db

            cost = self._get_cost(errors)
            self.cost_history.append(cost)

            if self.verbose and i % int(self.nb_of_iterations * 0.1) == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")

            if i > 0 and abs(prev_cost - cost) < self.epsilon:
                break
            prev_cost = cost

    def _get_cost(self, errors):
        m = errors.shape[0]
        SSE = np.sum((errors) ** 2)
        reg_term = self.lambda_ * np.sum(self.w ** 2)
        J_wb = (SSE + reg_term) / (2 * m)
        return J_wb

    def _normalize(self, x):
        return (x - self.mu) / self.sigma

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=1e-2, nb_of_iterations=10_000, epsilon=1e-8, verbose=False, lambda_=0.1):
        self.learning_rate = learning_rate
        self.nb_of_iterations = nb_of_iterations
        self.epsilon = epsilon
        self.verbose = verbose
        self.lambda_ = lambda_
        self.w = None
        self.b = 0
        self.cost_history = []

    def predict(self, x):
        if self.mu is None or self.sigma is None:
            raise ValueError("Model not fitted yet.")

        x = self._normalize(x)
        return (self._predict_proba(x) >= 0.5).astype(int)

    def fit(self, x, y):
        m = x.shape[0]
        self.w = np.zeros(x.shape[1])
        prev_cost = 0

        # normalize
        self.mu = np.mean(x, axis=0)

        # added 1e-8 to prevent /0 if std=0
        self.sigma = np.std(x, axis=0) + 1e-8
        x = self._normalize(x)

        for i in range(self.nb_of_iterations):
            predictions = self._predict_proba(x)
            errors = predictions - y

            dJ_dw = (1 / m) * np.dot(x.T, errors) + (self.lambda_ / m) * self.w
            dJ_db = (1 / m) * np.sum(errors)

            self.w = self.w - self.learning_rate * dJ_dw
            self.b = self.b - self.learning_rate * dJ_db

            cost = self._get_cost(predictions, y)
            self.cost_history.append(cost)

            if self.verbose and i % int(self.nb_of_iterations * 0.1) == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")

            if i > 0 and abs(prev_cost - cost) < self.epsilon:
                break
            prev_cost = cost

    def _get_cost(self, predictions, y):
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)

        m = y.shape[0]
        log_loss = - (1 / m) * np.sum(y * np.log(predictions) +
                                      (1 - y) * np.log(1 - predictions))
        reg_term = (self.lambda_ / (2 * m)) * np.sum(self.w ** 2)

        J_wb = log_loss + reg_term
        return J_wb

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _predict_proba(self, x):
        return self._sigmoid(np.dot(x, self.w) + self.b)

    def _normalize(self, x):
        return (x - self.mu) / self.sigma

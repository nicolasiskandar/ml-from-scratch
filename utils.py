import numpy as np


def compute_mu_sigma(x, eps=1e-8):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0) + eps
    return mu, sigma


def standardize(x, mu, sigma):
    return (x - mu) / sigma

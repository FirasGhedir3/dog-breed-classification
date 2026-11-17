"""Une petite PCA construite sur la SVD, avec la même interface fit / transform que scikit-learn."""
import numpy as np


class SimplePCA:
    """Centre les données, calcule la SVD, garde les premières directions.

    Avec X_c = U S V^T, les lignes de V^T sont les directions principales et la
    variance le long de chacune est s_i^2 / (n - 1).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = self.n_components or len(s)
        variance = s**2 / (len(X) - 1)
        self.components_ = vt[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / variance.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return np.asarray(Z) @ self.components_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

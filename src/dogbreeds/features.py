"""Transformateurs scikit-learn qui transforment des images 64x64 aplaties en vecteurs de features."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from .hog import compute_hog_fast
from .preprocessing import SIZE


class PerClassPCA(BaseEstimator, TransformerMixin):
    """Une PCA par race.

    fit() apprend un petit sous-espace pour chaque race, avec uniquement les images
    d'entraînement de cette race. transform() projette chaque image sur tous ces sous-espaces,
    ce qui donne n_races * n_components features. L'idée : un chien se projette fortement
    sur le sous-espace de sa propre race.
    """

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.pcas_ = {c: PCA(n_components=self.n_components).fit(X[y == c]) for c in self.classes_}
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.hstack([self.pcas_[c].transform(X) for c in self.classes_])


class HOGFeatures(BaseEstimator, TransformerMixin):
    """Descripteur HOG de chaque image. Rien à apprendre, donc fit() ne fait rien."""

    def __init__(self, cell=16, n_bins=8):
        self.cell = cell
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        images = np.asarray(X, dtype=float).reshape(-1, SIZE, SIZE)
        return np.array([compute_hog_fast(img, self.cell, self.n_bins) for img in images])

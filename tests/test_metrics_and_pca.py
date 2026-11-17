import numpy as np
from sklearn.decomposition import PCA

from dogbreeds.metrics import cosine_similarity, entropy
from dogbreeds.pca import SimplePCA


def test_entropy_of_balanced_classes_is_log_k():
    assert np.isclose(entropy([10, 10, 10, 10]), np.log(4))


def test_entropy_of_single_class_is_zero():
    assert entropy([50, 0, 0]) == 0.0


def test_entropy_accepts_counts_or_probabilities():
    assert np.isclose(entropy([30, 70]), entropy([0.3, 0.7]))


def test_cosine_similarity_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert np.isclose(cosine_similarity(p, p), 1.0)


def test_simple_pca_matches_sklearn():
    X = np.random.default_rng(1).normal(size=(100, 12)) @ np.random.default_rng(2).normal(size=(12, 12))
    mine, ref = SimplePCA(n_components=4).fit(X), PCA(n_components=4).fit(X)
    assert np.allclose(mine.explained_variance_ratio_, ref.explained_variance_ratio_)
    # les composantes sont définies au signe près
    assert np.allclose(np.abs(mine.transform(X)), np.abs(ref.transform(X)))


def test_simple_pca_full_reconstruction_is_exact():
    X = np.random.default_rng(3).normal(size=(30, 8))
    pca = SimplePCA().fit(X)
    assert np.allclose(pca.inverse_transform(pca.transform(X)), X)

"""Petites fonctions pour décrire une distribution de classes."""
import numpy as np


def entropy(counts):
    """Entropie de Shannon (log népérien) d'une distribution de classes.

    Accepte des effectifs ou des probabilités. Elle vaut 0 si tout est dans une seule classe
    et ln(K) quand les K classes sont parfaitement équilibrées.
    """
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]  # une classe vide donnerait log(0)
    return float(-np.sum(p * np.log(p)))


def cosine_similarity(p, q):
    """Vaut 1.0 quand deux distributions de classes ont exactement les mêmes proportions."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q)))

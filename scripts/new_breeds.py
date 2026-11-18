"""Est-il facile d'ajouter de nouvelles races ? Comparaison PCA+HOG, SIFT et VGG16 (si ses features existent).

Protocole : on entraîne avec 4 races « anciennes » en entier, puis on ajoute 2 races « nouvelles »
avec seulement n images d'entraînement chacune (n = 5, 10, 20, 50, 100). On mesure sur le test :
  - la précision sur les images des 2 races nouvelles (classification à 6 races),
  - la précision sur les images des 4 races anciennes (le modèle les oublie-t-il ?),
  - le temps de ré-entraînement.
Les features SIFT et VGG16 ne dépendent pas des races : seul le SVM est ré-entraîné.
La PCA par classe doit en plus apprendre un sous-espace pour chaque nouvelle race.
Tous les SVM utilisent class_weight="balanced", sinon les races rares ne sont presque jamais prédites.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dogbreeds.data import PROCESSED_FILE, load_flat, load_processed, split
from dogbreeds.pipeline import build_pipeline
from dogbreeds.plotting import INK, clean_axes, save_figure
from sift_features import dense_sift

NEW_BREED_PAIRS = [("chow", "basset"), ("Chihuahua", "groenendael"), ("Kerry blue terrier", "malinois")]
SAMPLE_SIZES = [5, 10, 20, 50, 100]
N_SPLITS = 5
VGG_FILE = PROCESSED_FILE.parent / "vgg16_features.npz"


def make_methods():
    """Nom -> (matrice de features, constructeur du modèle). Les features sont calculées une fois."""
    X_flat, y = load_flat()
    X_img, _ = load_processed()
    methods = {
        "PCA + HOG": (X_flat, lambda: build_pipeline(5, classifier=SVC(C=1, gamma="scale", class_weight="balanced"))),
        "SIFT": (dense_sift(X_img),
                 lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", class_weight="balanced"))),
    }
    if VGG_FILE.exists():
        methods["VGG16"] = (np.load(VGG_FILE)["F"],
                            lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", class_weight="balanced")))
    else:
        print("(features VGG16 absentes : lancez d'abord scripts/vgg16_features.py pour les inclure)")
    return methods, y


def main():
    methods, y = make_methods()
    results = {name: {n: {"new": [], "old": [], "time": []} for n in SAMPLE_SIZES} for name in methods}

    for new_breeds in NEW_BREED_PAIRS:
        is_new = np.isin(y, new_breeds)
        for seed in range(N_SPLITS):
            idx_train, idx_test = split(np.arange(len(y)), y, seed=seed)[:2]
            rng = np.random.default_rng(seed)
            for n in SAMPLE_SIZES:
                # toutes les images d'entraînement des races anciennes + n images de chaque race nouvelle
                keep = [i for i in idx_train if not is_new[i]]
                for breed in new_breeds:
                    candidates = [i for i in idx_train if y[i] == breed]
                    keep += list(rng.choice(candidates, size=min(n, len(candidates)), replace=False))
                keep = np.array(keep)
                for name, (features, build) in methods.items():
                    start = time.perf_counter()
                    model = build().fit(features[keep], y[keep])
                    elapsed = time.perf_counter() - start
                    pred = model.predict(features[idx_test])
                    results[name][n]["new"].append(100 * np.mean(pred[is_new[idx_test]] == y[idx_test][is_new[idx_test]]))
                    results[name][n]["old"].append(100 * np.mean(pred[~is_new[idx_test]] == y[idx_test][~is_new[idx_test]]))
                    results[name][n]["time"].append(elapsed)

    print(f"Précision sur le test, moyenne sur {len(NEW_BREED_PAIRS)} paires de races x {N_SPLITS} splits (hasard = 16,7 %)")
    print(f"{'méthode':10s} | {'n':>4s} | {'races nouvelles':>15s} | {'races anciennes':>15s} | {'temps':>7s}")
    for name in methods:
        for n in SAMPLE_SIZES:
            r = results[name][n]
            print(f"{name:10s} | {n:4d} | {np.mean(r['new']):13.1f} % | {np.mean(r['old']):13.1f} % | {np.mean(r['time']):5.2f} s")

    colors = {"PCA + HOG": "#1f6fb4", "SIFT": "#eb6834", "VGG16": "#1baf7a"}
    markers = {"PCA + HOG": "o", "SIFT": "s", "VGG16": "^"}
    fig, ax = plt.subplots(figsize=(8.6, 5), dpi=160)
    for name in methods:
        ax.plot(SAMPLE_SIZES, [np.mean(results[name][n]["new"]) for n in SAMPLE_SIZES], color=colors[name],
                marker=markers[name], ms=6, lw=1.8, label=name)
    ax.axhline(100 / 6, color="#6b7280", lw=1, ls=":")
    ax.text(SAMPLE_SIZES[-1], 100 / 6 + 1, "chance", ha="right", fontsize=8.5, color="#6b7280")
    ax.set_xlabel("Training images per new breed", color=INK, fontsize=10)
    ax.set_ylabel("Accuracy on the 2 new breeds (%)", color=INK, fontsize=10)
    ax.set_title("Adding new breeds with few examples", fontsize=11.5, color=INK)
    ax.set_xticks(SAMPLE_SIZES)
    clean_axes(ax, grid="both")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#d1d5db", fontsize=10, loc="upper left")
    save_figure(fig, "new_breeds.png")


if __name__ == "__main__":
    main()

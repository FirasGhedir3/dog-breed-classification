"""Split stratifié, pipeline complète avec k-NN, robustesse sur 10 splits et erreur en fonction de k.

k est choisi par validation croisée dans le train. Le test sert uniquement à
rapporter l'erreur finale.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from dogbreeds.data import SEED, load_flat, split
from dogbreeds.metrics import cosine_similarity
from dogbreeds.pipeline import build_pipeline
from dogbreeds.plotting import INK, clean_axes, save_figure

KS = np.arange(1, 41)
N_SPLITS = 10


def class_shares(y, classes):
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    return counts / counts.sum()


def knn_error_for_all_k(F_fit, y_fit, F_eval, y_eval, ks):
    """Erreur du k-NN pour tous les k d'un coup : les voisins ne sont triés qu'une fois."""
    classes, fit_idx = np.unique(y_fit, return_inverse=True)
    eval_idx = np.searchsorted(classes, y_eval)
    neighbours = NearestNeighbors(n_neighbors=max(ks)).fit(F_fit).kneighbors(F_eval, return_distance=False)
    labels = fit_idx[neighbours]
    errors = []
    for k in ks:
        votes = np.zeros((len(F_eval), len(classes)), dtype=int)
        for j in range(k):
            np.add.at(votes, (np.arange(len(F_eval)), labels[:, j]), 1)
        errors.append(np.mean(votes.argmax(axis=1) != eval_idx))
    return np.array(errors)


def main():
    X, y = load_flat()
    classes = np.unique(y)

    # Split, puis vérification que train et test ont les mêmes proportions de classes
    X_train, X_test, y_train, y_test = split(X, y)
    p_train, p_test = class_shares(y_train, classes), class_shares(y_test, classes)
    print(f"Split stratifié : {len(y_train)} train / {len(y_test)} test")
    print(f"similarité cosinus des distributions de classes : {cosine_similarity(p_train, p_test):.6f}")

    # Pipeline complète : MinMax -> (PCA par classe + HOG) -> StandardScaler -> 5-NN
    model = build_pipeline(n_components=5).fit(X_train, y_train)
    pred = model.predict(X_test)
    err_train = hamming_loss(y_train, model.predict(X_train))
    err_test = hamming_loss(y_test, pred)
    print(f"5-NN : erreur train {100 * err_train:.1f} %, erreur test {100 * err_test:.1f} %")
    print(f"référence : toujours prédire la plus grande classe donne {100 * (1 - p_test.max()):.1f} % d'erreur, "
          f"le hasard {100 * (1 - 1 / len(classes)):.1f} %")

    cm = confusion_matrix(y_test, pred, labels=classes)
    off_diagonal = cm.astype(float)
    np.fill_diagonal(off_diagonal, 0)
    i, j = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    print(f"confusion la plus fréquente : {classes[i]} prédit comme {classes[j]} ({int(off_diagonal[i, j])} images)")
    for k, breed in enumerate(classes):
        print(f"  {breed:20s} bien classés : {cm[k, k]:3d} / {cm[k].sum():3d}")

    # Erreur en fonction de k, moyennée sur 10 splits différents
    train_err, cv_err, test_err = [], [], []
    for seed in range(N_SPLITS):
        X_tr, X_te, y_tr, y_te = split(X, y, seed=seed)
        extractor = build_pipeline(n_components=5)[:-1]
        F_tr = extractor.fit_transform(X_tr, y_tr)
        F_te = extractor.transform(X_te)
        train_err.append(knn_error_for_all_k(F_tr, y_tr, F_tr, y_tr, KS))
        test_err.append(knn_error_for_all_k(F_tr, y_tr, F_te, y_te, KS))

        cv = np.zeros(len(KS))
        for fit_idx, val_idx in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_tr, y_tr):
            fold_extractor = build_pipeline(n_components=5)[:-1]
            F_fit = fold_extractor.fit_transform(X_tr[fit_idx], y_tr[fit_idx])
            F_val = fold_extractor.transform(X_tr[val_idx])
            cv += knn_error_for_all_k(F_fit, y_tr[fit_idx], F_val, y_tr[val_idx], KS) * len(val_idx)
        cv_err.append(cv / len(X_tr))
    train_err, cv_err, test_err = (100 * np.array(e) for e in (train_err, cv_err, test_err))
    mean_train, mean_cv, mean_test = train_err.mean(0), cv_err.mean(0), test_err.mean(0)

    print(f"\nMoyenne sur {N_SPLITS} splits stratifiés (erreur en %)")
    print("   k | train |   CV  |  test")
    for k in (1, 3, 5, 7, 10, 15, 20, 25, 30, 40):
        print(f"{k:4d} | {mean_train[k - 1]:5.1f} | {mean_cv[k - 1]:5.1f} | {mean_test[k - 1]:5.1f}")
    k_best = int(KS[np.argmin(mean_cv)])
    print(f"k choisi par validation croisée sur le train : {k_best} "
          f"(erreur CV {mean_cv[k_best - 1]:.1f} %, erreur test {mean_test[k_best - 1]:.1f} % "
          f"+/- {test_err.std(0)[k_best - 1]:.1f})")
    print(f"pour comparaison, k = 5 : erreur test {mean_test[4]:.1f} %")

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(KS, mean_train, color="#1f6fb4", lw=1.8, marker="o", ms=5.5, label="Training error")
    ax.plot(KS, mean_test, color="#c00000", lw=1.8, marker="s", ms=5.5, label="Test error (generalization)")
    ax.set_xlabel("Number of neighbours k", color=INK, fontsize=10)
    ax.set_ylabel("Error rate (%)", color=INK, fontsize=10)
    ax.set_title("k-NN: training and test error as a function of k", fontsize=11.5, color=INK)
    ax.set_xticks(np.arange(0, KS[-1] + 1, 2))
    ax.set_xlim(0, KS[-1] + 1)
    ax.set_ylim(0, 70)
    clean_axes(ax, grid="both")
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95, edgecolor="#d1d5db")
    save_figure(fig, "knn_error_vs_k.png")


if __name__ == "__main__":
    main()

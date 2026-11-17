"""SVM (linéaire puis RBF), réglage de n_components / C / gamma par GridSearchCV, puis OvO vs OvR.

Pipeline : MinMax -> (PCA par classe + HOG) -> StandardScaler -> SVC
"""
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, hamming_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC

from dogbreeds.data import SEED, load_flat, split
from dogbreeds.pipeline import build_pipeline
from dogbreeds.plotting import INK, save_figure

BREED_ORDER = ["Chihuahua", "basset", "Kerry blue terrier", "groenendael", "malinois", "chow"]
PARAM_GRID = {
    "all_features__pca__n_components": [5, 10, 20],
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": [0.01, 0.05, 0.1, 0.25, 0.5],
}


def error(model, X, y):
    return hamming_loss(y, model.predict(X))


def plot_grid_heatmaps(grid):
    Ns = PARAM_GRID["all_features__pca__n_components"]
    Cs = PARAM_GRID["classifier__C"]
    Gs = PARAM_GRID["classifier__gamma"]
    cv_error = np.zeros((len(Ns), len(Cs), len(Gs)))
    for params, score in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]):
        cv_error[Ns.index(params["all_features__pca__n_components"]),
                 Cs.index(params["classifier__C"]),
                 Gs.index(params["classifier__gamma"])] = 100 * (1 - score)
    vmin, vmax = cv_error.min(), cv_error.max()
    best = grid.best_params_

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), dpi=160, sharey=True)
    for ax, n, matrix in zip(axes, Ns, cv_error):
        image = ax.imshow(matrix, cmap="Blues", vmin=vmin - 2, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Gs)))
        ax.set_xticklabels(Gs, fontsize=9)
        ax.set_yticks(range(len(Cs)))
        ax.set_yticklabels(Cs, fontsize=9)
        ax.set_xlabel("gamma", color=INK)
        ax.set_title(f"n_components = {n}  ({6 * n} + 128 = {6 * n + 128} features)", fontsize=9.5, color=INK, loc="left")
        for i in range(len(Cs)):
            for j in range(len(Gs)):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > (vmin + vmax) / 2 + 3 else INK)
        if best["all_features__pca__n_components"] == n:
            bi, bj = Cs.index(best["classifier__C"]), Gs.index(best["classifier__gamma"])
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#c00000", lw=2.5))
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("C", color=INK)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Cross-validation error (%)", color=INK, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("RBF SVM: 5-fold cross-validation error over the grid (red frame = best combination)",
                 fontsize=11, color=INK, x=0.04, y=1.06, ha="left")
    save_figure(fig, "svm_grid_heatmaps.png")


def plot_ovo_ovr_confusion(cms, accuracies):
    vmax = max(cm.max() for cm in cms.values())  # même échelle de couleurs sur les deux panneaux
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=160)
    for ax, (name, cmap) in zip(axes, (("One-vs-One", "Blues"), ("One-vs-Rest", "Reds"))):
        cm = cms[name]
        image = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(BREED_ORDER)))
        ax.set_yticks(range(len(BREED_ORDER)))
        ax.set_xticklabels(BREED_ORDER, rotation=45, ha="right", fontsize=8.5, color=INK)
        ax.set_yticklabels(BREED_ORDER, fontsize=8.5, color=INK)
        ax.set_xlabel("Predicted label", color=INK, fontsize=9)
        ax.set_ylabel("True label", color=INK, fontsize=9)
        for i in range(len(BREED_ORDER)):
            for j in range(len(BREED_ORDER)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.55 * vmax else INK)
        ax.set_title(f"Confusion matrix: {name}\n(Accuracy: {100 * accuracies[name]:.1f}%)",
                     fontsize=11, color=INK, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, "confusion_ovo_ovr.png")


def main():
    X, y = load_flat()
    X_train, X_test, y_train, y_test = split(X, y, seed=SEED)

    # 1) Références avec hyperparamètres par défaut
    print("== 1) Références (C = 1) ==")
    baselines = {}
    for name, svc in (("linear", SVC(kernel="linear", C=1)), ("rbf (gamma='scale')", SVC(kernel="rbf", C=1, gamma="scale"))):
        model = build_pipeline(5, classifier=svc).fit(X_train, y_train)
        baselines[name] = model
        print(f"  SVC {name:20s} : erreur train {100 * error(model, X_train, y_train):5.1f} %, "
              f"erreur test {100 * error(model, X_test, y_test):5.1f} %")

    # 2) Recherche sur grille, CV stratifiée à 5 plis dans le train
    print("\n== 2) GridSearchCV sur n_components x C x gamma ==")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = build_pipeline(5, classifier=SVC(kernel="rbf"), memory=tempfile.mkdtemp())
    start = time.time()
    grid = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=1, refit=True).fit(X_train, y_train)
    n_combinations = len(grid.cv_results_["params"])
    print(f"  {n_combinations} combinaisons x {cv.get_n_splits()} plis = {n_combinations * cv.get_n_splits()} entraînements "
          f"en {time.time() - start:.0f} s")
    best = grid.best_params_
    best_model = grid.best_estimator_
    err_train, err_test = error(best_model, X_train, y_train), error(best_model, X_test, y_test)
    print(f"  meilleurs paramètres : {best}")
    print(f"  erreur de validation croisée : {100 * (1 - grid.best_score_):.1f} %")
    print(f"  réentraîné sur tout le train : erreur train {100 * err_train:.1f} %, erreur test {100 * err_test:.1f} %")
    svc = best_model.named_steps["classifier"]
    print(f"  vecteurs de support : {svc.n_support_.sum()} / {len(y_train)} ({100 * svc.n_support_.sum() / len(y_train):.0f} % du train)")
    on_edge = [k for k, v in best.items() if v in (min(PARAM_GRID[k]), max(PARAM_GRID[k]))]
    print(f"  paramètres au bord de la grille : {on_edge if on_edge else 'aucun'}")
    plot_grid_heatmaps(grid)

    # 3) One-vs-One vs One-vs-Rest avec les hyperparamètres réglés
    print("\n== 3) One-vs-One vs One-vs-Rest ==")
    n_comp, C, gamma = best["all_features__pca__n_components"], best["classifier__C"], best["classifier__gamma"]
    make_svc = lambda: SVC(kernel="rbf", C=C, gamma=gamma)
    strategies = {"One-vs-One": OneVsOneClassifier, "One-vs-Rest": OneVsRestClassifier}
    features_train = build_pipeline(n_comp, classifier=make_svc()).fit(X_train, y_train)[:-1].transform(X_train)
    cms, accuracies, predictions = {}, {}, {}
    for name, strategy in strategies.items():
        model = build_pipeline(n_comp, classifier=strategy(make_svc())).fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        cms[name] = confusion_matrix(y_test, predictions[name], labels=BREED_ORDER)
        accuracies[name] = accuracy_score(y_test, predictions[name])
        n_binary = len(model.named_steps["classifier"].estimators_)
        timings = []
        for _ in range(5):  # temps du classifieur seul, sur des features déjà calculées
            t = time.perf_counter()
            strategy(make_svc()).fit(features_train, y_train)
            timings.append(time.perf_counter() - t)
        print(f"  {name:12s} : {n_binary:2d} classifieurs binaires | erreur train {100 * error(model, X_train, y_train):.1f} %, "
              f"erreur test {100 * (1 - accuracies[name]):.1f} % | temps d'entraînement {1000 * np.median(timings):.0f} ms")
    same = np.mean(predictions["One-vs-One"] == predictions["One-vs-Rest"])
    print(f"  prédictions identiques entre OvO et OvR : {100 * same:.1f} % des {len(y_test)} images de test")
    for n_classes in (6, 20, 100, 1000):
        print(f"  {n_classes:4d} classes : OvO demande {n_classes * (n_classes - 1) // 2:>7,d} classifieurs, OvR en demande {n_classes:>5,d}")
    plot_ovo_ovr_confusion(cms, accuracies)

    # 4) Le réglage vaut-il le coup ? RBF par défaut vs RBF réglé sur 10 splits
    print("\n== 4) Erreur de test sur 10 splits stratifiés (%) ==")
    configs = {
        "RBF par défaut (n=5, C=1, gamma='scale')": lambda: build_pipeline(5, classifier=SVC(C=1, gamma="scale")),
        f"RBF réglé (n={n_comp}, C={C}, gamma={gamma})": lambda: build_pipeline(n_comp, classifier=SVC(C=C, gamma=gamma)),
    }
    for name, build in configs.items():
        errors = []
        for seed in range(10):
            a, b, c, d = split(X, y, seed=seed)
            errors.append(100 * error(build().fit(a, c), b, d))
        print(f"  {name:42s}: {np.mean(errors):.1f} (+/- {np.std(errors):.1f})")


if __name__ == "__main__":
    main()

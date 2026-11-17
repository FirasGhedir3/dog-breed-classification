"""Que devient le k-NN quand on retire le StandardScaler ?

1. Erreur de test avec et sans standardisation (10 splits).
2. Quelles features dominent les distances du k-NN : les 30 valeurs PCA ou les 128 valeurs HOG ?
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import hamming_loss

from dogbreeds.data import SEED, load_flat, split
from dogbreeds.pipeline import build_pipeline
from dogbreeds.plotting import BLUE, INK, ORANGE, RED, LIGHT_BLUE, clean_axes, save_figure

N_PCA_FEATURES = 30  # 6 classes x 5 composantes ; les 128 autres colonnes sont du HOG
N_SPLITS = 10


def train_test_errors(scale, X_tr, y_tr, X_te, y_te):
    model = build_pipeline(n_components=5, scale=scale).fit(X_tr, y_tr)
    return hamming_loss(y_tr, model.predict(X_tr)), hamming_loss(y_te, model.predict(X_te))


def distance_shares(features):
    """Part de la distance au carré attendue qui vient des colonnes PCA vs HOG (E||x - x'||^2 = 2 * somme des variances)."""
    variance = features.var(axis=0)
    return variance[:N_PCA_FEATURES].sum() / variance.sum(), variance[N_PCA_FEATURES:].sum() / variance.sum()


def main():
    X, y = load_flat()

    # 1) Erreurs avec et sans standardisation
    X_tr, X_te, y_tr, y_te = split(X, y, seed=SEED)
    for scale in (True, False):
        err_tr, err_te = train_test_errors(scale, X_tr, y_tr, X_te, y_te)
        print(f"split {SEED}, scaler {'actif  ' if scale else 'coupé  '} : train {100 * err_tr:.1f} %, test {100 * err_te:.1f} %")

    results = {True: [], False: []}
    for seed in range(N_SPLITS):
        a, b, c, d = split(X, y, seed=seed)
        for scale in (True, False):
            results[scale].append(train_test_errors(scale, a, c, b, d))
    results = {scale: 100 * np.array(r) for scale, r in results.items()}
    print(f"\nMoyenne sur {N_SPLITS} splits (erreur en %, +/- écart-type) :")
    for scale in (True, False):
        r = results[scale]
        print(f"  scaler {'actif  ' if scale else 'coupé  '} : train {r[:, 0].mean():.1f} (+/- {r[:, 0].std():.1f}), "
              f"test {r[:, 1].mean():.1f} (+/- {r[:, 1].std():.1f})")
    delta = results[False][:, 1] - results[True][:, 1]
    print(f"  variation de l'erreur test sans scaler : {delta.mean():+.1f} points (de {delta.min():+.1f} à {delta.max():+.1f})")

    # 2) Qui décide des distances ?
    raw_features = build_pipeline(n_components=5, scale=False)[:-1].fit_transform(X_tr, y_tr)
    std = raw_features.std(axis=0)
    standardised = (raw_features - raw_features.mean(0)) / np.where(std == 0, 1, std)
    ratio = np.median(std[N_PCA_FEATURES:]) / np.median(std[:N_PCA_FEATURES])
    print(f"\nécart-type par feature avant standardisation : PCA médiane {np.median(std[:N_PCA_FEATURES]):.3f}, "
          f"HOG médiane {np.median(std[N_PCA_FEATURES:]):.3f} (x{ratio:.0f})")
    share_raw, share_std = distance_shares(raw_features), distance_shares(standardised)
    print(f"part de la distance : sans scaler PCA {100 * share_raw[0]:.2f} % / HOG {100 * share_raw[1]:.2f} % ; "
          f"avec scaler PCA {100 * share_std[0]:.1f} % / HOG {100 * share_std[1]:.1f} %")

    # Figure 1 : erreurs avec / sans scaler
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    x = np.arange(2)
    train_mean = [results[s][:, 0].mean() for s in (True, False)]
    test_mean = [results[s][:, 1].mean() for s in (True, False)]
    train_std = [results[s][:, 0].std() for s in (True, False)]
    test_std = [results[s][:, 1].std() for s in (True, False)]
    bars_train = ax.bar(x - 0.19, train_mean, 0.36, yerr=train_std, capsize=3, color=BLUE,
                        label="Training error", error_kw={"lw": 1})
    bars_test = ax.bar(x + 0.19, test_mean, 0.36, yerr=test_std, capsize=3, color=RED,
                       label="Test error (generalization)", error_kw={"lw": 1})
    for bars, values in ((bars_train, train_mean), (bars_test, test_mean)):
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["With StandardScaler", "Without StandardScaler"], fontsize=10)
    ax.set_ylabel("Error rate (%)", color=INK)
    ax.set_ylim(0, 80)
    ax.set_title(f"k-NN (k = 5): effect of feature scaling (mean of {N_SPLITS} splits, bars = ±1 std)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.93))
    clean_axes(ax, grid="y")
    save_figure(fig, "scaling_effect.png")

    # Figure 2 : échelle des features et part dans la distance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=160, gridspec_kw={"width_ratios": [1.15, 1]})
    rng = np.random.default_rng(0)
    n_hog = raw_features.shape[1] - N_PCA_FEATURES
    groups = ((std[:N_PCA_FEATURES], LIGHT_BLUE, f"PCA features ({N_PCA_FEATURES})"),
              (std[N_PCA_FEATURES:], ORANGE, f"HOG features ({n_hog})"))
    for pos, (values, color, _) in enumerate(groups):
        ax1.scatter(pos + rng.uniform(-0.16, 0.16, len(values)), np.maximum(values, 1e-3), s=14,
                    color=color, edgecolors="white", linewidths=0.4, zorder=3)
        ax1.hlines(np.median(values), pos - 0.3, pos + 0.3, color=INK, lw=1.6, zorder=4)
        ax1.text(pos + 0.33, np.median(values), f"median {np.median(values):.2f}", va="center", fontsize=8.5, color=INK)
    ax1.set_yscale("log")
    ax1.set_yticks([3, 5, 10, 20, 30])
    ax1.set_yticklabels(["3", "5", "10", "20", "30"])
    ax1.minorticks_off()
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([g[2] for g in groups])
    ax1.set_xlim(-0.5, 1.9)
    ax1.set_ylabel("Standard deviation of the feature (log scale)", color=INK)
    ax1.set_title("Scale of each feature before standardisation", fontsize=10, color=INK, loc="left")
    clean_axes(ax1, grid="y")

    rows = ["Without scaler", "With scaler"]
    for i, (pca_share, hog_share) in enumerate((share_raw, share_std)):
        ax2.barh(i, 100 * pca_share, color=LIGHT_BLUE, height=0.5, edgecolor="white", linewidth=2,
                 label="PCA features" if i == 0 else None)
        ax2.barh(i, 100 * hog_share, left=100 * pca_share, color=ORANGE, height=0.5, edgecolor="white", linewidth=2,
                 label="HOG features" if i == 0 else None)
        if pca_share > 0.07:
            ax2.text(100 * pca_share / 2, i, f"{100 * pca_share:.1f} %", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
        else:
            ax2.text(100 * pca_share + 1, i + 0.34, f"PCA {100 * pca_share:.2f} %", ha="left", va="center",
                     color=INK, fontsize=8.5)
        ax2.text(100 * pca_share + 100 * hog_share / 2, i, f"{100 * hog_share:.1f} %", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(rows)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of the k-NN distance (%)", color=INK)
    ax2.set_title("Who decides the distances?", fontsize=10, color=INK, loc="left")
    ax2.invert_yaxis()
    ax2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    clean_axes(ax2)
    ax2.spines["left"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "scaling_feature_dominance.png")


if __name__ == "__main__":
    main()

"""PCA sur les pixels bruts : quelle part de la variance est concentrée, et comment reconstruire les images ?

Ici la PCA est apprise sur tout le jeu de données car elle sert seulement à explorer.
La pipeline de classification apprend sa PCA sur le train uniquement.
"""
import matplotlib.pyplot as plt
import numpy as np

from dogbreeds.data import load_flat
from dogbreeds.pca import SimplePCA
from dogbreeds.plotting import INK, LIGHT_BLUE, clean_axes, save_figure

SIZE = 64


def main():
    X_raw, y = load_flat()
    X = X_raw / 255.0
    pca = SimplePCA().fit(X)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)

    print(f"PC1 explique {100 * ratio[0]:.1f} % de la variance, PC2 {100 * ratio[1]:.1f} %")
    for target in (0.80, 0.90, 0.95, 0.99):
        print(f"  {int(100 * target)} % de la variance demandent {np.searchsorted(cumulative, target) + 1} composantes (sur 4096)")

    # Variance expliquée
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=160)
    ax1.bar(np.arange(1, 21), 100 * ratio[:20], color=LIGHT_BLUE, width=0.7)
    ax1.set_xlabel("Principal component", color=INK)
    ax1.set_ylabel("Explained variance (%)", color=INK)
    ax1.set_title("First 20 components", fontsize=10, color=INK, loc="left")
    ax2.plot(np.arange(1, 301), 100 * cumulative[:300], color=LIGHT_BLUE, lw=2)
    for target, marker in ((0.80, "o"), (0.90, "s"), (0.95, "D")):
        k = int(np.searchsorted(cumulative, target) + 1)
        ax2.plot([k], [100 * target], marker, color=LIGHT_BLUE, ms=7, markeredgecolor="white")
        ax2.annotate(f"{int(100 * target)} %: k = {k}", (k, 100 * target), xytext=(10, -14),
                     textcoords="offset points", fontsize=8.5, color=INK)
    ax2.set_xlabel("Number of components k", color=INK)
    ax2.set_ylabel("Cumulative explained variance (%)", color=INK)
    ax2.set_title("Cumulative variance", fontsize=10, color=INK, loc="left")
    ax2.set_ylim(0, 102)
    clean_axes(ax1, grid="y")
    clean_axes(ax2, grid="y")
    fig.tight_layout()
    save_figure(fig, "pca_variance.png")

    # Reconstruction à partir de k composantes
    ks = [2, 5, 10, 20, 50, 100, 300]
    picks = [int(np.where(y == breed)[0][3]) for breed in ("chow", "Chihuahua", "malinois")]
    coords = pca.transform(X[picks])
    fig, axes = plt.subplots(len(picks), len(ks) + 1, figsize=(11, 4.6), dpi=160)
    for r, idx in enumerate(picks):
        axes[r, 0].imshow(X_raw[idx].reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(y[idx], fontsize=9, color=INK)
        if r == 0:
            axes[r, 0].set_title("Original", fontsize=9, color=INK)
        for c, k in enumerate(ks, start=1):
            kept = coords[r].copy()
            kept[k:] = 0
            rebuilt = np.clip(pca.inverse_transform(kept), 0, 1)
            mse = np.mean((rebuilt - X[idx]) ** 2)
            axes[r, c].imshow(rebuilt.reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xlabel(f"MSE {mse:.4f}", fontsize=7.5, color="#6b7280")
            if r == 0:
                axes[r, c].set_title(f"k = {k}", fontsize=9, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Reconstruction from k principal components (4096 values -> k values)",
                 fontsize=11, color=INK, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "pca_reconstruction.png")

    print("erreur de reconstruction moyenne sur le jeu de données :")
    all_coords = pca.transform(X)
    for k in ks:
        kept = all_coords.copy()
        kept[:, k:] = 0
        print(f"  k = {k:3d}: MSE {np.mean((np.clip(pca.inverse_transform(kept), 0, 1) - X) ** 2):.5f}")


if __name__ == "__main__":
    main()

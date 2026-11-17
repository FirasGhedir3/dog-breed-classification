"""Gradients de Sobel et HOG : démonstrations visuelles, puis features HOG de tout le jeu de données.

Vérifie aussi quelle part du signal HOG vient simplement du remplissage du letterbox (ratio du recadrage).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from dogbreeds.data import list_samples, load_processed
from dogbreeds.hog import compute_hog, compute_hog_fast, sobel
from dogbreeds.pca import SimplePCA
from dogbreeds.plotting import BLUE, GREY, INK, clean_axes, save_figure
from dogbreeds.preprocessing import read_and_crop

CELL, N_BINS, SIZE = 16, 8, 64


def plot_sobel(X, y):
    picks = [int(np.where(y == b)[0][5]) for b in ("Chihuahua", "basset", "chow")]
    titles = ["Image (64x64)", "gx (horizontal)", "gy (vertical)", "Magnitude"]
    fig, axes = plt.subplots(3, 4, figsize=(9, 6.6), dpi=160)
    for row, idx in enumerate(picks):
        img = X[idx].astype(float) / 255
        gx, gy, magnitude, _ = sobel(img)
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].imshow(gx, cmap="gray", vmin=-np.abs(gx).max(), vmax=np.abs(gx).max(), interpolation="nearest")
        axes[row, 2].imshow(gy, cmap="gray", vmin=-np.abs(gy).max(), vmax=np.abs(gy).max(), interpolation="nearest")
        axes[row, 3].imshow(magnitude, cmap="gray", interpolation="nearest")
        axes[row, 0].set_ylabel(y[idx], fontsize=9, color=INK)
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9.5, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Sobel filters: edges are where brightness changes abruptly", fontsize=11, color=INK, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "sobel_demo.png")


def draw_hog(ax, hog_vector):
    """Un trait par orientation dans chaque cellule, tracé le long de la direction du contour."""
    hog = hog_vector.reshape(SIZE // CELL, SIZE // CELL, N_BINS)
    centers = (np.arange(N_BINS) + 0.5) * np.pi / N_BINS
    for i in range(hog.shape[0]):
        for j in range(hog.shape[1]):
            cy, cx = i * CELL + CELL / 2 - 0.5, j * CELL + CELL / 2 - 0.5
            for b in range(N_BINS):
                half = 0.5 * CELL * hog[i, j, b] / hog.max()
                dx, dy = -np.sin(centers[b]), np.cos(centers[b])  # le contour est perpendiculaire au gradient
                ax.plot([cx - half * dx, cx + half * dx], [cy - half * dy, cy + half * dy],
                        color="white", lw=1.1, solid_capstyle="round")


def plot_hog(X, y):
    picks = [int(np.where(y == b)[0][5]) for b in ("Chihuahua", "malinois")]
    titles = ["Image + 4x4 cell grid", "Sobel magnitude", f"HOG: {N_BINS} orientation bins per cell"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.3), dpi=160)
    for row, idx in enumerate(picks):
        img = X[idx].astype(float) / 255
        magnitude = sobel(img)[2]
        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for k in range(1, SIZE // CELL):
            axes[row, 0].axhline(k * CELL - 0.5, color="#f59e0b", lw=0.9)
            axes[row, 0].axvline(k * CELL - 0.5, color="#f59e0b", lw=0.9)
        axes[row, 1].imshow(magnitude, cmap="gray", interpolation="nearest")
        axes[row, 2].imshow(np.zeros((SIZE, SIZE)), cmap="gray", vmin=0, vmax=1)
        draw_hog(axes[row, 2], compute_hog(img, CELL, N_BINS))
        axes[row, 0].set_ylabel(y[idx], fontsize=9, color=INK)
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9.5, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, SIZE - 0.5)
        ax.set_ylim(SIZE - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("HOG: each cell votes for its dominant edge orientations (stick length = vote strength)",
                 fontsize=11, color=INK, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "hog_demo.png")


def plot_hog_pca_by_breed(Z, ratio, y):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), dpi=160, sharex=True, sharey=True)
    for ax, breed in zip(axes.ravel(), sorted(set(y))):
        mask = y == breed
        ax.scatter(Z[~mask, 0], Z[~mask, 1], s=9, color=GREY, linewidths=0, zorder=1)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=14, color="#2a78d6", edgecolors="white", linewidths=0.4, zorder=2)
        ax.set_title(f"{breed} (n = {mask.sum()})", fontsize=10, color=INK, loc="left")
        clean_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f} %)", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f} %)", color=INK, fontsize=9)
    fig.suptitle("HOG features (128-d, standardised) on the first two principal components",
                 fontsize=11, color=INK, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "hog_pca_by_breed.png")


def main():
    X, y = load_processed()

    start = time.time()
    features = np.array([compute_hog_fast(img.astype(float) / 255, CELL, N_BINS) for img in X])
    print(f"HOG de tout le jeu de données : {features.shape} en {time.time() - start:.1f} s "
          f"({SIZE // CELL}x{SIZE // CELL} cellules x {N_BINS} bins = {features.shape[1]} valeurs)")
    print(f"Valeurs HOG : min {features.min():.2f}, médiane {np.median(features):.2f}, max {features.max():.2f}")

    plot_sobel(X, y)
    plot_hog(X, y)

    scaled = StandardScaler().fit_transform(features)
    pca = SimplePCA().fit(scaled)
    Z = pca.transform(scaled)
    ratio = pca.explained_variance_ratio_
    print(f"PCA des features HOG standardisées : PC1 {100 * ratio[0]:.1f} %, PC2 {100 * ratio[1]:.1f} %")
    plot_hog_pca_by_breed(Z, ratio, y)

    # Quelle part de PC1 est le ratio du recadrage (c'est-à-dire la taille des bandes noires) ?
    image_paths, annotation_paths, _ = list_samples()
    log_ratio = np.array([np.log(c.shape[1] / c.shape[0])
                          for c in (read_and_crop(i, a) for i, a in zip(image_paths, annotation_paths))])
    for name, component in (("PC1", Z[:, 0]), ("PC2", Z[:, 1])):
        print(f"corrélation de {name} (HOG) avec log(largeur/hauteur) : {np.corrcoef(component, log_ratio)[0, 1]:+.2f}")


if __name__ == "__main__":
    main()

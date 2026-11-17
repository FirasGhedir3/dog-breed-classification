"""Pourquoi le letterbox : les recadrages sont loin d'être carrés et les étirer déforme les chiens."""
import matplotlib.pyplot as plt
import numpy as np

from dogbreeds.data import list_samples
from dogbreeds.plotting import BLUE, INK, clean_axes, save_figure
from dogbreeds.preprocessing import letterbox, read_and_crop, resize_stretch


def main():
    image_paths, annotation_paths, labels = list_samples()
    crops = [read_and_crop(i, a) for i, a in zip(image_paths, annotation_paths)]
    ratio = np.array([c.shape[1] / c.shape[0] for c in crops])  # width / height

    print(f"ratio largeur/hauteur des recadrages : min {ratio.min():.2f}, médiane {np.median(ratio):.2f}, max {ratio.max():.2f}")
    print(f"{100 * np.mean(np.abs(ratio - 1) > 0.2):.0f} % des recadrages s'écartent de plus de 20 % d'un carré")

    # Histogramme des ratios
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    ax.hist(ratio, bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(1.0, color="#6b7280", lw=1, ls="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.92, "square", fontsize=8, color="#6b7280")
    ax.set_xlabel("Crop aspect ratio (width / height)", color=INK)
    ax.set_ylabel("Number of images", color=INK)
    ax.set_title("Dog crops are far from square", fontsize=11, color=INK, loc="left")
    clean_axes(ax, grid="y")
    save_figure(fig, "aspect_ratio_hist.png")

    # Recadrage le plus haut, médian, le plus large : original, étiré, letterbox
    order = np.argsort(ratio)
    picks = [order[0], order[len(order) // 2], order[-1]]
    titles = ["tallest crop", "median crop", "widest crop"]
    rows = ["Cropped (gray)", "Stretched 64x64", "Letterboxed 64x64"]
    fig, axes = plt.subplots(3, 3, figsize=(8, 8.2), dpi=160)
    for col, (idx, title) in enumerate(zip(picks, titles)):
        crop = crops[idx]
        for row, img in enumerate([crop, resize_stretch(crop), letterbox(crop)]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{title}\n{labels[idx]} ({crop.shape[1]}x{crop.shape[0]} px)", fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=9, color=INK)
    fig.suptitle("Stretching vs letterboxing", fontsize=12, color=INK, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "resize_comparison.png")


if __name__ == "__main__":
    main()

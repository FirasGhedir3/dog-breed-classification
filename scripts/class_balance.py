"""Les six races sont-elles équilibrées ? Effectifs par classe et entropie de Shannon."""
import matplotlib.pyplot as plt
import numpy as np

from dogbreeds.data import load_processed
from dogbreeds.metrics import entropy
from dogbreeds.plotting import BLUE, INK, clean_axes, save_figure


def main():
    _, y = load_processed()
    breeds, counts = np.unique(y, return_counts=True)
    n_classes, n_images = len(breeds), counts.sum()
    h, h_max = entropy(counts), np.log(n_classes)

    print(f"{n_images} images, {n_classes} races")
    for breed, count in zip(breeds, counts):
        print(f"  {breed:20s} {count:4d} ({100 * count / n_images:.1f} %)")
    print(f"entropie H = {h:.4f}, maximum ln(K) = {h_max:.4f}, H / ln(K) = {h / h_max:.4f}")
    print(f"toujours prédire la plus grande classe : {100 * counts.max() / n_images:.1f} % de précision "
          f"(hasard uniforme : {100 / n_classes:.1f} %)")
    print(f"pour comparaison, deux classes 90/10 donnent H = {entropy([90, 10]):.3f} (max {np.log(2):.3f})")

    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    bars = ax.bar(breeds[order], counts[order], color=BLUE, width=0.6)
    for bar, count in zip(bars, counts[order]):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 3, str(count), ha="center", fontsize=9, color=INK)
    ax.axhline(n_images / n_classes, color="#6b7280", lw=1, ls="--")
    ax.text(n_classes - 0.5, n_images / n_classes + 3, f"perfect balance ({n_images / n_classes:.0f})",
            ha="right", fontsize=8, color="#6b7280")
    ax.set_ylabel("Number of images", color=INK)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_title(f"Class distribution in SmallDB\nH = {h:.3f}  /  ln(K) = {h_max:.3f}  (H/ln K = {h / h_max:.3f})",
                 fontsize=11, color=INK, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    clean_axes(ax, grid="y")
    save_figure(fig, "class_distribution.png")


if __name__ == "__main__":
    main()

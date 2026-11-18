"""La PCA comme générateur de données : fabriquer de nouveaux chiens à partir de coordonnées tirées au hasard.

Pour chaque race, on apprend une PCA sur les images d'entraînement, on tire des coordonnées
dans l'espace réduit selon la variance de chaque composante, puis `inverse_transform` les
reconvertit en images 64x64. On teste ensuite si ces images synthétiques aident le SVM.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.svm import SVC

from dogbreeds.data import SEED, load_flat, split
from dogbreeds.pca import SimplePCA
from dogbreeds.pipeline import build_pipeline
from dogbreeds.plotting import INK, save_figure

SIZE = 64
N_COMPONENTS = 20  # composantes gardées par race
N_SYNTHETIC = 100  # images synthétiques ajoutées par race


def generate(X_train, y_train, n_per_breed, rng, n_components=N_COMPONENTS):
    """Images synthétiques (valeurs 0..255) et leurs races, générées à partir du train uniquement."""
    images, labels = [], []
    for breed in np.unique(y_train):
        real = X_train[y_train == breed] / 255.0
        pca = SimplePCA(n_components=n_components).fit(real)
        std = np.sqrt(pca.explained_variance_)  # écart-type le long de chaque composante
        coords = rng.normal(0.0, std, size=(n_per_breed, n_components))
        fake = np.clip(pca.inverse_transform(coords), 0, 1) * 255.0
        images.append(fake)
        labels += [breed] * n_per_breed
    return np.vstack(images), np.array(labels)


def plot_examples(X_train, y_train, rng):
    breeds = list(np.unique(y_train))
    n_real, n_fake = 3, 5
    synthetic, fake_labels = generate(X_train, y_train, n_fake, rng)
    fig, axes = plt.subplots(len(breeds), n_real + n_fake, figsize=(11, 8), dpi=160)
    for row, breed in enumerate(breeds):
        real = X_train[y_train == breed][:n_real]
        fake = synthetic[fake_labels == breed]
        for col, img in enumerate(list(real) + list(fake)):
            ax = axes[row, col]
            ax.imshow(img.reshape(SIZE, SIZE), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0 and col in (0, n_real):
                ax.set_title("Real images" if col == 0 else "Synthetic images", fontsize=10, color=INK, loc="left")
        axes[row, 0].set_ylabel(breed, fontsize=8.5, color=INK)
    fig.suptitle("PCA as a generator: random coordinates in the low-dimensional space -> new 64x64 dogs",
                 fontsize=11, color=INK, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "pca_generator.png")


def main():
    X, y = load_flat()
    X_train, X_test, y_train, y_test = split(X, y, seed=SEED)
    plot_examples(X_train, y_train, np.random.default_rng(0))

    # Les images synthétiques aident-elles le SVM ? On les ajoute au train (jamais au test).
    configs = {
        "RBF par défaut (n=5, C=1)": lambda: build_pipeline(5, classifier=SVC(C=1, gamma="scale")),
        "RBF réglé (n=10, C=10, gamma=0.01)": lambda: build_pipeline(10, classifier=SVC(C=10, gamma=0.01)),
    }
    print(f"Ajout de {N_SYNTHETIC} images synthétiques par race au train (erreur test en %, 10 splits)")
    for name, build in configs.items():
        base, augmented = [], []
        for seed in range(10):
            a, b, c, d = split(X, y, seed=seed)
            fake_X, fake_y = generate(a, c, N_SYNTHETIC, np.random.default_rng(seed))
            base.append(100 * hamming_loss(d, build().fit(a, c).predict(b)))
            model = build().fit(np.vstack([a, fake_X]), np.concatenate([c, fake_y]))
            augmented.append(100 * hamming_loss(d, model.predict(b)))
        print(f"  {name:36s} : sans ajout {np.mean(base):.1f} (+/- {np.std(base):.1f}) | "
              f"avec ajout {np.mean(augmented):.1f} (+/- {np.std(augmented):.1f})")


if __name__ == "__main__":
    main()

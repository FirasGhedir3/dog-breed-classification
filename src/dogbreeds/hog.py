"""Gradients de Sobel et histogrammes de gradients orientés (HOG), écrits à la main."""
import numpy as np
from scipy.ndimage import correlate

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T


def sobel(img):
    """Gradients d'une image en niveaux de gris : gx, gy, magnitude et angle dans [0, pi).

    L'angle est non signé : un contour sombre vers clair et un contour clair vers sombre
    comptent comme la même orientation, ce qui rend le descripteur moins sensible à l'éclairage.
    """
    img = np.asarray(img, dtype=float)
    gx = correlate(img, SOBEL_X, mode="nearest")
    gy = correlate(img, SOBEL_Y, mode="nearest")
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi
    return gx, gy, magnitude, angle


def compute_hog(img, cell=16, n_bins=8):
    """HOG avec des boucles explicites. Chaque pixel vote pour la classe (bin) de son angle, pondéré par sa magnitude.

    Renvoie un vecteur de (H/cell) * (W/cell) * n_bins valeurs.
    """
    _, _, magnitude, angle = sobel(img)
    n_rows, n_cols = img.shape[0] // cell, img.shape[1] // cell
    bin_width = np.pi / n_bins
    hog = np.zeros((n_rows, n_cols, n_bins))

    for i in range(n_rows):
        for j in range(n_cols):
            for y in range(i * cell, (i + 1) * cell):
                for x in range(j * cell, (j + 1) * cell):
                    b = int(angle[y, x] // bin_width)
                    # Un angle négatif minuscule devient exactement pi après le modulo,
                    # ce qui donnerait un indice juste après le dernier bin.
                    b = min(b, n_bins - 1)
                    hog[i, j, b] += magnitude[y, x]
    return hog.ravel()


def compute_hog_fast(img, cell=16, n_bins=8):
    """Version vectorisée de compute_hog (même résultat, utilisée dans la pipeline)."""
    _, _, magnitude, angle = sobel(img)
    height, width = img.shape
    n_rows, n_cols = height // cell, width // cell
    bins = np.minimum((angle // (np.pi / n_bins)).astype(int), n_bins - 1)
    cell_id = (np.arange(height)[:, None] // cell) * n_cols + (np.arange(width)[None, :] // cell)
    flat_index = (cell_id * n_bins + bins).ravel()
    return np.bincount(flat_index, weights=magnitude.ravel(), minlength=n_rows * n_cols * n_bins)

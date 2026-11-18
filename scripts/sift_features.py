"""Features SIFT denses + SVM, à comparer à PCA + HOG.

SIFT détecte d'habitude des points d'intérêt, mais sur des images 64x64 il en trouve trop peu et
leur nombre varie d'une image à l'autre. On place donc les points sur une grille régulière
(SIFT dense) : chaque image donne toujours le même nombre de descripteurs de 128 valeurs,
concaténés en un seul vecteur. Aucun apprentissage : pas de fuite entre train et test.
"""
import cv2
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dogbreeds.data import SEED, load_processed, split

UPSCALE = 128  # SIFT a besoin d'images plus grandes que 64x64 pour des points de taille raisonnable
GRID = 4  # grille GRID x GRID de points
KEYPOINT_SIZE = 32.0


def dense_sift(images):
    sift = cv2.SIFT_create()
    step = UPSCALE / GRID
    keypoints = [cv2.KeyPoint(step * (j + 0.5), step * (i + 0.5), KEYPOINT_SIZE)
                 for i in range(GRID) for j in range(GRID)]
    features = []
    for img in images:
        big = cv2.resize(img, (UPSCALE, UPSCALE), interpolation=cv2.INTER_CUBIC)
        _, descriptors = sift.compute(big, keypoints)
        features.append(descriptors.ravel())
    return np.array(features)


def main():
    X_img, y = load_processed()
    F = dense_sift(X_img)
    print(f"SIFT dense : {GRID * GRID} points x 128 valeurs = {F.shape[1]} features par image")

    # Split 42, SVM RBF réglé par validation croisée dans le train
    F_train, F_test, y_train, y_test = split(F, y, seed=SEED)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    grid = {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.0005, 0.0001]}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy").fit(F_train, y_train)
    print(f"meilleurs paramètres : {search.best_params_} (erreur CV {100 * (1 - search.best_score_):.1f} %)")
    print(f"split {SEED} : erreur train {100 * hamming_loss(y_train, search.predict(F_train)):.1f} %, "
          f"erreur test {100 * hamming_loss(y_test, search.predict(F_test)):.1f} %")

    # Robustesse : SVM RBF par défaut sur 10 splits
    errors = []
    for seed in range(10):
        a, b, c, d = split(F, y, seed=seed)
        model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")).fit(a, c)
        errors.append(100 * hamming_loss(d, model.predict(b)))
    print(f"SVM RBF par défaut, 10 splits : erreur test {np.mean(errors):.1f} % (+/- {np.std(errors):.1f})")


if __name__ == "__main__":
    main()

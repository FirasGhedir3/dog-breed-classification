"""Transfer learning : features d'un VGG16 pré-entraîné sur ImageNet + SVM.

On retire la tête de classification de VGG16 et on garde la sortie de la dernière couche de
pooling (512 x 7 x 7 = 25088 valeurs par image). On entraîne ensuite le même SVM RBF que pour
les features faites à la main. VGG16 n'est pas ré-entraîné : il sert seulement d'extracteur.

Prérequis : pip install torch torchvision
Au premier lancement, torchvision télécharge les poids ImageNet (environ 530 Mo).
Les features sont sauvegardées dans data/processed/vgg16_features.npz pour ne pas les recalculer.
"""
import sys

import numpy as np
import torch
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models

from dogbreeds.data import PROCESSED_FILE, SEED, load_processed, split

FEATURES_FILE = PROCESSED_FILE.parent / "vgg16_features.npz"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # normalisation ImageNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_vgg16_features(images, batch_size=32, pretrained=True):
    """Images 64x64 en niveaux de gris -> vecteurs de 25088 valeurs (sortie du dernier pooling)."""
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    extractor = models.vgg16(weights=weights).features.eval()  # couches convolutives + pooling, sans la tête
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).float().div(255.0).unsqueeze(1)
            batch = torch.nn.functional.interpolate(batch, size=224, mode="bilinear", align_corners=False)
            batch = batch.repeat(1, 3, 1, 1)  # VGG16 attend 3 canaux : on recopie le gris
            batch = (batch - MEAN) / STD
            outputs.append(extractor(batch).flatten(start_dim=1).numpy())
            print(f"  {min(start + batch_size, len(images))} / {len(images)} images", end="\r")
    print()
    return np.vstack(outputs)


def main():
    X_img, y = load_processed()
    if FEATURES_FILE.exists():
        F = np.load(FEATURES_FILE)["F"]
        print(f"Features chargées depuis {FEATURES_FILE}")
    else:
        print("Extraction des features VGG16 (quelques minutes sur CPU)...")
        F = extract_vgg16_features(X_img)
        np.savez_compressed(FEATURES_FILE, F=F)
    print(f"VGG16 : {F.shape[1]} features par image")

    F_train, F_test, y_train, y_test = split(F, y, seed=SEED)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    grid = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 1e-5, 1e-6]}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy").fit(F_train, y_train)
    print(f"meilleurs paramètres : {search.best_params_} (erreur CV {100 * (1 - search.best_score_):.1f} %)")
    print(f"split {SEED} : erreur train {100 * hamming_loss(y_train, search.predict(F_train)):.1f} %, "
          f"erreur test {100 * hamming_loss(y_test, search.predict(F_test)):.1f} %")

    errors = []
    for seed in range(10):
        a, b, c, d = split(F, y, seed=seed)
        model = make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")).fit(a, c)
        errors.append(100 * hamming_loss(d, model.predict(b)))
    print(f"SVM RBF par défaut, 10 splits : erreur test {np.mean(errors):.1f} % (+/- {np.std(errors):.1f})")


if __name__ == "__main__":
    if "--help" in sys.argv:
        print(__doc__)
    else:
        main()

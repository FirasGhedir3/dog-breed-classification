"""Recadre chaque chien, convertit en niveaux de gris, applique le letterbox en 64x64 et sauvegarde le résultat.

À lancer une fois avant tous les autres scripts.
"""
import numpy as np

from dogbreeds.data import PROCESSED_FILE
from dogbreeds.preprocessing import build_dataset


def main():
    X, y = build_dataset()
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_FILE, X=X, y=y)
    print(f"{len(X)} images de {X.shape[1]}x{X.shape[2]} enregistrées dans {PROCESSED_FILE}")


if __name__ == "__main__":
    main()

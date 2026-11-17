"""Chemins des données, lecture des annotations et découpage train/test utilisé partout."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "SmallDB"
PROCESSED_FILE = ROOT / "data" / "processed" / "smalldb_64.npz"
FIGURES_DIR = ROOT / "results" / "figures"

SEED = 42
TEST_SIZE = 0.25


def breed_name(folder_name):
    """'n02085620-Chihuahua' -> 'Chihuahua'"""
    return folder_name.split("-", 1)[1].replace("_", " ")


def read_boxes(xml_path):
    """Toutes les boîtes englobantes (xmin, ymin, xmax, ymax) d'un fichier d'annotation."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        box = obj.find("bndbox")
        boxes.append(tuple(int(float(box.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")))
    return boxes


def list_samples(raw_dir=RAW_DIR):
    """Chemins des images, chemins des annotations et races, dans un ordre fixe pour que les exécutions soient reproductibles."""
    image_paths, annotation_paths, labels = [], [], []
    for folder in sorted(p.name for p in (Path(raw_dir) / "Images").iterdir()):
        for image in sorted((Path(raw_dir) / "Images" / folder).iterdir()):
            image_paths.append(image)
            annotation_paths.append(Path(raw_dir) / "Annotation" / folder / image.stem)
            labels.append(breed_name(folder))
    return image_paths, annotation_paths, np.array(labels)


def load_processed(path=PROCESSED_FILE):
    """Images prétraitées X de forme (N, 64, 64) et races y."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} introuvable. Lancez d'abord `python scripts/prepare_data.py`.")
    data = np.load(path)
    return data["X"], data["y"]


def load_flat(path=PROCESSED_FILE):
    """Comme load_processed, mais les images sont aplaties en vecteurs de 4096 flottants."""
    X, y = load_processed(path)
    return X.reshape(len(X), -1).astype(float), y


def split(X, y, seed=SEED):
    """Découpage train/test 75/25 stratifié."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)

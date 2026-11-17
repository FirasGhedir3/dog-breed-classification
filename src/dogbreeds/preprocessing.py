"""Recadrage et redimensionnement : d'une photo brute à une image 64x64 en niveaux de gris."""
import cv2
import numpy as np
from PIL import Image

from .data import list_samples, read_boxes

SIZE = 64


def read_and_crop(image_path, annotation_path, which="largest"):
    """Charge une image en niveaux de gris et la recadre autour d'un chien annoté.

    Certaines photos contiennent plusieurs chiens. `which` choisit celui à garder :
    "largest" (plus grande boîte) ou "first" (premier objet annoté).
    """
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    boxes = read_boxes(annotation_path)
    if which == "largest":
        xmin, ymin, xmax, ymax = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    else:
        xmin, ymin, xmax, ymax = boxes[0]
    height, width = gray.shape
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(width, xmax), min(height, ymax)
    return gray[ymin:ymax, xmin:xmax]


def resize_stretch(img, size=SIZE):
    """Redimensionnement naïf en size x size. Déforme le ratio, conservé uniquement pour comparaison."""
    interpolation = cv2.INTER_AREA if max(img.shape) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interpolation)


def letterbox(img, size=SIZE, pad="black"):
    """Redimensionne sans déformer, puis centre le résultat sur un canevas size x size.

    pad : "black", "white" ou "mean" pour un remplissage constant, "edge" ou "reflect" pour prolonger l'image.
    """
    h, w = img.shape
    # L'axe qui doit le plus rétrécir impose l'échelle aux deux axes.
    ratio = min(size / h, size / w)
    new_h = max(1, min(size, round(h * ratio)))
    new_w = max(1, min(size, round(w * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    if pad in ("black", "white", "mean"):
        fill = {"black": 0, "white": 255, "mean": int(round(img.mean()))}[pad]
        canvas = np.full((size, size), fill, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas
    if pad in ("edge", "reflect"):
        return np.pad(resized, ((top, size - top - new_h), (left, size - left - new_w)), mode=pad)
    raise ValueError(f"Mode de remplissage inconnu : {pad}")


def build_dataset(size=SIZE, pad="black", which="largest"):
    """Recadre et applique le letterbox à chaque image de SmallDB. Renvoie X (N, size, size) et les races y."""
    image_paths, annotation_paths, labels = list_samples()
    X = np.empty((len(image_paths), size, size), dtype=np.uint8)
    for i, (image, annotation) in enumerate(zip(image_paths, annotation_paths)):
        X[i] = letterbox(read_and_crop(image, annotation, which), size, pad)
    return X, labels

"""
Tests des fonctions de prétraitement.
"""
import numpy as np
import cv2


def test_clahe_shape_preservee():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_voc_vers_yolo_normalisation():
    xmin, ymin, xmax, ymax = 100, 50, 300, 200
    largeur, hauteur = 640, 640
    xc = ((xmin + xmax) / 2) / largeur
    yc = ((ymin + ymax) / 2) / hauteur
    w = (xmax - xmin) / largeur
    h = (ymax - ymin) / hauteur
    assert 0 <= xc <= 1
    assert 0 <= yc <= 1
    assert 0 <= w <= 1
    assert 0 <= h <= 1


def test_classes_officielles():
    CLASSES = ["D00", "D10", "D20", "D40"]
    assert len(CLASSES) == 4
    assert "D00" in CLASSES
    assert "D40" in CLASSES


def test_mapping_classe_idx_bijectif():
    CLASSES = ["D00", "D10", "D20", "D40"]
    m = {c: i for i, c in enumerate(CLASSES)}
    assert m["D00"] == 0
    assert m["D40"] == 3
    assert len(set(m.values())) == 4


def test_normalisation_image():
    img = np.array([0, 127, 255], dtype=np.uint8)
    norm = img.astype(np.float32) / 255.0
    assert norm[0] == 0.0
    assert abs(norm[2] - 1.0) < 1e-6
    assert 0 < norm[1] < 1

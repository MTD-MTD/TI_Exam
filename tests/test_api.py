"""
Tests unitaires de l'API FastAPI.
À exécuter avec pytest une fois l'API démarrée.
"""

import requests
import pytest
from pathlib import Path

BASE = "http://127.0.0.1:8000"


def test_racine():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "version" in data


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["modele_charge"] is True


def test_classes():
    r = requests.get(f"{BASE}/classes")
    assert r.status_code == 200
    data = r.json()
    assert set(data["classes"].values()) == {"D00", "D10", "D20", "D40"}


def test_predict_image_valide():
    # Remplacer par un chemin réel dans ton environnement
    image_dir = Path("/content/datasets/RDD2022_split/Japan/test/images")
    images = list(image_dir.glob("*.jpg"))
    if not images:
        pytest.skip("Pas d'image de test disponible.")

    with open(images[0], "rb") as f:
        r = requests.post(
            f"{BASE}/predict",
            files={"file": (images[0].name, f, "image/jpeg")},
        )
    assert r.status_code == 200
    data = r.json()
    assert "detections" in data
    assert "nb_detections" in data


def test_predict_fichier_invalide():
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.txt", b"ceci n est pas une image", "text/plain")},
    )
    assert r.status_code == 400

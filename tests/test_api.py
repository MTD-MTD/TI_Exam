"""
Tests unitaires de l'API FastAPI via TestClient.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient


with patch("api.main.charger_modele"):
    from api.main import app

client = TestClient(app)


def test_racine():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "service" in data
    assert "version" in data


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_classes():
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert set(data["classes"].values()) == {"D00", "D10", "D20", "D40"}


def test_predict_fichier_invalide():
    r = client.post("/predict",
                    files={"file": ("test.txt", b"pas une image", "text/plain")})
    assert r.status_code in (400, 503)


def test_predict_sans_fichier():
    r = client.post("/predict")
    assert r.status_code == 422

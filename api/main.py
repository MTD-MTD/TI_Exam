"""
API FastAPI pour la détection de fissures routières.
Modèle : YOLOv8s fine-tuné sur RDD2022 (7 pays).
"""

import os
from pathlib import Path
from typing import List
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MODELE_PATH = Path(os.environ.get("MODELE_PATH", "/app/models/best.pt")))
CLASSES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
IMGSZ = 640
CONF_THRESHOLD = 0.25

# --- Chargement du modèle au démarrage ---
app = FastAPI(
    title="Détection de fissures routières",
    description="API d'inférence YOLOv8s sur images de chaussée",
    version="1.0.0",
)

modele = None


@app.on_event("startup")
def charger_modele():
    """Charge le modèle YOLO en mémoire au démarrage."""
    global modele
    if not MODELE_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODELE_PATH}")
    modele = YOLO(str(MODELE_PATH))
    print(f"Modèle chargé : {MODELE_PATH.name}")


@app.get("/")
def racine():
    """Endpoint racine — vérification que l'API tourne."""
    return {
        "service": "RDD2022 Crack Detection API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/classes"],
    }


@app.get("/health")
def health():
    """Vérification de l'état de l'API et du modèle."""
    return {
        "status": "ok",
        "modele_charge": modele is not None,
        "chemin_modele": str(MODELE_PATH),
    }


@app.get("/classes")
def classes():
    """Liste des classes détectables."""
    return {"classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image, retourne les bounding boxes détectées.
    Format de sortie : liste de dicts {classe, confiance, bbox}.
    """
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    # --- Validation du fichier ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier non image.")

    # --- Chargement de l'image ---
    try:
        contenu = await file.read()
        image = Image.open(io.BytesIO(contenu)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image illisible : {e}")

    # --- Inférence ---
    resultats = modele.predict(
        source=np.array(image),
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    # --- Extraction des détections ---
    detections = []
    for r in resultats:
        for box in r.boxes:
            classe_id = int(box.cls[0])
            confiance = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "classe": CLASSES.get(classe_id, "inconnue"),
                "classe_id": classe_id,
                "confiance": round(confiance, 4),
                "bbox": {
                    "xmin": round(x1, 2),
                    "ymin": round(y1, 2),
                    "xmax": round(x2, 2),
                    "ymax": round(y2, 2),
                },
            })

    return {
        "fichier": file.filename,
        "dimensions": {"largeur": image.width, "hauteur": image.height},
        "nb_detections": len(detections),
        "detections": detections,
    }

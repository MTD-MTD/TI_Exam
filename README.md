# TI_Exam — Détection de fissures sur routes

Projet MLOps : détection de dommages routiers (RDD2022).

## Modèle

- YOLOv8s fine-tuné sur 7 pays (19 000 images d'entraînement)
- mAP@50 sur test : 0.622

## Structure

- `api/` : API FastAPI
- `tests/` : tests unitaires
- `models/best.pt.dvc` : modèle versionné par DVC
- `Dockerfile` : image Docker
- `.gitlab-ci.yml` : pipeline CI/CD GitLab
- `.github/workflows/` : CI GitHub Actions
- `EXAMEN_TI.ipynb` : notebook principal

## Installation

```bash
pip install -r api/requirements.txt
```

## Docker

```bash
docker build -t ti-exam:latest .
```

```bash
dvc pull models/best.pt.dvc
docker run -p 8000:8000 -v $(pwd)/models/best.pt:/app/models/best.pt ti-exam:latest
```

## Tester l'API

```bash
curl http://localhost:8000/health
curl -X POST -F file=@image.jpg http://localhost:8000/predict
```

## Résultats

- D00 (fissure longitudinale) : AP@50 = 0.582
- D10 (fissure transversale)  : AP@50 = 0.601
- D20 (alligator)             : AP@50 = 0.710
- D40 (nid-de-poule)          : AP@50 = 0.595

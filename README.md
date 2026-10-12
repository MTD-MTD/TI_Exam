# TI_Exam — Détection de fissures sur routes

Projet MLOps : détection de dommages routiers (RDD2022).

## Modèle
- YOLOv8s fine-tuné sur 7 pays (19 000 images d'entraînement)
- mAP@50 sur test : 0.622

## Structure
- Notebook principal : pipeline complet (EDA, split, entraînement, évaluation)

## Résultats
- D00 (fissure longitudinale) : AP@50 = 0.582
- D10 (fissure transversale)  : AP@50 = 0.601
- D20 (alligator)             : AP@50 = 0.710
- D40 (nid-de-poule)          : AP@50 = 0.595

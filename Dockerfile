FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour OpenCV et Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY api/ ./api/

# Le modèle sera monté au runtime via volume (voir README)
# Exemple : docker run -v /chemin/local/best.pt:/app/models/best.pt ...

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

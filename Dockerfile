FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8010 \
    MODEL_PATH=models/best_model.joblib \
    DATASET_PATH=dataset_com_id_gestante.csv \
    METRICS_PATH=data/metricas_modelos.csv \
    APP_NAME=CardioPrenatal-ML \
    APP_VERSION=1.0.0 \
    LOG_LEVEL=info

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY analise_descritiva ./analise_descritiva
COPY data ./data
COPY dataset_com_id_gestante.csv ./dataset_com_id_gestante.csv

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]

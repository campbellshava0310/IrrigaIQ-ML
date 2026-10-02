# IrrigaIQ ML service. The trained models ship in the image (models/); the embedding model is
# fetched at build time so the container needs no internet at run time.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    IRRIGAIQ_EMBEDDINGS_CACHE=/opt/irrigaiq/embeddings

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY irrigaiq_ml ./irrigaiq_ml
COPY models ./models
RUN python -c "from irrigaiq_ml.embeddings import embed; embed(['warm up'])"

RUN useradd --system --uid 10001 irrigaiq && chown -R irrigaiq /opt/irrigaiq
USER irrigaiq
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health')"
CMD ["uvicorn", "irrigaiq_ml.api:app", "--host", "0.0.0.0", "--port", "8090"]

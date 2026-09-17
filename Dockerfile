FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data \
    TOKENIZERS_PARALLELISM=false \
    API_VERSION=2026-03-20-full-app-py-v3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# Do not download Hugging Face models at build time — Render builds often cannot reach huggingface.co.
# Models (MiniLM, BART, CLIP) load lazily at runtime on first request.

COPY app.py runtime_data.py /app/
COPY project/ /app/project/
COPY deploy/data/ /app/data/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

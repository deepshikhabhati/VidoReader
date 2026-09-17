FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/deploy/data \
    TOKENIZERS_PARALLELISM=false \
    API_VERSION=2026-03-20-ask-ai

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# Bake the query encoder into the image so free-tier hosts can run offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY deploy/app.py /app/deploy/app.py
COPY project/ /app/project/
COPY english_chunks.json german_chunks.json comparison_results.json embedding_comparison_results.json /app/deploy/data/
COPY hamlet_translation_analysis/output/hamlet_4x4_comparison.json /app/deploy/data/hamlet_4x4_comparison.json

WORKDIR /app/deploy

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

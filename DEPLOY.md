# Deploy VidoReader API (full `app.py`)

The server runs the **repo root `app.py`** (all FastAPI routes), not `deploy/app.py`.

Data files live in `deploy/data/` and are mounted at `DATA_ROOT=/app/data` in Docker.

## Prepare data (local + before Docker build)

```bash
chmod +x deploy/prepare_data.sh
./deploy/prepare_data.sh
```

## Local run

```bash
export DATA_ROOT="./deploy/data"
export OPENAI_API_KEY="your_key"
pip install -r deploy/requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check http://localhost:8000/ — the `routes` list should include `/ask-ai`, `/find-similar-full-reduced`, `/search`, etc.

## Render

Repo: https://github.com/deepshikhabhati/VidoReader

After push, Render → **vidoreader** → **Manual Deploy** → **Clear build cache & deploy**.

Confirm https://vidoreader.onrender.com/ shows `"deployed_module": "app.py"` and `/ask-ai` in `routes`.

**Note:** Full `app.py` includes CLIP and large JSON corpora (~70MB data). Free Render may be slow to start or hit memory limits on `/search` (CLIP). Textbook + Hamlet search endpoints should work.

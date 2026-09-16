# Deploy Hamlet API (Free Tier)

This repo includes a **lightweight deploy app** at `deploy/app.py` that works on free hosts like **Render**.

It serves:

- `GET /` health check
- `GET /ui/comparison`
- `GET /hamlet/4x4-comparison`
- `POST /compare-embedding-results`
- `POST /compare-embedding-retrieval`
- `POST /compare-embedding-batch`

It does **not** load CLIP/BART at startup, so it fits much better on free tiers than the full `app.py`.

## 1. Prepare data

```bash
chmod +x deploy/prepare_data.sh
./deploy/prepare_data.sh
```

This copies the JSON files the API needs into `deploy/data/`.

## 2. Test locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r deploy/requirements.txt
export OPENAI_API_KEY="your_key"
export DATA_DIR="./deploy/data"
uvicorn app:app --app-dir deploy --host 0.0.0.0 --port 8000 --reload
```

Open:

- http://localhost:8000/
- http://localhost:8000/ui/comparison
- http://localhost:8000/hamlet/4x4-comparison

## 3. Deploy to Render (free)

Repo: https://github.com/deepshikhabhati/hamlet-translation-api

### One-click deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/deepshikhabhati/hamlet-translation-api)

Or open:

https://render.com/deploy?repo=https://github.com/deepshikhabhati/hamlet-translation-api

Steps:

1. Sign in to Render with GitHub.
2. Click **Apply** on the detected `render.yaml` blueprint.
3. Add environment variable:
   - `OPENAI_API_KEY` = your OpenAI key
4. Wait for the Docker build to finish.

Render builds from the root `Dockerfile`.

### CLI deploy (after `render login`)

```bash
chmod +x deploy/render_deploy.sh
./deploy/render_deploy.sh
```

## 4. Deploy with Docker anywhere

```bash
./deploy/prepare_data.sh
docker build -t hamlet-api .
docker run -p 8000:8000 -e OPENAI_API_KEY="your_key" hamlet-api
```

## Notes

- Free Render instances sleep after inactivity; first request may take ~30s.
- `/compare-embedding-batch` can take a long time and may hit free-tier timeouts.
- The full heavy `app.py` (CLIP, BART, many endpoints) is **not** included in this deploy image.
- For the full app, use an Oracle Cloud free VM instead.

## Security

Never commit your OpenAI key. Set it only as a server environment variable.

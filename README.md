# Medical-Chatbot

Minimal **FastAPI** service for a medical information chatbot.

## Endpoints

- `GET /` – service info
- `GET /health` – Render health check
- `POST /chat` – JSON `{ "message": "..." }`
- `POST /get` – legacy form-style `msg` (plain-text response)

## Local run (Windows-friendly)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Deploy on Render (fixes 502)

This repo includes `render.yaml` + `Procfile` so the service **binds to `$PORT`** using Gunicorn + Uvicorn worker.

If you configure it manually in Render, use:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

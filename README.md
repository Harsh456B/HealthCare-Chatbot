# Medical-Chatbot

Minimal API for a medical information chatbot (Groq-backed).

## Local run

1. Create `.env` (copy from `.env.example`) and set `GROQ_API_KEY`.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Start the server:

```bash
uvicorn main:app --reload
```

4. Test:

- `GET /health`
- `POST /chat` with JSON: `{ "message": "..." }`

## Deployment (Render)

Create a Render **Web Service** pointing to this repo and use:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment**: set `GROQ_API_KEY` (and optionally `GROQ_MODEL`)

If you see a `502 Bad Gateway`, it usually means your service isn’t listening on `$PORT` or it crashed on startup.

## Render deploy (fixes 502)

This repo includes a small FastAPI service that **binds to Render’s `$PORT`** and exposes:

- `GET /health` for Render health checks
- `POST /chat` for chatbot messages

### Local run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Deploy on Render

- Connect the repo on Render as a **Web Service**
- It will use `render.yaml` automatically (or set the start command to the same value)
# Medical-Chatbot

Flask-based medical chatbot with:
- Web chat UI at `/`
- Health endpoint at `/health`
- Chat endpoint at `/get` (form `msg`)
- Background RAG initialization with fallback LLM mode

## Local run

1. Create `.env` from `.env.example` and set:
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

## Deploy on Render (recommended)

This repo is now deployment-ready with `render.yaml`.

If configuring manually in Render:
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --preload --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 300 --graceful-timeout 120 app:app`
- **Health Check Path**: `/health`

## Note on 502/HTML in chat

The chat UI now suppresses proxy HTML error pages (like Render 502 pages) and shows a friendly fallback message instead.

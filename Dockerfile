FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system build deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]

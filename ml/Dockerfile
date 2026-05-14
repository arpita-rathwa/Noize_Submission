# ============================================================
# NOIZE — Dockerfile
# Build : docker build -t noize-api .
# Run   : docker run -p 8000:8000 noize-api
# ============================================================

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-create data directory
RUN mkdir -p /app/data /app/models /app/reports

# Expose API port
EXPOSE 8000

# Start
# FIX 3: Gunicorn with UvicornWorker handles concurrent requests properly.
# 2 workers = safe for 512MB RAM. Increase to 4 for 1GB+ containers.
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

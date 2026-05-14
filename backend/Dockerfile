# ============================================================
# NOIZE Backend — Dockerfile
# ============================================================

FROM python:3.11-slim

# Don't write .pyc files; don't buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Run as non-root for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# FIX 3: Gunicorn with UvicornWorker handles concurrent requests properly.
# 2 workers = safe for 512MB RAM. Increase to 4 for 1GB+ containers.
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8001", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

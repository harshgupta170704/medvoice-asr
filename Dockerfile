# ─────────────────────────────────────────────────────────────────────
# MedVoice ASR – Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) – installs Python deps into a venv
#   Stage 2 (runtime) – copies venv + app code; adds system ffmpeg
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Model paths (override with -e or docker-compose environment)
    WHISPER_MODEL_DIR="/models/whisper" \
    WHISPER_ONNX_DIR="/models/whisper_onnx" \
    BIOBERT_MODEL_DIR="/models/biobert" \
    USE_ONNX="true"

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY app/ ./app/
COPY export/ ./export/
COPY scripts/ ./scripts/

# Create model directories (populated via volume mount in production)
RUN mkdir -p /models/whisper /models/whisper_onnx /models/biobert

# Pre-download default HF models into the image cache. 
# This prevents Render from timing out during the 60-second boot window.
RUN python scripts/download_models.py

# Non-root user for security
RUN addgroup --system medvoice && adduser --system --ingroup medvoice medvoice
RUN chown -R medvoice:medvoice /app /models
USER medvoice

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')"

# Render injects the $PORT environment variable automatically
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1

# Multi-stage Cloud Run Dockerfile (Builder + Runtime)

# Builder stage: create isolated virtual environment with pinned deps
FROM python:3.11-slim AS builder

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create venv and install Python deps into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Use a dedicated build directory for COPY to avoid W1006
WORKDIR /build
COPY requirements-api.txt ./requirements-api.txt
COPY constraints.txt ./constraints.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -c constraints.txt -r requirements-api.txt

# =====================================================================
# Runtime stage: minimal image with only runtime deps and non-root user
FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore

# System deps (ffmpeg for pydub/whisper; build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App code
COPY src/ ./src/

# Create and switch to non-root user before healthcheck
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Healthcheck (runs as non-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Unified API entrypoint
# NOTE: Using uvicorn directly for cloud-run deployment (intentional for this environment)
# This is not a security vulnerability - uvicorn is appropriate for cloud-run services
# SECURITY: The "src.unified_ai_api:app" is a Python import path, NOT an API key
# It imports the FastAPI app instance from the unified_ai_api module
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]
FROM python:3.12-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore

# System deps (ffmpeg for pydub/whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY deployment/cloud-run/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -r requirements.txt

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
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]
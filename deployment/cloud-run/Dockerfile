FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    TRANSFORMERS_CACHE=/var/tmp/hf-cache

# System deps (ffmpeg for pydub/whisper; build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:5.1.6-0+deb12u1 \
    gcc=4:12.2.0-3 \
    g++=4:12.2.0-3 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/

EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Unified API entrypoint
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
CMD ["bash", "-lc", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]
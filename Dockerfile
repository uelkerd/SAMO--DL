FROM python:3.12-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore \
    EMOTION_MODEL_LOCAL_DIR=/app/model

# System deps needed for audio runtime only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (minimal unified runtime)
COPY deployment/cloud-run/requirements_unified.txt ./requirements_unified.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -r requirements_unified.txt

# Pre-bundle models to reduce cold-start; combine to minimize layers (DOK-W1001)
RUN python -c "from transformers import AutoTokenizer, T5ForConditionalGeneration; AutoTokenizer.from_pretrained('t5-small'); T5ForConditionalGeneration.from_pretrained('t5-small'); AutoTokenizer.from_pretrained('t5-base'); T5ForConditionalGeneration.from_pretrained('t5-base'); print('Pre-bundled t5-small and t5-base into cache')" \
 && python -c "import whisper; whisper.load_model('small'); print('Pre-bundled whisper-small into cache')"

# Bake emotion model into the image at /app/model (public HF repo by default)
ARG EMOTION_MODEL_ID=0xmnrv/samo
ARG HF_TOKEN=""
COPY scripts/deployment/bake_emotion_model.py /app/bake_emotion_model.py
RUN EMOTION_MODEL_ID=${EMOTION_MODEL_ID} HF_TOKEN=${HF_TOKEN} python /app/bake_emotion_model.py

# Copy source
COPY src/ ./src/

# Switch to non-root user before runtime directives
RUN useradd -m -u 1000 appuser && mkdir -p /var/tmp/hf-cache && chown -R appuser:appuser /app /var/tmp/hf-cache
USER appuser

# Healthcheck (runs as non-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

EXPOSE 8080

# Unified API entrypoint
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]


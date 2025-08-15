# SECURE DOCKERFILE - Addresses Trivy vulnerabilities while maintaining functionality
FROM python:3.12-slim-bookworm

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore \
    EMOTION_MODEL_LOCAL_DIR=/app/model

# SECURITY: Update packages and fix vulnerabilities found by Trivy
# SECURITY: Pin versions to avoid DOK-DL3008 and ensure reproducible builds
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # SECURITY: Pin FFmpeg to fix CVE-2023-6603, CVE-2025-1594
    ffmpeg=7:5.1.6-0+deb12u1 \
    # SECURITY: Pin libaom3 to fix CVE-2023-6879
    libaom3=3.6.0-1+deb12u1 \
    # SECURITY: Pin libavcodec/libavformat to fix vulnerabilities
    libavcodec-extra=7:5.1.6-0+deb12u1 \
    libavformat-extra=7:5.1.6-0+deb12u1 \
    # SECURITY: Pin curl to fix vulnerabilities
    curl=7.88.1-10+deb12u12 \
    # SECURITY: Build tools (will be removed after use)
    build-essential=12.9 \
    gcc=4:12.2.0-3 \
    g++=4:12.2.0-3 \
    cmake=3.25.1-1 \
    pkgconf=1.8.1-1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (minimal unified runtime)
COPY deployment/cloud-run/requirements_unified.txt ./requirements_unified.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -r requirements_unified.txt

# SECURITY: Remove build tools after compilation to reduce attack surface
RUN apt-get purge -y build-essential gcc g++ cmake pkgconf \
    && apt-get autoremove -y \
    && apt-get clean

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

# SECURITY: Create and switch to non-root user for runtime
RUN useradd -m -u 1000 appuser \
    && mkdir -p /var/tmp/hf-cache \
    && chown -R appuser:appuser /app /var/tmp/hf-cache
USER appuser

# Healthcheck (runs as non-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

# Unified API entrypoint
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host ${HOST:-0.0.0.0} --port ${PORT}"]


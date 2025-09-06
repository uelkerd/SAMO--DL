# Consolidated Cloud Run Dockerfile
# Build different variants using build arguments:
# --build-arg BUILD_TYPE=minimal|unified|secure|production
# --build-arg INCLUDE_ML=true|false
# --build-arg INCLUDE_SECURITY=true|false

# Builder stage: create isolated virtual environment with pinned deps
FROM python:3.11-slim-bookworm AS builder

# Declare build arguments in this stage
ARG BUILD_TYPE=minimal
ARG INCLUDE_ML=false
ARG INCLUDE_SECURITY=false

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build tools only when ML dependencies are needed
RUN if [ "$INCLUDE_ML" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Create venv and install Python deps into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Use a dedicated build directory for COPY to avoid W1006
WORKDIR /build

# Copy all requirements files and constraints from dependencies/
COPY dependencies/requirements_*.txt ./dependencies/
COPY dependencies/constraints.txt ./dependencies/

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && cp dependencies/requirements_${BUILD_TYPE}.txt requirements.txt \
 && pip install --no-cache-dir -c dependencies/constraints.txt -r requirements.txt

# Create a simple minimal API server (always created for runtime stage compatibility)
RUN echo '#!/usr/bin/env python3' > ./minimal_api_server.py && \
    echo 'from flask import Flask, jsonify' >> ./minimal_api_server.py && \
    echo 'app = Flask(__name__)' >> ./minimal_api_server.py && \
    echo '@app.route("/health")' >> ./minimal_api_server.py && \
    echo 'def health():' >> ./minimal_api_server.py && \
    echo '    return jsonify({"status": "healthy", "variant": "minimal"})' >> ./minimal_api_server.py && \
    echo 'if __name__ == "__main__":' >> ./minimal_api_server.py && \
    echo '    app.run(host="0.0.0.0", port=8080)' >> ./minimal_api_server.py

# =====================================================================
# Runtime stage: minimal image with only runtime deps and non-root user
FROM python:3.11-slim-bookworm

# Declare build arguments again in runtime stage
ARG BUILD_TYPE=minimal
ARG INCLUDE_ML=false
ARG INCLUDE_SECURITY=false
ARG PIP_CONSTRAINT=""
ARG EMOTION_MODEL_DIR_ARG="/models/emotion-english-distilroberta-base"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore \
    EMOTION_PROVIDER=hf \
    EMOTION_LOCAL_ONLY=1 \
    EMOTION_MODEL_DIR=${EMOTION_MODEL_DIR_ARG}

# Install system deps based on build type
RUN if [ "$INCLUDE_ML" = "true" ]; then \
        # ML version needs ffmpeg for audio processing
        apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl=7.88.1-10+deb12u12 \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        # Minimal version only needs curl for health checks
        apt-get update && apt-get install -y --no-install-recommends \
        curl=7.88.1-10+deb12u12 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Bring in Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-bundle ML models for unified build type
RUN if [ "$BUILD_TYPE" = "unified" ] && [ "$INCLUDE_ML" = "true" ]; then \
        # Pre-bundle summarization and ASR models into cache to avoid cold downloads
        python -c "from transformers import AutoTokenizer, T5ForConditionalGeneration; AutoTokenizer.from_pretrained('t5-small'); T5ForConditionalGeneration.from_pretrained('t5-small'); AutoTokenizer.from_pretrained('t5-base'); T5ForConditionalGeneration.from_pretrained('t5-base'); print('Pre-bundled t5-small and t5-base into cache')" \
        && python -c "import whisper; whisper.load_model('small'); print('Pre-bundled whisper-small into cache')"; \
    fi

# App code
COPY src/ ./src/

# Copy additional files from builder stage for specific build types
COPY --from=builder /build/minimal_api_server.py ./minimal_api_server.py

# Download emotion model during build (if ML is enabled)
RUN if [ "$INCLUDE_ML" = "true" ]; then \
        echo "📥 Downloading emotion model for offline operation..." && \
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='j-hartmann/emotion-english-distilroberta-base', local_dir='${EMOTION_MODEL_DIR_ARG}', local_dir_use_symlinks=False)" && \
        echo "✅ Emotion model downloaded successfully"; \
    fi

# Create and configure user based on build type
RUN if [ "$INCLUDE_SECURITY" = "true" ]; then \
        # Secure version with enhanced security
        useradd -m -u 1000 appuser \
        && mkdir -p /var/tmp/hf-cache \
        && chown -R appuser:appuser /app /var/tmp/hf-cache \
        && chmod 755 /app /var/tmp/hf-cache; \
    else \
        # Standard version
        useradd -m -u 1000 appuser \
        && mkdir -p /var/tmp/hf-cache /app/models \
        && chown -R appuser:appuser /app /var/tmp/hf-cache /app/models; \
    fi

USER appuser

EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

# Unified API entrypoint (non-root)
# NOTE: Using uvicorn directly for cloud-run deployment (intentional for this environment)
# This is not a security vulnerability - uvicorn is appropriate for cloud-run services
# SECURITY: The "src.unified_ai_api:app" is a Python import path, NOT an API key
# It imports the FastAPI app instance from the unified_ai_api module

# Create entrypoint script based on build type
RUN if [ "$BUILD_TYPE" = "minimal" ]; then \
        echo '#!/bin/sh\nexec gunicorn -b 0.0.0.0:${PORT:-8080} minimal_api_server:app' > /app/entrypoint.sh; \
    elif [ "$BUILD_TYPE" = "secure" ]; then \
        echo '#!/bin/sh\nexec uvicorn src.secure_api_server:app --host 0.0.0.0 --port ${PORT:-8080}' > /app/entrypoint.sh; \
    else \
        echo '#!/bin/sh\nexec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT:-8080}' > /app/entrypoint.sh; \
    fi && \
    chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

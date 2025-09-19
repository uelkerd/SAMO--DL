# Optimized CPU-only Dockerfile for Cloud Run deployment
# Minimal dependencies, no GPU packages, smaller image size
FROM python:3.10-slim-bookworm

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

# Set working directory
WORKDIR /app

# Install minimal system dependencies including audio processing
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20230311 \
    curl=7.88.1-10+deb12u6 \
    ffmpeg=7:5.1.2-7+deb12u1 \
    libsndfile1=1.2.0-3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy optimized requirements first for better caching
COPY deployment/docker/requirements-api-optimized.txt ./requirements.txt

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir -r requirements.txt

# Copy the unified SAMO API with voice processing and all dependencies
COPY src/ ./src/
COPY scripts/ ./scripts/

# Pre-download the SAMO models and Whisper during build to avoid OOM during startup
RUN mkdir -p /app/models && \
    python scripts/pre_download_models.py

# Create non-root user for security (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Cloud Run requirement)
EXPOSE 8080

# Health check following Cloud Run best practices
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use exec form for CMD (Docker best practice)
# Run the unified SAMO API with FastAPI/Uvicorn
CMD ["sh", "-c", "exec python -m uvicorn src.unified_ai_api:app --host 0.0.0.0 --port $PORT --workers 1"]
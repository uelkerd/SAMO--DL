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

# Install minimal system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20230311 \
    curl=7.88.1-10+deb12u5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy optimized requirements first for better caching
COPY deployment/docker/requirements-api-optimized.txt ./requirements.txt

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual production code from the PRs
COPY deployment/cloud-run/secure_api_server.py .
COPY deployment/cloud-run/model_utils.py .
COPY deployment/cloud-run/security_headers.py .
COPY deployment/cloud-run/rate_limiter.py .

# Pre-download the model during build to avoid OOM during startup
RUN mkdir -p /app/models && \
    python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    model_name='j-hartmann/emotion-english-distilroberta-base'; \
    print(f'Pre-downloading model {model_name}...'); \
    AutoTokenizer.from_pretrained(model_name, cache_dir='/app/models'); \
    AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/app/models'); \
    print('Model pre-downloaded successfully');"

# Create non-root user for security (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Cloud Run requirement)
EXPOSE 8080

# Health check following Cloud Run best practices
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Use exec form for CMD (Docker best practice)
# Set timeout to 0 for Cloud Run (allows unlimited request timeouts)
# Run the production Flask-RESTX server
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 --keep-alive 5 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - --log-level info secure_api_server:app"]

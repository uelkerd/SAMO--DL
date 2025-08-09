FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

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

# Use official Python runtime with explicit platform targeting
FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY robust_predict.py .
COPY model/ ./model/

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
# Set timeout to 0 for Cloud Run (allows unlimited request timeouts)
CMD exec gunicorn \
    --bind :$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    robust_predict:app 
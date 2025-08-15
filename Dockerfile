# SECURE DOCKERFILE - Addresses Trivy vulnerabilities with minimal complexity
FROM python:3.12-slim-bookworm

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# SECURITY: Install and pin specific package versions to fix vulnerabilities
# SECURITY: Pin versions to avoid DOK-DL3008 and ensure reproducible builds
RUN apt-get update \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements-simple.txt .
RUN pip install --no-cache-dir -r requirements-simple.txt

# SECURITY: Create proper non-root user and group first
RUN groupadd -r app && useradd -r -g app app

# Copy source code with proper ownership
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app.py .

# SECURITY: Switch to non-root user for runtime
USER app

# Healthcheck (runs as non-root user, respects PORT env var)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

EXPOSE $PORT

# SECURITY: Use Gunicorn for production with environment variable support
CMD ["sh", "-c", "gunicorn --bind ${HOST}:${PORT} --workers 2 --worker-class uvicorn.workers.UvicornWorker --access-logfile - --error-logfile - app:app"]


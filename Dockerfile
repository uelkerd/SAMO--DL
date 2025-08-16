# SECURE MULTI-STAGE DOCKERFILE - Addresses Trivy vulnerabilities with minimal complexity
# Pin base image to immutable digest for reproducible builds
# TODO: Update this digest to the current version before merging
# Get current digest: docker pull python:3.12-slim-bookworm && docker images --digests | grep python:3.12-slim-bookworm
# Builder stage: create isolated virtual environment with dependencies
FROM python:3.12-slim-bookworm@sha256:placeholder-update-before-merge AS builder

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create virtual environment and install Python deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Runtime stage: minimal image with only runtime deps
FROM python:3.12-slim-bookworm@sha256:placeholder-update-before-merge

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# SECURITY: Install and pin specific package versions to fix vulnerabilities
# SECURITY: Pin versions to avoid DOK-DL3008 and ensure reproducible builds
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # SECURITY: Pin FFmpeg to a known secure version on Debian bookworm
    ffmpeg=7:5.1.6-0+deb12u1 \
    # SECURITY: Pin curl to fix vulnerabilities
    curl=7.88.1-10+deb12u12 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in Python environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# SECURITY: Create proper non-root user and group first
RUN groupadd -r app && useradd -r -g app app

# Copy source code with proper ownership
COPY --chown=app:app src/ ./src/

# SECURITY: Switch to non-root user for runtime
USER app

# Healthcheck (runs as non-root user, respects PORT env var)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# EXPOSE with concrete port value (Docker doesn't expand env vars in EXPOSE)
EXPOSE 8000

# SECURITY: Use Gunicorn for production with environment variable support
CMD ["sh", "-c", "gunicorn --bind ${HOST}:${PORT} --workers 2 --worker-class uvicorn.workers.UvicornWorker --access-logfile - --error-logfile - src.unified_ai_api:app"]


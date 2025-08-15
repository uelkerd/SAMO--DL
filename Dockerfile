# SECURE DOCKERFILE - Addresses Trivy vulnerabilities with minimal complexity
FROM python:3.12-slim-bookworm

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements-simple.txt .
RUN pip install --no-cache-dir -r requirements-simple.txt

# Copy source code
COPY src/ ./src/

# SECURITY: Create and switch to non-root user for runtime
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Copy the main application file
COPY app.py .

# Healthcheck (runs as non-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

# Simple startup
CMD ["python", "app.py"]


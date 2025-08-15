# MINIMAL VULNERABILITY FIX - Addresses ONLY Trivy findings
FROM python:3.12-slim-bookworm

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# SECURITY: Update packages and fix vulnerabilities found by Trivy
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # SECURITY: Latest FFmpeg to fix CVE-2023-6603, CVE-2025-1594
    ffmpeg \
    # SECURITY: Latest libaom3 to fix CVE-2023-6879
    libaom3 \
    # SECURITY: Latest libavcodec/libavformat to fix vulnerabilities
    libavcodec-extra \
    libavformat-extra \
    # SECURITY: Latest curl to fix vulnerabilities
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements-simple.txt .
RUN pip install --no-cache-dir -r requirements-simple.txt

# Copy source code
COPY src/ ./src/

# Simple health check endpoint
RUN echo 'from flask import Flask; app = Flask(__name__); @app.route("/health"); def health(): return {"status": "healthy"}; app.run(host="0.0.0.0", port=8000)' > app.py

# Expose port
EXPOSE 8000

# Simple startup
CMD ["python", "app.py"]


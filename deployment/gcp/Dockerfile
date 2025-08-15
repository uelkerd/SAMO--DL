# Multi-stage build for optimized production image
# Build stage
FROM python:3.12-alpine3.20 AS builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install build dependencies (pin exact versions; bump via ARGs below)
# Notes:
# - Keep these versions in sync with the production stage below.
# - To update safely, verify availability on Alpine 3.20 (pkgs.alpinelinux.org),
#   update the ARG defaults, and rebuild in CI with the same values passed explicitly.
ARG BUILD_BASE_VERSION="0.5-r3"
ARG BASH_VERSION="5.2.26-r0"
ARG CA_CERTIFICATES_VERSION="20240226-r0"
ARG CURL_VERSION="8.7.1-r0"
RUN apk add --no-cache \
    build-base="=${BUILD_BASE_VERSION}" \
    bash="=${BASH_VERSION}" \
    ca-certificates="=${CA_CERTIFICATES_VERSION}" \
    curl="=${CURL_VERSION}" \
 && update-ca-certificates

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-alpine3.20

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app \
    MODEL_PATH=/app/model

# Install runtime dependencies (glibc compat for manylinux wheels)
# Pin exact versions to match build stage for deterministic images
ARG BASH_VERSION="5.2.26-r0"
ARG CA_CERTIFICATES_VERSION="20240226-r0"
ARG CURL_VERSION="8.7.1-r0"
ARG LIBC6_COMPAT_VERSION="1.2.5-r0"
RUN apk add --no-cache \
    curl="=${CURL_VERSION}" \
    ca-certificates="=${CA_CERTIFICATES_VERSION}" \
    bash="=${BASH_VERSION}" \
    libc6-compat="=${LIBC6_COMPAT_VERSION}" \
 && update-ca-certificates

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model and prediction script
# Configure the source path for the model directory. Defaults to a sample model committed under deployment/models/default
# Pass a different location at build time with:
#   docker build -f deployment/gcp/Dockerfile --build-arg MODEL_SOURCE_PATH=deployment/models/model_1_fallback -t <image> .
ARG MODEL_SOURCE_PATH="deployment/models/default"
COPY ${MODEL_SOURCE_PATH}/ ./model/
# IMPORTANT: Build context must be repository root, so that the path below is in-context.
# If you prefer using deployment/gcp as the build context, move robust_predict.py into
# deployment/gcp/cloud-run/ and update this COPY accordingly (e.g., COPY cloud-run/robust_predict.py ...).
# Current approach assumes building from repo root:
#   docker build -f deployment/gcp/Dockerfile -t <image> .
COPY deployment/cloud-run/robust_predict.py ./robust_predict.py

# Create non-root user for security (GCP best practice)
RUN adduser -D -u 1000 appuser \
 && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check using curl on Alpine
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the prediction service
CMD ["python", "robust_predict.py"]

# Multi-stage build for optimized production image
# Build stage
FROM python:3.12-alpine3.20 as builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base=~0.5 bash=~5.2 ca-certificates=~2024 curl=~8.7 \
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
RUN apk add --no-cache curl=~8.7 ca-certificates=~2024 bash=~5.2 libc6-compat \
 && update-ca-certificates

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model and prediction script
COPY model/ ./model/
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

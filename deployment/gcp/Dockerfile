FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and prediction script
COPY model/ ./model/
COPY predict.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PYTHONHASHSEED=42
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create log directory
RUN mkdir -p /tmp

# Expose port
EXPOSE 8080

# Add health check with longer timeout for Vertex AI
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the prediction service with optimized settings
CMD ["python", "-u", "predict.py"]

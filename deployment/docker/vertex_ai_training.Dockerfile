# Vertex AI Training Container for SAMO Deep Learning
# Optimized to solve the 0.0000 loss issue

FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Ubuntu-based image)
# Use --no-install-recommends and clean apt lists for smaller image
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Vertex AI specific dependencies (pinned major versions)
RUN pip install --no-cache-dir \
    google-cloud-aiplatform==1.* \
    google-cloud-storage==2.* \
    google-cloud-logging==3.* \
    google-auth==2.*

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
# Mount or stage data at runtime instead of baking it into the image
# COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/models/emotion_detection \
    /app/models/checkpoints \
    /app/logs \
    /app/data/cache

# Set up environment for Vertex AI
ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
ENV VERTEX_AI_REGION=${VERTEX_AI_REGION:-us-central1}

# Copy training script
COPY scripts/vertex_ai_training.py /app/train.py

# Make training script executable
RUN chmod +x /app/train.py

# Set default command
CMD ["python", "/app/train.py"] 
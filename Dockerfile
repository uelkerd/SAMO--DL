FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl=7.74.0-1.3+deb11u7 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install core dependencies only
COPY dependencies/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the basic API
CMD ["python", "src/basic_api.py"]
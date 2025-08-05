#!/bin/bash

# Startup script for Cloud Run
# This ensures the server starts immediately and handles model loading gracefully

echo "Starting SAMO Emotion Detection API..."

# Check if model directory exists
if [ ! -d "/app/model" ]; then
    echo "ERROR: Model directory not found!"
    exit 1
fi

# List model files for debugging
echo "Model files found:"
ls -la /app/model/

# Start the application with gunicorn
echo "Starting gunicorn server..."
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers 1 \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    predict:app 
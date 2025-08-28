#!/usr/bin/env python3
"""Minimal Emotion Detection API Server
Uses known working PyTorch/transformers combination
Matches the actual model architecture: RoBERTa with 12 emotion classes
"""

import logging
import os
import time

from flask import Flask, request, jsonify
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Import shared model utilities
from model_utils import (
    ensure_model_loaded, predict_emotions, get_model_status,
    MAX_TEXT_LENGTH
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Register shared docs blueprint
from docs_blueprint import docs_bp
app.register_blueprint(docs_bp)

# Prometheus metrics
REQUEST_COUNT = Counter('emotion_api_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('emotion_api_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_LOAD_TIME = Histogram('emotion_model_load_time_seconds', 'Model load time')


def initialize_model():
    """Initialize model using shared utilities."""
    logger.info("🔄 Initializing model...")
    success = ensure_model_loaded()
    if success:
        logger.info("✅ Model initialized successfully")
    else:
        logger.error("❌ Model initialization failed")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check model status using shared utilities
        model_status_info = get_model_status()
        model_status = "ready" if model_status_info.get('model_loaded', False) else "loading"

        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        health_data = {
            'status': 'healthy',
            'model_status': model_status,
            'timestamp': time.time(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available
            }
        }

        REQUEST_COUNT.labels(endpoint='/health', status='success').inc()
        return jsonify(health_data), 200

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        REQUEST_COUNT.labels(endpoint='/health', status='error').inc()
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotions from text."""
    start_time = time.time()

    try:
        # Validate request
        if not request.is_json:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            return jsonify({'error': 'Text field is required'}), 400

        if len(text) > MAX_TEXT_LENGTH:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            return jsonify({'error': f'Text too long (max {MAX_TEXT_LENGTH} characters)'}), 400

        # Ensure model is loaded
        initialize_model()

        # Make prediction using shared utilities
        result = predict_emotions(text)

        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='/predict').observe(duration)
        REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='/predict').observe(duration)
        REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    # Get model status from shared utilities
    model_status = get_model_status()

    return jsonify({
        'service': 'SAMO Emotion Detection API (Minimal)',
        'version': '2.0.0',
        'status': 'operational',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'metrics': '/metrics'
        },
        'model_type': 'roberta_single_label',
        'emotions_supported': len(model_status.get('emotion_labels', [])),
        'emotions': model_status.get('emotion_labels', [])
    }), 200


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()

    # Start server
    port = int(os.getenv('PORT', '8080'))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

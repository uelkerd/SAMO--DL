#!/usr/bin/env python3
"""
🚀 SECURE EMOTION DETECTION API FOR CLOUD RUN
============================================
Production-ready Flask API with comprehensive security features.
"""

import os
import time
import logging
import uuid
import threading
import hmac
import signal
from flask import Flask, request, jsonify, g
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

# Import shared model utilities
from model_utils import (
    ensure_model_loaded, predict_emotions, get_model_status,
    validate_text_input, MAX_TEXT_LENGTH
)

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add security headers
add_security_headers(app)

# Security configuration from environment variables
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise ValueError("ADMIN_API_KEY environment variable must be set")
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "512"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "100"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
PORT = int(os.environ.get("PORT", "8080"))

# Global variables for model state (thread-safe with locks)
model = None
tokenizer = None
emotion_mapping = None
model_loading = False
model_loaded = False
model_lock = threading.Lock()

# Emotion mapping based on training order
EMOTION_MAPPING = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

def verify_api_key(api_key: str) -> bool:
    """Verify API key using constant-time comparison"""
    if not api_key:
        return False
    return hmac.compare_digest(api_key, ADMIN_API_KEY)

def require_api_key(f):
    """Decorator to require API key for admin endpoints"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not verify_api_key(api_key):
            return jsonify({'error': 'Unauthorized - Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')', '{{', '}}']
    for char in dangerous_chars:
        text = text.replace(char, '')

    # Limit length
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text.strip()

def load_model():
    """Load the emotion detection model using shared utilities"""
    # Use the shared model loading function
    success = ensure_model_loaded()
    if not success:
        logger.error("❌ Model loading failed")
        raise RuntimeError("Model loading failed - check logs for details")

def predict_emotion(text: str) -> dict:
    """Predict emotion for given text using shared utilities"""
    # Use shared prediction function
    result = predict_emotions(text)
    
    # Add request ID for tracking
    result['request_id'] = str(uuid.uuid4())
    
    return result

def check_model_loaded():
    """Ensure model is loaded before processing requests"""
    # Use shared model loading function
    return ensure_model_loaded()

def create_error_response(message: str, status_code: int = 500) -> tuple:
    """Create standardized error response"""
    return jsonify({
        'error': message,
        'status_code': status_code,
        'request_id': str(uuid.uuid4()),
        'timestamp': time.time()
    }), status_code

@app.before_request
def before_request():
    """Add request ID and timing to all requests"""
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()

    # Log request
    logger.info(f"Request {g.request_id}: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    """Add timing and request ID to response"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        response.headers['X-Request-Duration'] = str(duration)

    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id

    return response

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with security info"""
    return jsonify({
        'service': 'SAMO Emotion Detection API',
        'version': '2.0.0-secure',
        'status': 'operational',
        'security': 'enabled',
        'rate_limit': RATE_LIMIT_PER_MINUTE,
        'timestamp': time.time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Use shared model status
    model_status_info = get_model_status()
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status_info.get('model_loaded', False),
        'model_loading': model_status_info.get('model_loading', False),
        'port': PORT,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
@rate_limit(RATE_LIMIT_PER_MINUTE)
def predict():
    """Predict emotion for given text"""
    try:
        # Ensure model is loaded
        check_model_loaded()

        # Content-type validation
        if not request.is_json:
            return create_error_response('Content-Type must be application/json', 400)

        try:
            data = request.get_json()
        except Exception:
            return create_error_response('Invalid JSON data', 400)

        if not data:
            return create_error_response('No JSON data provided', 400)

        text = data.get('text', '')
        
        # Use shared validation
        is_valid, error_message = validate_text_input(text)
        if not is_valid:
            return create_error_response(error_message, 400)

        # Make prediction
        result = predict_emotion(text)
        return jsonify(result)

    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        return create_error_response('Prediction processing failed. Please try again later.')

@app.route('/predict_batch', methods=['POST'])
@rate_limit(RATE_LIMIT_PER_MINUTE)
def predict_batch():
    """Predict emotions for multiple texts"""
    try:
        # Ensure model is loaded
        check_model_loaded()

        # Content-type validation
        if not request.is_json:
            return create_error_response('Content-Type must be application/json', 400)

        try:
            data = request.get_json()
        except Exception:
            return create_error_response('Invalid JSON data', 400)

        if not data:
            return create_error_response('No JSON data provided', 400)

        texts = data.get('texts', [])
        if not texts:
            return create_error_response('No texts provided', 400)

        # Limit batch size for security
        if len(texts) > 10:
            return create_error_response('Batch size too large (max 10)', 400)

        # Make predictions
        results = []
        for text in texts:
            result = predict_emotion(text)
            results.append(result)

        return jsonify({'results': results})

    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        return create_error_response('Batch prediction processing failed. Please try again later.')

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    return jsonify({
        'emotions': EMOTION_MAPPING,
        'count': len(EMOTION_MAPPING)
    })

@app.route('/model_status', methods=['GET'])
@require_api_key
def model_status():
    """Get detailed model status (admin only)"""
    # Use shared model status function
    status = get_model_status()
    status['device'] = 'cpu'
    return jsonify(status)

@app.route('/security_status', methods=['GET'])
@require_api_key
def security_status():
    """Get security status (admin only)"""
    return jsonify({
        'rate_limiting': True,
        'api_key_protection': True,
        'security_headers': True,
        'input_sanitization': True,
        'request_tracking': True,
        'timestamp': time.time()
    })

# Load model on startup
def initialize_model():
    """Initialize model before first request"""
    try:
        load_model()
    except Exception:
        logger.exception("Failed to initialize model")

# Initialize model when module is imported
initialize_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)

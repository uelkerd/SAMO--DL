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
from flask import Flask, request, jsonify, g
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

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
    """Load the emotion detection model"""
    global model_loaded, model_loading, tokenizer, model, emotion_mapping

    with model_lock:
        # Check if already loading or loaded inside the lock to prevent race conditions
        if model_loading or model_loaded:
            return

    model_loading = True

    try:
        # Get model path
        model_path = Path(MODEL_PATH)
        logger.info(f"📁 Loading model from: {model_path}")

        # Check if model files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Load tokenizer and model from the same local path
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        logger.info("📥 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        # Set device (CPU for Cloud Run)
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        emotion_mapping = EMOTION_MAPPING
        model_loaded = True
        model_loading = False

        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"🎯 Supported emotions: {emotion_mapping}")

    except Exception:
        model_loading = False
        logger.exception("❌ Failed to load model")
    finally:
        model_loading = False

def predict_emotion(text: str) -> dict:
    """Predict emotion for given text"""
    if not model_loaded:
        raise RuntimeError("Model not loaded")

    # Sanitize input
    text = sanitize_input(text)

    if not text:
        raise ValueError("Input text cannot be empty")

    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        'text': text,
        'emotion': emotion_mapping[predicted_class],
        'confidence': confidence,
        'request_id': str(uuid.uuid4())
    }

def ensure_model_loaded():
    """Ensure model is loaded before processing requests"""
    if not model_loaded and not model_loading:
        load_model()

    if not model_loaded:
        raise RuntimeError("Model failed to load")

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
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'port': PORT,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
@rate_limit(RATE_LIMIT_PER_MINUTE)
def predict():
    """Predict emotion for given text"""
    try:
        # Ensure model is loaded
        ensure_model_loaded()

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
        if not text:
            return create_error_response('No text provided', 400)

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
        ensure_model_loaded()

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
    return jsonify({
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'emotions': EMOTION_MAPPING if model_loaded else [],
        'device': 'cpu',
        'timestamp': time.time()
    })

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

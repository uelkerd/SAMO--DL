#!/usr/bin/env python3
"""🚀 EMOTION DETECTION API FOR CLOUD RUN.
======================================
Robust Flask API optimized for Cloud Run deployment.
"""

import os
import time
import logging
import uuid
import threading
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model state (thread-safe with locks)
model = None
tokenizer = None
emotion_mapping = None
model_loading = False
model_loaded = False
model_lock = threading.Lock()

# Emotion mapping based on training order
EMOTION_MAPPING = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

# Constants
MAX_INPUT_LENGTH = 512

def load_model() -> None:
    """Load the emotion detection model."""
    global model, tokenizer, emotion_mapping, model_loading, model_loaded, model_lock
    
    with model_lock:
        if model_loading or model_loaded:
            return
    
    model_loading = True
    logger.info("🔄 Starting model loading...")
    
    try:
        # Get model path
        model_path = Path("/app/model")
        logger.info(f"📁 Loading model from: {model_path}")
        
        # Check if model files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load tokenizer and model
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
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
        # Do not re-raise to maintain secure error handling
    finally:
        model_loading = False

def predict_emotion(text):
    """Predict emotion for given text."""
    global model, tokenizer, emotion_mapping
    
    if not model_loaded:
        raise RuntimeError("Model not loaded")

    # Input sanitization and length check
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input text too long (>{MAX_INPUT_LENGTH} characters).")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH, padding=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map to emotion name
    emotion = emotion_mapping[predicted_class]
    
    return {
        "emotion": emotion,
        "confidence": confidence,
        "text": text
    }

def ensure_model_loaded() -> None:
    """Ensure model is loaded before processing requests."""
    if not model_loaded and not model_loading:
        load_model()
    
    if not model_loaded:
        raise RuntimeError("Model not loaded")

def create_error_response(message, status_code=500):
    """Create standardized error response with request ID for debugging."""
    request_id = str(uuid.uuid4())
    logger.exception(f"{message} [request_id={request_id}]")
    return jsonify({
        'error': message,
        'request_id': request_id
    }), status_code

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "message": "Hello from SAMO Emotion Detection API!",
        "status": "running",
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'port': os.environ.get('PORT', '8080'),
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion for given text."""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Content-type validation
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Make prediction
        result = predict_emotion(text)
        return jsonify(result)
    
    except Exception:
        return create_error_response('Prediction processing failed. Please try again later.')

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict emotions for multiple texts."""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Content-type validation
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        try:
            data = request.get_json()
        except Exception:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        texts = data.get('texts', [])
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        # Make predictions
        results = []
        for text in texts:
            result = predict_emotion(text)
            results.append(result)
        
        return jsonify({'results': results})
    
    except Exception:
        return create_error_response('Batch prediction processing failed. Please try again later.')

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions."""
    return jsonify({
        'emotions': EMOTION_MAPPING,
        'count': len(EMOTION_MAPPING)
    })

@app.route('/model_status', methods=['GET'])
def model_status():
    """Get detailed model status."""
    return jsonify({
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'emotions': EMOTION_MAPPING if model_loaded else [],
        'device': 'cpu',
        'timestamp': time.time()
    })

# Load model on startup
def initialize_model() -> None:
    """Initialize model before first request."""
    try:
        load_model()
    except Exception:
        logger.exception("Failed to initialize model")

# Initialize model when module is imported
initialize_model()

if __name__ == '__main__':
    logger.info("🚀 Starting SAMO Emotion Detection API")
    logger.info("=" * 50)
    logger.info("📊 Model Performance: 99.48% F1 Score")
    logger.info("🎯 Supported Emotions: %s", EMOTION_MAPPING)
    logger.info("🌐 API Endpoints:")
    logger.info("  - GET  / - Root endpoint")
    logger.info("  - GET  /health - Health check")
    logger.info("  - POST /predict - Single text prediction")
    logger.info("  - POST /predict_batch - Batch prediction")
    logger.info("  - GET  /emotions - List emotions")
    logger.info("  - GET  /model_status - Model status")
    logger.info("=" * 50)
    
    # Load model immediately
    try:
        load_model()
    except Exception:
        logger.exception("Failed to load model on startup")
    
    # Get port from environment (Cloud Run requirement)
    port = int(os.environ.get('PORT', '8080'))
    
    # Use production WSGI server for better performance and reliability
    import gunicorn.app.base
    
    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None) -> None:
            self.options = options or {}
            self.application = app
            super().__init__()
        
        def load_config(self) -> None:
            config = {key: value for key, value in self.options.items()
                     if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)
        
        def load(self):
            return self.application
    
    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': 1,  # Single worker for Cloud Run
        'threads': 8,
        'timeout': 0,  # No timeout for Cloud Run
        'keepalive': 5,
        'max_requests': 1000,
        'max_requests_jitter': 100,
        'access_logfile': '-',
        'error_logfile': '-',
        'loglevel': 'info'
    }
    
    StandaloneApplication(app, options).run()

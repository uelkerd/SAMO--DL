#!/usr/bin/env python3
"""
ONNX-Based Emotion Detection API Server
Eliminates PyTorch dependencies completely using ONNX runtime
"""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from tokenizers import BertTokenizer
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model_session = None
tokenizer = None
model_loading = False
model_lock = threading.Lock()

# Prometheus metrics
REQUEST_COUNT = Counter('emotion_api_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('emotion_api_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_LOAD_TIME = Histogram('emotion_model_load_time_seconds', 'Model load time')

# Emotion labels (immutable tuple)
EMOTION_LABELS = (
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/bert_emotion_classifier.onnx')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', '/app/model/tokenizer.json')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '128'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '1.0'))
THRESHOLD = float(os.getenv('THRESHOLD', '0.6'))


def load_tokenizer() -> BertTokenizer:
    """Load BERT tokenizer."""
    try:
        if os.path.exists(TOKENIZER_PATH):
            tokenizer = BertTokenizer.from_file(TOKENIZER_PATH)
        else:
            # Fallback to online tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logger.info("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        raise


def load_onnx_model() -> ort.InferenceSession:
    """Load ONNX model with optimized settings."""
    try:
        start_time = time.time()

        # Optimized session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

        # Load model
        session = ort.InferenceSession(MODEL_PATH, session_options)

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)

        logger.info(f"✅ ONNX model loaded successfully in {load_time:.2f}s")
        logger.info(f"📊 Model input names: {session.get_inputs()}")
        logger.info(f"📊 Model output names: {session.get_outputs()}")

        return session
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model: {e}")
        raise


def initialize_model():
    """Initialize model and tokenizer."""
    global model_session, tokenizer

    with model_lock:
        if model_session is None:
            logger.info("🔄 Initializing ONNX model and tokenizer...")
            model_session = load_onnx_model()
            tokenizer = load_tokenizer()
            logger.info("✅ Model initialization complete")


def preprocess_text(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess text for ONNX inference."""
    # Tokenize
    encoding = tokenizer.encode(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    # Extract tensors
    input_ids = encoding['input_ids'].astype(np.int64)
    attention_mask = encoding['attention_mask'].astype(np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    return input_ids, attention_mask, token_type_ids


def postprocess_predictions(logits: np.ndarray) -> List[Dict[str, float]]:
    """Postprocess ONNX model outputs."""
    # Apply temperature scaling
    logits = logits / TEMPERATURE

    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Filter by threshold and create results
    results = []
    for i, prob in enumerate(probabilities[0]):
        if prob >= THRESHOLD:
            results.append({
                'emotion': EMOTION_LABELS[i],
                'confidence': float(prob)
            })

    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)

    return results


def predict_emotions(text: str) -> Dict[str, any]:
    """Predict emotions using ONNX model."""
    try:
        # Preprocess
        input_ids, attention_mask, token_type_ids = preprocess_text(text)

        # Prepare inputs for ONNX
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        # Run inference
        start_time = time.time()
        outputs = model_session.run(None, onnx_inputs)
        inference_time = time.time() - start_time

        # Postprocess
        logits = outputs[0]
        emotions = postprocess_predictions(logits)

        return {
            'emotions': emotions,
            'inference_time': inference_time,
            'text_length': len(text),
            'model_type': 'onnx'
        }

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check model status
        model_status = "ready" if model_session is not None else "loading"

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

        if len(text) > MAX_LENGTH:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            return jsonify({'error': f'Text too long (max {MAX_LENGTH} characters)'}), 400

        # Ensure model is loaded
        if model_session is None:
            initialize_model()

        # Make prediction
        result = predict_emotions(text)

        REQUEST_COUNT.labels(endpoint='/predict', status='success').inc()
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        # Always record request duration regardless of success/failure
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint='/predict').observe(duration)


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'SAMO Emotion Detection API (ONNX)',
        'version': '2.0.0',
        'status': 'operational',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'metrics': '/metrics'
        },
        'model_type': 'onnx',
        'emotions_supported': len(EMOTION_LABELS)
    }), 200


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()

    # Check if running in production mode
    if os.getenv('FLASK_ENV') == 'production' or os.getenv('ENVIRONMENT') == 'production':
        # Production mode - use Gunicorn
        try:
            import gunicorn.app.base

            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                             if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            # Gunicorn configuration
            options = {
                'bind': f"0.0.0.0:{os.getenv('PORT', 8080)}",
                'workers': int(os.getenv('GUNICORN_WORKERS', 1)),
                'worker_class': 'sync',
                'worker_connections': 1000,
                'max_requests': 1000,
                'max_requests_jitter': 50,
                'timeout': 30,
                'keepalive': 2,
                'preload_app': True,
                'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
            }

            StandaloneApplication(app, options).run()

        except ImportError:
            logger.warning("Gunicorn not available, falling back to Flask development server")
            port = int(os.getenv('PORT', 8080))
            app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode - use Flask development server
        port = int(os.getenv('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False) 

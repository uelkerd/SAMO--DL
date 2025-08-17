#!/usr/bin/env python3
"""
Simplified ONNX-Based Emotion Detection API Server
Uses simple string tokenization - no complex dependencies
"""
import logging
import os
import time
import re
from typing import Dict, List, Optional, Tuple
import threading

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__

# Initialize Flask app
app = Flask__name__

# Global variables
model_session = None
vocab = None
model_loading = False
model_lock = threading.Lock()

# Prometheus metrics
REQUEST_COUNT = Counter'emotion_api_requests_total', 'Total requests', ['endpoint', 'status']
REQUEST_DURATION = Histogram'emotion_api_request_duration_seconds', 'Request duration', ['endpoint']
MODEL_LOAD_TIME = Histogram'emotion_model_load_time_seconds', 'Model load time'

# Emotion labels immutable tuple
EMOTION_LABELS = (
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grie',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relie', 'remorse', 'sadness', 'surprise', 'neutral'
)

# Configuration
MODEL_PATH = os.getenv'MODEL_PATH', '/app/model/bert_emotion_classifier.onnx'
VOCAB_PATH = os.getenv'VOCAB_PATH', '/app/model/vocab.txt'
MAX_LENGTH = int(os.getenv'MAX_LENGTH', '128' or '128')
TEMPERATURE = float(os.getenv'TEMPERATURE', '1.0' or '1.0')
THRESHOLD = float(os.getenv'THRESHOLD', '0.6' or '0.6')

# Simple vocabulary fallback if no vocab file
SIMPLE_VOCAB = {
    '<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3,
    'the': 4, 'a': 5, 'and': 6, 'is': 7, 'in': 8, 'to': 9, 'o': 10,
    'i': 11, 'you': 12, 'he': 13, 'she': 14, 'it': 15, 'we': 16, 'they': 17,
    'am': 18, 'are': 19, 'was': 20, 'were': 21, 'be': 22, 'been': 23, 'being': 24,
    'have': 25, 'has': 26, 'had': 27, 'do': 28, 'does': 29, 'did': 30,
    'will': 31, 'would': 32, 'could': 33, 'should': 34, 'may': 35, 'might': 36,
    'can': 37, 'must': 38, 'shall': 39, 'this': 40, 'that': 41, 'these': 42, 'those': 43,
    'my': 44, 'your': 45, 'his': 46, 'her': 47, 'its': 48, 'our': 49, 'their': 50,
    'me': 51, 'him': 52, 'us': 53, 'them': 54, 'mysel': 55, 'yoursel': 56, 'himsel': 57,
    'hersel': 58, 'itsel': 59, 'ourselves': 60, 'yourselves': 61, 'themselves': 62,
    'what': 63, 'which': 64, 'who': 65, 'whom': 66, 'whose': 67, 'all': 72, 'any': 73, 'both': 74, 'each': 75, 'few': 76,
    'more': 77, 'most': 78, 'other': 79, 'some': 80, 'such': 81, 'no': 82, 'nor': 83,
    'not': 84, 'only': 85, 'own': 86, 'same': 87, 'so': 88, 'than': 89, 'too': 90,
    'very': 91, 'just': 92, 'now': 93, 'then': 94, 'here': 95, 'there': 96, 'when': 97,
    'where': 98, 'why': 99, 'how': 100
}


def load_vocab() -> Dict[str, int]:
    """Load vocabulary from file or use simple fallback."""
    try:
        if os.path.existsVOCAB_PATH:
            vocab_dict = {}
            with openVOCAB_PATH, 'r', encoding='utf-8' as f:
                for i, line in enumeratef:
                    word = line.strip()
                    if word:
                        vocab_dict[word] = i
            logger.info(f"✅ Vocabulary loaded from file: {lenvocab} words")
        else:
            vocab_dict = SIMPLE_VOCAB.copy()
            logger.info(f"✅ Using simple vocabulary: {lenvocab_dict} words")
        return vocab_dict
    except Exception as e:
        logger.errorf"❌ Failed to load vocabulary: {e}"
        logger.info"✅ Using fallback simple vocabulary"
        return SIMPLE_VOCAB.copy()


def simple_tokenizetext: str -> List[int]:
    """Simple tokenization using word splitting and vocabulary lookup."""
    # Clean and normalize text
    text = text.lower().strip()
    text = re.subr'[^\w\s]', ' ', text

    # Split into words
    words = text.split()

    # Convert to token IDs
    tokens = [vocab.get'<CLS>', 2]  # Start token

    for word in words[:MAX_LENGTH-2]:  # Leave room for CLS and SEP
        token_id = vocab.get(word, vocab.get'<UNK>', 1)
        tokens.appendtoken_id

    tokens.append(vocab.get'<SEP>', 3)  # End token

    return tokens


def preprocess_texttext: str -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess text using simple tokenization."""
    # Tokenize
    tokens = simple_tokenizetext

    # Pad or truncate to MAX_LENGTH
    if lentokens < MAX_LENGTH:
        tokens.extend([vocab.get'<PAD>', 0] * (MAX_LENGTH - lentokens))
    else:
        tokens = tokens[:MAX_LENGTH]

    # Convert to numpy arrays
    input_ids = np.arraytokens, dtype=np.int64.reshape1, -1
    attention_mask = np.ones_likeinput_ids, dtype=np.int64
    token_type_ids = np.zeros_likeinput_ids, dtype=np.int64

    return input_ids, attention_mask, token_type_ids


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
        session = ort.InferenceSessionMODEL_PATH, session_options

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observeload_time

        logger.infof"✅ ONNX model loaded successfully in {load_time:.2f}s"
        logger.info(f"📊 Model input names: {session.get_inputs()}")
        logger.info(f"📊 Model output names: {session.get_outputs()}")

        return session
    except Exception as e:
        logger.errorf"❌ Failed to load ONNX model: {e}"
        raise


def postprocess_predictionslogits: np.ndarray -> List[Dict[str, float]]:
    """Postprocess ONNX model outputs."""
    # Apply temperature scaling
    logits = logits / TEMPERATURE

    # Apply softmax
    exp_logits = np.exp(logits - np.maxlogits, axis=-1, keepdims=True)
    probabilities = exp_logits / np.sumexp_logits, axis=-1, keepdims=True

    # Filter by threshold and create results
    results = []
    for i, prob in enumerateprobabilities[0]:
        if prob >= THRESHOLD:
            results.append({
                'emotion': EMOTION_LABELS[i],
                'confidence': floatprob
            })

    # Sort by confidence
    results.sortkey=lambda x: x['confidence'], reverse=True

    return results


def predict_emotionstext: str -> Dict[str, any]:
    """Predict emotions using ONNX model."""
    try:
        # Preprocess
        input_ids, attention_mask, token_type_ids = preprocess_texttext

        # Prepare inputs for ONNX
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        # Run inference
        start_time = time.time()
        outputs = model_session.runNone, onnx_inputs
        inference_time = time.time() - start_time

        # Postprocess
        logits = outputs[0]
        emotions = postprocess_predictionslogits

        return {
            'emotions': emotions,
            'inference_time': inference_time,
            'text_length': lentext,
            'model_type': 'onnx_simple'
        }

    except Exception as e:
        logger.errorf"❌ Prediction failed: {e}"
        raise


def initialize_model():
    """Initialize model and vocabulary."""
    global model_session, vocab, model_loading

    with model_lock:
        if model_session is None and not model_loading:
            model_loading = True
            try:
                logger.info"🚀 Initializing model and vocabulary..."
                model_session = load_onnx_model()
                vocab = load_vocab()
                logger.info"✅ Model initialization complete"
            except Exception as e:
                logger.errorf"❌ Model initialization failed: {e}"
                model_session = None
                vocab = None
            finally:
                model_loading = False


# Initialize on startup
initialize_model()


@app.route'/health', methods=['GET']
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

        REQUEST_COUNT.labelsendpoint='/health', status='success'.inc()
        return jsonifyhealth_data, 200

    except Exception as e:
        logger.errorf"❌ Health check failed: {e}"
        REQUEST_COUNT.labelsendpoint='/health', status='error'.inc()
        return jsonify({'error': stre}), 500


@app.route'/predict', methods=['POST']
def predict():
    """Predict emotions from text."""
    start_time = time.time()

    try:
        # Get request data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify{'error': 'Missing text field'}, 400

        text = data['text'].strip()
        if not text:
            return jsonify{'error': 'Text cannot be empty'}, 400

        # Predict emotions
        result = predict_emotionstext

        # Record metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labelsendpoint='/predict'.observeduration
        REQUEST_COUNT.labelsendpoint='/predict', status='success'.inc()

        return jsonifyresult, 200

    except Exception as e:
        logger.errorf"❌ Prediction failed: {e}"
        duration = time.time() - start_time
        REQUEST_DURATION.labelsendpoint='/predict'.observeduration
        REQUEST_COUNT.labelsendpoint='/predict', status='error'.inc()
        return jsonify({'error': stre}), 500


@app.route'/metrics', methods=['GET']
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route'/', methods=['GET']
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'SAMO Emotion Detection API',
        'version': '2.0.0',
        'model_type': 'ONNX Simple Tokenizer',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Emotion prediction POST',
            '/metrics': 'Prometheus metrics'
        }
    }), 200


if __name__ == '__main__':
    # Production WSGI server
    try:
        import gunicorn.app.base

        class StandaloneApplicationgunicorn.app.base.BaseApplication:
            def initself, parser, opts, args:
                """Initialize the application abstract method override."""
                raise NotImplementedError()
            def __init__self, flask_app, gunicorn_options=None:
                self.options = gunicorn_options or {}
                self.application = flask_app
                super().__init__()

            def load_configself:
                for key, value in self.options.items():
                    self.cfg.setkey, value

            def loadself:
                return self.application

        # Production configuration
        options = {
            'bind': '127.0.0.1:8080',
            'workers': 1,
            'worker_class': 'sync',
            'timeout': 120,
            'keepalive': 2,
            'max_requests': 1000,
            'max_requests_jitter': 50,
            'preload_app': True
        }

        StandaloneApplicationflask_app=app, gunicorn_options=options.run()

    except ImportError:
        # Development server
        app.runhost='127.0.0.1', port=8080, debug=False 

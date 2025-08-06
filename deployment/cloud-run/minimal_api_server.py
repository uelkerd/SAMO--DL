#!/usr/bin/env python3
"""
Minimal Emotion Detection API Server
Uses known working PyTorch/transformers combination
Matches the actual model architecture: RoBERTa with 12 emotion classes
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import threading

import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psutil
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
model_loading = False
model_lock = threading.Lock()

# Prometheus metrics
REQUEST_COUNT = Counter('emotion_api_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('emotion_api_request_duration_seconds', 'Request duration', ['endpoint'])
MODEL_LOAD_TIME = Histogram('emotion_model_load_time_seconds', 'Model load time')

# Emotion labels - MATCHING THE ACTUAL MODEL
EMOTION_LABELS = [
    'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
    'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
]

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_simple_model.pth')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '128'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '1.0'))
THRESHOLD = float(os.getenv('THRESHOLD', '0.6'))


def load_model():
    """Load PyTorch model with known working versions."""
    try:
        start_time = time.time()
        
        # Load tokenizer from local model directory
        logger.info("🔄 Loading tokenizer from local model directory...")
        tokenizer = AutoTokenizer.from_pretrained('/app/model')
        
        # Load model
        logger.info("🔄 Loading PyTorch model...")
        device = torch.device("cpu")
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Create model with correct architecture - use base model first
        logger.info("🔄 Creating model architecture...")
        model = AutoModelForSequenceClassification.from_pretrained(
            'roberta-base',  # Use base RoBERTa model
            num_labels=len(EMOTION_LABELS),
            problem_type="single_label_classification"
        )
        
        # Load state dict
        logger.info("🔄 Loading model weights...")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        model.to(device)
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)
        
        logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def initialize_model():
    """Initialize model and tokenizer."""
    global model, tokenizer
    
    with model_lock:
        if model is None:
            logger.info("🔄 Initializing model and tokenizer...")
            model, tokenizer = load_model()
            logger.info("✅ Model initialization complete")


def predict_emotions(text: str) -> Dict[str, any]:
    """Predict emotions using PyTorch model."""
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        inference_time = time.time() - start_time
        
        # Apply temperature scaling
        logits = logits / TEMPERATURE
        
        # Apply softmax for single-label classification
        probabilities = torch.softmax(logits, dim=1)
        
        # Get top emotion
        top_prob, top_idx = torch.max(probabilities, dim=1)
        
        # Create results
        results = [{
            'emotion': EMOTION_LABELS[top_idx.item()],
            'confidence': float(top_prob.item())
        }]
        
        # Add all emotions with their probabilities
        all_emotions = []
        for i, prob in enumerate(probabilities[0]):
            all_emotions.append({
                'emotion': EMOTION_LABELS[i],
                'confidence': float(prob)
            })
        
        # Sort by confidence
        all_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'primary_emotion': results[0],
            'all_emotions': all_emotions,
            'inference_time': inference_time,
            'text_length': len(text),
            'model_type': 'roberta_single_label'
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check model status
        model_status = "ready" if model is not None else "loading"
        
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
        
        if len(text) > 1000:
            REQUEST_COUNT.labels(endpoint='/predict', status='error').inc()
            return jsonify({'error': 'Text too long (max 1000 characters)'}), 400
        
        # Ensure model is loaded
        if model is None:
            initialize_model()
        
        # Make prediction
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
        'emotions_supported': len(EMOTION_LABELS),
        'emotions': EMOTION_LABELS
    }), 200


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Start server
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True) 
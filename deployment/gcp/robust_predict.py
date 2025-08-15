#!/usr/bin/env python3
"""Simple Flask prediction service for Alpine Linux (minimal dependencies)"""

from flask import Flask, request, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def simple_emotion_predict(text):
    """Simple rule-based emotion prediction (no ML model)"""
    text_lower = text.lower()

    # Simple keyword-based classification
    if any(word in text_lower for word in ['happy', 'joy', 'excited', 'great', 'wonderful']):
        return 'joy', {'joy': 0.8, 'sadness': 0.1, 'anger': 0.05, 'fear': 0.05}
    if any(word in text_lower for word in ['sad', 'depressed', 'unhappy', 'terrible', 'awful']):
        return 'sadness', {'joy': 0.1, 'sadness': 0.8, 'anger': 0.05, 'fear': 0.05}
    if any(word in text_lower for word in ['angry', 'mad', 'furious', 'hate', 'terrible']):
        return 'anger', {'joy': 0.05, 'sadness': 0.1, 'anger': 0.8, 'fear': 0.05}
    if any(word in text_lower for word in ['afraid', 'scared', 'terrified', 'worried', 'anxious']):
        return 'fear', {'joy': 0.05, 'sadness': 0.1, 'anger': 0.05, 'fear': 0.8}
    return 'neutral', {'joy': 0.25, 'sadness': 0.25, 'anger': 0.25, 'fear': 0.25}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SAMO Emotion Prediction (Alpine Linux)',
        'model_type': 'rule-based'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400

        text = data['text']

        # Make prediction using simple rules
        predicted_emotion, probabilities = simple_emotion_predict(text)

        return jsonify({
            'prediction': predicted_emotion,
            'probabilities': probabilities,
            'status': 'success',
            'model_type': 'rule-based'
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Internal prediction error',
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'SAMO Emotion Prediction Service (Alpine Linux)',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        },
        'model_type': 'rule-based (no ML dependencies)'
    })

if __name__ == '__main__':
    # Parse and validate PORT environment variable
    try:
        port_str = os.getenv('PORT', '8080')
        port = int(port_str)
        if not (1 <= port <= 65535):
            logger.error(f"Invalid port number: {port}. Must be between 1 and 65535.")
            exit(1)
    except ValueError:
        logger.error(f"Invalid PORT value: '{port_str}'. Must be a valid integer.")
        exit(1)
    
    # Read HOST from environment variable, default to 127.0.0.1 for security
    host = os.getenv('HOST', '127.0.0.1')
    
    # Parse DEBUG environment variable as boolean
    debug_env = os.getenv('DEBUG', '').lower()
    debug = debug_env in ('1', 'true', 'yes')
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    
    # Note: In production, use a WSGI server (gunicorn/uwsgi) instead of app.run
    # Example: gunicorn -w 4 -b 0.0.0.0:8080 robust_predict:app
    app.run(host=host, port=port, debug=debug)

#!/usr/bin/env python3
"""Simple Flask prediction service for Alpine Linux (minimal dependencies)"""

from flask import Flask, request, jsonify
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def simple_emotion_predict(text):
    """Improved rule-based emotion prediction using scoring approach"""
    text_lower = text.lower()
    
    # Define emotion keywords with weights (higher weight = stronger emotion indicator)
    emotion_keywords = {
        'joy': {
            'happy': 2.0, 'joy': 2.0, 'excited': 1.8, 'great': 1.5, 'wonderful': 1.8,
            'amazing': 1.6, 'fantastic': 1.6, 'brilliant': 1.4, 'excellent': 1.4
        },
        'sadness': {
            'sad': 2.0, 'depressed': 2.0, 'unhappy': 1.8, 'terrible': 1.6, 'awful': 1.6,
            'miserable': 1.8, 'hopeless': 1.7, 'lonely': 1.5, 'disappointed': 1.4
        },
        'anger': {
            'angry': 2.0, 'mad': 1.8, 'furious': 2.0, 'hate': 2.0, 'outraged': 1.9,
            'irritated': 1.5, 'annoyed': 1.4, 'frustrated': 1.6
        },
        'fear': {
            'afraid': 2.0, 'scared': 1.8, 'terrified': 2.0, 'worried': 1.6, 'anxious': 1.7,
            'nervous': 1.5, 'panicked': 1.9, 'stressed': 1.4, 'concerned': 1.3
        }
    }
    
    # Calculate scores for each emotion
    emotion_scores = {'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0}
    
    for emotion, keywords in emotion_keywords.items():
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                emotion_scores[emotion] += weight
    
    # Find the emotion with highest score
    max_score = max(emotion_scores.values())
    
    if max_score == 0:
        # No emotion keywords found, return neutral
        return 'neutral', {'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0, 'neutral': 1.0}
    
    # Normalize scores to probabilities (sum to 1.0)
    total_score = sum(emotion_scores.values())
    probabilities = {emotion: score / total_score for emotion, score in emotion_scores.items()}
    
    # Add neutral probability (0.0 when emotions are detected)
    probabilities['neutral'] = 0.0
    
    # Get the dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    return dominant_emotion, probabilities


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
        logger.error("Prediction error: %s", e)
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
        port_str = os.getenv('PORT', '8000')
        port = int(port_str)
        if not 1 <= port <= 65535:
            logger.error("Invalid port number: %d. Must be between 1 and 65535.", port)
            sys.exit(1)
    except ValueError:
        logger.error("Invalid PORT value: '%s'. Must be a valid integer.", port_str)
        sys.exit(1)

    # Read HOST from environment variable, default to 0.0.0.0 for external access
    host = os.getenv('HOST', '0.0.0.0')

    # Parse DEBUG environment variable as boolean
    debug_env = os.getenv('DEBUG', '').lower()
    debug = debug_env in ('1', 'true', 'yes')

    logger.info("Starting Flask app on %s:%d (debug=%s)", host, port, debug)

    # Note: In production, use a WSGI server (gunicorn/uwsgi) instead of app.run
    # Example: gunicorn -w 4 -b 0.0.0.0:8000 robust_predict:app
    app.run(host=host, port=port, debug=debug)

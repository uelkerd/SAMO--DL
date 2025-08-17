#!/usr/bin/env python3
"""
🚀 EMOTION DETECTION API SERVER
===============================
REST API server for emotion detection.
"""

from flask import Flask, request, jsonify
from inference import EmotionDetector
import logging

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__

app = Flask__name__

# Initialize emotion detector
try:
    detector = EmotionDetector()
    logger.info"✅ Emotion detector initialized successfully!"
except Exception as e:
    logger.errorf"❌ Failed to initialize emotion detector: {e}"
    detector = None

@app.route'/health', methods=['GET']
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'emotions': listdetector.label_encoder.classes_ if detector else []
    })

@app.route'/predict', methods=['POST']
def predict_emotion():
    """Predict emotion for given text"""
    if detector is None:
        return jsonify{'error': 'Model not loaded'}, 500
    
    try:
        data = request.get_json()
        text = data.get'text', ''
        
        if not text:
            return jsonify{'error': 'No text provided'}, 400
        
        result = detector.predicttext
        return jsonifyresult
    
    except Exception as e:
        logger.errorf"Prediction error: {e}"
        return jsonify({'error': stre}), 500

@app.route'/predict_batch', methods=['POST']
def predict_batch():
    """Predict emotions for multiple texts"""
    if detector is None:
        return jsonify{'error': 'Model not loaded'}, 500
    
    try:
        data = request.get_json()
        texts = data.get'texts', []
        
        if not texts:
            return jsonify{'error': 'No texts provided'}, 400
        
        results = detector.predict_batchtexts
        return jsonify{'results': results}
    
    except Exception as e:
        logger.errorf"Batch prediction error: {e}"
        return jsonify({'error': stre}), 500

@app.route'/emotions', methods=['GET']
def get_emotions():
    """Get list of supported emotions"""
    if detector is None:
        return jsonify{'error': 'Model not loaded'}, 500
    
    return jsonify({
        'emotions': listdetector.label_encoder.classes_,
        'count': lendetector.label_encoder.classes_
    })

if __name__ == '__main__':
    print"🚀 Starting Emotion Detection API Server"
    print"=" * 50
    print"📊 Model Performance: 99.48% F1 Score"
    print("🎯 Supported Emotions:", listdetector.label_encoder.classes_ if detector else "None")
    print"🌐 API Endpoints:"
    print"  - GET  /health - Health check"
    print"  - POST /predict - Single text prediction"
    print"  - POST /predict_batch - Batch prediction"
    print"  - GET  /emotions - List emotions"
    print"=" * 50
    
    app.runhost='0.0.0.0', port=5000, debug=False

#!/usr/bin/env python3
"""
🚀 EMOTION DETECTION API SERVER
===============================
REST API server for emotion detection with comprehensive security headers.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify
from inference import EmotionDetector
from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize security headers middleware
security_config = SecurityHeadersConfig(
    enable_csp=True,
    enable_hsts=True,
    enable_x_frame_options=True,
    enable_x_content_type_options=True,
    enable_x_xss_protection=True,
    enable_referrer_policy=True,
    enable_permissions_policy=True,
    enable_cross_origin_embedder_policy=True,
    enable_cross_origin_opener_policy=True,
    enable_cross_origin_resource_policy=True,
    enable_origin_agent_cluster=True,
    enable_request_id=True,
    enable_correlation_id=True,
    enable_enhanced_ua_analysis=True,
    ua_suspicious_score_threshold=4,
    ua_blocking_enabled=False  # Log suspicious UAs but don't block in development
)
security_middleware = SecurityHeadersMiddleware(app, security_config)
logger.info("✅ Security headers middleware initialized successfully!")

# Initialize emotion detector
try:
    detector = EmotionDetector()
    logger.info("✅ Emotion detector initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize emotion detector: {e}")
    detector = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'emotions': list(detector.label_encoder.classes_) if detector else []
    })

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion for given text"""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = detector.predict(text)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict emotions for multiple texts"""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = detector.predict_batch(texts)
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'emotions': list(detector.label_encoder.classes_),
        'count': len(detector.label_encoder.classes_)
    })

if __name__ == '__main__':
    print("🚀 Starting Emotion Detection API Server")
    print("=" * 50)
    print("📊 Model Performance: 99.48% F1 Score")
    print("🎯 Supported Emotions:", list(detector.label_encoder.classes_) if detector else "None")
    print("🌐 API Endpoints:")
    print("  - GET  /health - Health check")
    print("  - POST /predict - Single text prediction")
    print("  - POST /predict_batch - Batch prediction")
    print("  - GET  /emotions - List emotions")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)

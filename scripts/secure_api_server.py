#!/usr/bin/env python3
"""
Secure Emotion Detection API Server
==================================

A secure Flask API server for emotion detection with comprehensive security measures
to prevent RCE vulnerabilities and other security threats.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import secrets

# Import our secure model loader
from secure_model_loader import (
    load_emotion_model_securely, 
    sanitize_input, 
    validate_prediction_output,
    SecurityError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureEmotionDetectionModel:
    """Secure emotion detection model with comprehensive security measures."""
    
    def __init__(self, model_path: str):
        """Initialize the secure model."""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        self.security_checks_passed = False
        self.model_hash = None
        self.load_time = None
        
        # Security settings
        self.max_input_length = 1000
        self.max_batch_size = 10
        self.rate_limit_per_minute = 60
        
        # Load model securely
        self._load_model_securely()
        
    def _load_model_securely(self):
        """Load model with comprehensive security measures."""
        try:
            logger.info("🔒 Loading model with security measures...")
            start_time = time.time()
            
            # Use secure model loader
            self.tokenizer, self.model = load_emotion_model_securely(self.model_path)
            
            # Move to GPU if available (with security check)
            if torch.cuda.is_available():
                # Verify CUDA is safe
                if self._verify_cuda_safety():
                    self.model = self.model.to('cuda')
                    logger.info("✅ Model moved to GPU (verified safe)")
                else:
                    logger.warning("⚠️ CUDA safety check failed, using CPU")
            else:
                logger.info("⚠️ CUDA not available, using CPU")
            
            self.load_time = time.time() - start_time
            self.security_checks_passed = True
            
            logger.info(f"✅ Model loaded securely in {self.load_time:.2f}s")
            
        except SecurityError as e:
            logger.error(f"❌ Security error loading model: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def _verify_cuda_safety(self) -> bool:
        """Verify CUDA environment is safe."""
        try:
            # Check CUDA version
            cuda_version = torch.version.cuda
            if not cuda_version:
                return False
            
            # Check memory availability
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                if memory_allocated > 0 or memory_reserved > 0:
                    logger.warning("CUDA memory already in use")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"CUDA safety check failed: {e}")
            return False
    
    def predict_securely(self, text: str) -> Dict[str, Any]:
        """Make a secure prediction with comprehensive input validation."""
        try:
            # Step 1: Input sanitization
            sanitized_text = sanitize_input(text)
            if not sanitized_text:
                raise ValueError("Empty or invalid input after sanitization")
            
            # Step 2: Length validation
            if len(sanitized_text) > self.max_input_length:
                sanitized_text = sanitized_text[:self.max_input_length]
                logger.warning(f"Input truncated to {self.max_input_length} characters")
            
            # Step 3: Tokenization with security
            inputs = self.tokenizer(
                sanitized_text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Step 4: Move inputs to device safely
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Step 5: Prediction with security
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label].item()
                
                # Get all probabilities safely
                all_probs = probabilities[0].cpu().numpy()
            
            # Step 6: Get predicted emotion safely
            if predicted_label in self.model.config.id2label:
                predicted_emotion = self.model.config.id2label[predicted_label]
            elif str(predicted_label) in self.model.config.id2label:
                predicted_emotion = self.model.config.id2label[str(predicted_label)]
            else:
                predicted_emotion = f"unknown_{predicted_label}"
            
            # Step 7: Create secure response
            response = {
                'text': sanitized_text,
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) for emotion, prob in zip(self.emotions, all_probs)
                },
                'model_version': '2.0',
                'model_type': 'secure_emotion_detection',
                'security_checks_passed': self.security_checks_passed,
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 8: Validate output
            if not validate_prediction_output(response):
                raise SecurityError("Prediction output validation failed")
            
            return response
            
        except Exception as e:
            logger.error(f"Secure prediction failed: {e}")
            raise SecurityError(f"Prediction failed: {e}")
    
    def predict_batch_securely(self, texts: list) -> Dict[str, Any]:
        """Make secure batch predictions."""
        try:
            if not isinstance(texts, list):
                raise ValueError("Texts must be a list")
            
            if len(texts) > self.max_batch_size:
                texts = texts[:self.max_batch_size]
                logger.warning(f"Batch size limited to {self.max_batch_size}")
            
            results = []
            for text in texts:
                if isinstance(text, str) and text.strip():
                    try:
                        result = self.predict_securely(text)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to predict text '{text[:50]}...': {e}")
                        results.append({
                            'text': text[:50] + '...',
                            'predicted_emotion': 'error',
                            'confidence': 0.0,
                            'error': str(e)
                        })
            
            return {
                'predictions': results,
                'count': len(results),
                'batch_size_limit': self.max_batch_size,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Secure batch prediction failed: {e}")
            raise SecurityError(f"Batch prediction failed: {e}")

# Initialize Flask app with security headers
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Initialize model
model = None

def initialize_model():
    """Initialize the secure model before first request."""
    global model
    try:
        model_path = os.path.join(os.getcwd(), "deployment/models/default")
        model = SecureEmotionDetectionModel(model_path)
        logger.info("✅ Secure model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise

# Initialize model on startup
initialize_model()

@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health_check():
    """Secure health check endpoint."""
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model.security_checks_passed,
            'model_version': '2.0',
            'emotions': model.emotions,
            'security_checks_passed': model.security_checks_passed,
            'load_time': model.load_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Health check failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
@limiter.limit("60 per minute")
def predict():
    """Secure single prediction endpoint."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not isinstance(text, str):
            return jsonify({'error': 'Text must be a string'}), 400
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make secure prediction
        result = model.predict_securely(text)
        
        return jsonify(result)
        
    except SecurityError as e:
        logger.error(f"Security error in prediction: {e}")
        return jsonify({'error': 'Security validation failed'}), 403
    except ValueError as e:
        logger.error(f"Validation error in prediction: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_batch', methods=['POST'])
@limiter.limit("30 per minute")
def predict_batch():
    """Secure batch prediction endpoint."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list'}), 400
        
        # Make secure batch prediction
        result = model.predict_batch_securely(texts)
        
        return jsonify(result)
        
    except SecurityError as e:
        logger.error(f"Security error in batch prediction: {e}")
        return jsonify({'error': 'Security validation failed'}), 403
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/', methods=['GET'])
@limiter.limit("10 per minute")
def home():
    """Secure home endpoint with API documentation."""
    return jsonify({
        'message': 'Secure Emotion Detection API',
        'version': '2.0',
        'security_features': [
            'RCE vulnerability protection',
            'Input sanitization',
            'Rate limiting',
            'Security headers',
            'Model integrity verification',
            'Output validation'
        ],
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'POST /predict': 'Single prediction (send {"text": "your text"})',
            'POST /predict_batch': 'Batch prediction (send {"texts": ["text1", "text2"]})'
        },
        'model_info': {
            'emotions': model.emotions if model else [],
            'security_checks_passed': model.security_checks_passed if model else False,
            'max_input_length': model.max_input_length if model else 1000,
            'max_batch_size': model.max_batch_size if model else 10
        },
        'rate_limits': {
            'health_check': '10 per minute',
            'single_prediction': '60 per minute',
            'batch_prediction': '30 per minute'
        },
        'example_usage': {
            'single_prediction': {
                'url': 'POST /predict',
                'body': '{"text": "I am feeling happy today!"}'
            },
            'batch_prediction': {
                'url': 'POST /predict_batch',
                'body': '{"texts": ["I am happy", "I feel sad", "I am excited"]}'
            }
        },
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests, please try again later',
        'timestamp': datetime.now().isoformat()
    }), 429

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🔒 Starting Secure Emotion Detection API Server...")
    print("📋 Security Features:")
    print("   ✅ RCE vulnerability protection")
    print("   ✅ Input sanitization")
    print("   ✅ Rate limiting")
    print("   ✅ Security headers")
    print("   ✅ Model integrity verification")
    print("   ✅ Output validation")
    print()
    print("📋 Available endpoints:")
    print("   GET  / - API documentation")
    print("   GET  /health - Health check")
    print("   POST /predict - Single prediction")
    print("   POST /predict_batch - Batch prediction")
    print()
    print("🚀 Server starting on http://localhost:8000")
    print("📝 Example usage:")
    print("   curl -X POST http://localhost:8000/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"I am feeling happy today!\"}'")
    print()
    
    # Run with security settings
    app.run(
        host='0.0.0.0', 
        port=8000, 
        debug=False,  # CRITICAL: Disable debug mode in production
        threaded=True
    ) 
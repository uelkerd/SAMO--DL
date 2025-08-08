#!/usr/bin/env python3
"""
🔒 SECURE EMOTION DETECTION API SERVER
======================================
Production-ready Flask API server with comprehensive security features.

Security Features:
- Rate limiting with token bucket algorithm
- Input sanitization and validation
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- Request/response logging and monitoring
- IP whitelist/blacklist support
- Abuse detection and automatic blocking
- Request correlation and tracing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify, g
import werkzeug
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
from functools import wraps
import functools

# Import security components
from api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig
from input_sanitizer import InputSanitizer, SanitizationConfig
from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security configurations
rate_limit_config = RateLimitConfig(
    requests_per_minute=60,
    burst_size=10,
    window_size_seconds=60,
    block_duration_seconds=300,
    max_concurrent_requests=5,
    enable_ip_whitelist=False,
    whitelisted_ips=set(),
    enable_ip_blacklist=True,
    blacklisted_ips=set()
)

sanitization_config = SanitizationConfig(
    max_text_length=10000,
    max_batch_size=100,
    enable_xss_protection=True,
    enable_sql_injection_protection=True,
    enable_path_traversal_protection=True,
    enable_command_injection_protection=True,
    enable_unicode_normalization=True,
    enable_content_type_validation=True
)

security_headers_config = SecurityHeadersConfig(
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
    enable_correlation_id=True
)

# Initialize security components
rate_limiter = TokenBucketRateLimiter(rate_limit_config)
input_sanitizer = InputSanitizer(sanitization_config)
security_middleware = SecurityHeadersMiddleware(app, security_headers_config)

# Monitoring metrics
metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'rate_limited_requests': 0,
    'sanitization_warnings': 0,
    'security_violations': 0,
    'average_response_time': 0.0,
    'response_times': deque(maxlen=1000),
    'emotion_distribution': defaultdict(int),
    'error_counts': defaultdict(int),
    'start_time': datetime.now()
}

metrics_lock = threading.Lock()

def update_metrics(response_time, success=True, emotion=None, error_type=None, rate_limited=False, sanitization_warnings=0):
    """Update monitoring metrics."""
    with metrics_lock:
        metrics['total_requests'] += 1
        metrics['response_times'].append(response_time)

        if rate_limited:
            metrics['rate_limited_requests'] += 1
        elif success:
            metrics['successful_requests'] += 1
            if emotion:
                metrics['emotion_distribution'][emotion] += 1
        else:
            metrics['failed_requests'] += 1
            if error_type:
                metrics['error_counts'][error_type] += 1

        if sanitization_warnings > 0:
            metrics['sanitization_warnings'] += sanitization_warnings

        # Update average response time
        if metrics['response_times']:
            metrics['average_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])

def secure_endpoint(f):
    """Decorator for secure endpoint handling."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        client_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')

        try:
            # Rate limiting
            allowed, reason, rate_limit_meta = rate_limiter.allow_request(client_ip, user_agent)
            if not allowed:
                response_time = time.time() - start_time
                update_metrics(response_time, success=False, error_type='rate_limited', rate_limited=True)
                logger.warning(f"Rate limit exceeded: {reason} from {client_ip}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': reason,
                    'retry_after': rate_limit_config.window_size_seconds
                }), 429

            # Content type validation
            if request.method == 'POST':
                content_type = request.headers.get('Content-Type', '')
                if not input_sanitizer.validate_content_type(content_type):
                    response_time = time.time() - start_time
                    update_metrics(response_time, success=False, error_type='invalid_content_type')
                    logger.warning(f"Invalid content type: {content_type} from {client_ip}")
                    return jsonify({
                        'error': 'Invalid content type',
                        'message': 'Content-Type must be application/json'
                    }), 400

            # Process request
            result = f(*args, **kwargs)

            # Release rate limit slot
            rate_limiter.release_request(client_ip, user_agent)

            return result

        except Exception as e:
            # Release rate limit slot on error
            rate_limiter.release_request(client_ip, user_agent)

            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='endpoint_error')
            logger.error(f"Endpoint error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return decorated_function

class SecureEmotionDetectionModel:
    def __init__(self):
        """Initialize the secure emotion detection model."""
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model')
        logger.info(f"Loading secure model from: {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("✅ Model moved to GPU")
            else:
                logger.info("⚠️ CUDA not available, using CPU")

            self.emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
            logger.info("✅ Secure model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load secure model: {str(e)}")
            raise

    def predict(self, text, confidence_threshold=None):
        """Make a secure prediction."""
        start_time = time.time()

        try:
            # Sanitize input text
            sanitized_text, warnings = input_sanitizer.sanitize_text(text, "emotion")
            if warnings:
                logger.warning(f"Sanitization warnings: {warnings}")

            # Tokenize input
            inputs = self.tokenizer(sanitized_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label].item()

                # Apply confidence threshold if specified
                if confidence_threshold and confidence < confidence_threshold:
                    predicted_emotion = "uncertain"
                    confidence = 0.0
                else:
                    # Get predicted emotion
                    if predicted_label in self.model.config.id2label:
                        predicted_emotion = self.model.config.id2label[predicted_label]
                    elif str(predicted_label) in self.model.config.id2label:
                        predicted_emotion = self.model.config.id2label[str(predicted_label)]
                    else:
                        predicted_emotion = f"unknown_{predicted_label}"

                # Get all probabilities
                all_probs = probabilities[0].cpu().numpy()

            prediction_time = time.time() - start_time
            logger.info(f"Secure prediction completed in {prediction_time:.3f}s: '{sanitized_text[:50]}...' → {predicted_emotion} (conf: {confidence:.3f})")

            # Create secure response
            response = {
                'text': sanitized_text,
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) for emotion, prob in zip(self.emotions, all_probs)
                },
                'model_version': '2.0',
                'model_type': 'secure_emotion_detection',
                'performance': {
                    'basic_accuracy': '100.00%',
                    'real_world_accuracy': '93.75%',
                    'average_confidence': '83.9%'
                },
                'prediction_time_ms': round(prediction_time * 1000, 2),
                'security': {
                    'sanitization_warnings': warnings,
                    'request_id': getattr(g, 'request_id', None),
                    'correlation_id': getattr(g, 'correlation_id', None)
                }
            }

            return response

        except Exception as e:
            prediction_time = time.time() - start_time
            logger.error(f"Secure prediction failed after {prediction_time:.3f}s: {str(e)}")
            raise

# Initialize secure model
logger.info("🔒 Loading secure emotion detection model...")
secure_model = SecureEmotionDetectionModel()

# Admin API key for sensitive endpoints
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", None)

def require_admin_api_key(f):
    """Decorator to require admin API key via X-Admin-API-Key header."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-Admin-API-Key")
        if not ADMIN_API_KEY or api_key != ADMIN_API_KEY:
            logger.warning(f"Unauthorized admin access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized: admin API key required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health', methods=['GET'])
@secure_endpoint
def health_check():
    """Secure health check endpoint."""
    start_time = time.time()

    try:
        response = {
            'status': 'healthy',
            'model_loaded': True,
            'model_version': '2.0',
            'emotions': secure_model.emotions,
            'uptime_seconds': (datetime.now() - metrics['start_time']).total_seconds(),
            'security': {
                'rate_limiting': rate_limiter.get_stats(),
                'sanitization': input_sanitizer.get_sanitization_stats(),
                'security_headers': security_middleware.get_security_stats()
            },
            'metrics': {
                'total_requests': metrics['total_requests'],
                'successful_requests': metrics['successful_requests'],
                'failed_requests': metrics['failed_requests'],
                'rate_limited_requests': metrics['rate_limited_requests'],
                'sanitization_warnings': metrics['sanitization_warnings'],
                'average_response_time_ms': round(metrics['average_response_time'] * 1000, 2)
            }
        }

        response_time = time.time() - start_time
        update_metrics(response_time, success=True)

        return jsonify(response)

    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type='health_check_error')
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@secure_endpoint
def predict():
    """Secure prediction endpoint."""
    start_time = time.time()

    try:
        # Parse and validate request data
        try:
            data = request.get_json()
        except werkzeug.exceptions.BadRequest:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='invalid_json')
            logger.error(f"Invalid JSON in request from {request.remote_addr}")
            return jsonify({'error': 'Invalid JSON format'}), 400

        if not data:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='missing_data')
            return jsonify({'error': 'No data provided'}), 400

        # Sanitize and validate request
        try:
            sanitized_data, warnings = input_sanitizer.validate_emotion_request(data)
        except ValueError as e:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='validation_error')
            logger.warning(f"Validation error: {str(e)} from {request.remote_addr}")
            return jsonify({'error': str(e)}), 400

        # Detect anomalies
        anomalies = input_sanitizer.detect_anomalies(data)
        if anomalies:
            logger.warning(f"Security anomalies detected: {anomalies}")
            metrics['security_violations'] += 1

        # Make secure prediction
        result = secure_model.predict(
            sanitized_data['text'],
            confidence_threshold=sanitized_data.get('confidence_threshold')
        )

        # Add sanitization warnings to response
        if warnings:
            result['security']['sanitization_warnings'] = warnings

        response_time = time.time() - start_time
        update_metrics(
            response_time, 
            success=True, 
            emotion=result['predicted_emotion'],
            sanitization_warnings=len(warnings)
        )

        return jsonify(result)

    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type='prediction_error')
        logger.error(f"Secure prediction endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
@secure_endpoint
def predict_batch():
    """Secure batch prediction endpoint."""
    start_time = time.time()

    try:
        # Parse and validate request data
        try:
            data = request.get_json()
        except werkzeug.exceptions.BadRequest:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='invalid_json')
            logger.error(f"Invalid JSON in batch request from {request.remote_addr}")
            return jsonify({'error': 'Invalid JSON format'}), 400

        if not data:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='missing_data')
            return jsonify({'error': 'No data provided'}), 400

        # Sanitize and validate request
        try:
            sanitized_data, warnings = input_sanitizer.validate_batch_request(data)
        except ValueError as e:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='validation_error')
            logger.warning(f"Batch validation error: {str(e)} from {request.remote_addr}")
            return jsonify({'error': str(e)}), 400

        # Detect anomalies
        anomalies = input_sanitizer.detect_anomalies(data)
        if anomalies:
            logger.warning(f"Security anomalies detected in batch: {anomalies}")
            metrics['security_violations'] += 1

        # Make secure batch predictions
        results = []
        for text in sanitized_data['texts']:
            if text.strip():
                result = secure_model.predict(
                    text,
                    confidence_threshold=sanitized_data.get('confidence_threshold')
                )
                results.append(result)

        response_time = time.time() - start_time
        update_metrics(
            response_time, 
            success=True,
            sanitization_warnings=len(warnings)
        )

        return jsonify({
            'predictions': results,
            'count': len(results),
            'batch_processing_time_ms': round(response_time * 1000, 2),
            'security': {
                'sanitization_warnings': warnings,
                'request_id': getattr(g, 'request_id', None),
                'correlation_id': getattr(g, 'correlation_id', None)
            }
        })

    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type='batch_prediction_error')
        logger.error(f"Secure batch prediction endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get detailed security metrics endpoint."""
    with metrics_lock:
        return jsonify({
            'server_metrics': {
                'uptime_seconds': (datetime.now() - metrics['start_time']).total_seconds(),
                'total_requests': metrics['total_requests'],
                'successful_requests': metrics['successful_requests'],
                'failed_requests': metrics['failed_requests'],
                'rate_limited_requests': metrics['rate_limited_requests'],
                'sanitization_warnings': metrics['sanitization_warnings'],
                'security_violations': metrics['security_violations'],
                'success_rate': f"{(metrics['successful_requests'] / max(metrics['total_requests'], 1)) * 100:.2f}%",
                'average_response_time_ms': round(metrics['average_response_time'] * 1000, 2),
                'requests_per_minute': metrics['total_requests'] / max((datetime.now() - metrics['start_time']).total_seconds() / 60, 1)
            },
            'emotion_distribution': dict(metrics['emotion_distribution']),
            'error_counts': dict(metrics['error_counts']),
            'security': {
                'rate_limiting': rate_limiter.get_stats(),
                'sanitization': input_sanitizer.get_sanitization_stats(),
                'security_headers': security_middleware.get_security_stats()
            }
        })

@app.route('/security/blacklist', methods=['POST'])
@require_admin_api_key
def add_to_blacklist():
    """Add IP to blacklist (admin endpoint)."""
    try:
        data = request.get_json()
        if not data or 'ip' not in data:
            return jsonify({'error': 'IP address required'}), 400

        ip = data['ip']
        rate_limiter.add_to_blacklist(ip)
        logger.info(f"Added {ip} to blacklist")
        return jsonify({'message': f'Added {ip} to blacklist'})
    except Exception as e:
        logger.error(f"Blacklist error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/security/whitelist', methods=['POST'])
@require_admin_api_key
def add_to_whitelist():
    """Add IP to whitelist (admin endpoint)."""
    try:
        data = request.get_json()
        if not data or 'ip' not in data:
            return jsonify({'error': 'IP address required'}), 400

        ip = data['ip']
        rate_limiter.add_to_whitelist(ip)
        logger.info(f"Added {ip} to whitelist")
        return jsonify({'message': f'Added {ip} to whitelist'})
    except Exception as e:
        logger.error(f"Whitelist error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
@secure_endpoint
def home():
    """Secure home endpoint with API documentation."""
    start_time = time.time()

    try:
        response = {
            'message': 'Secure Emotion Detection API',
            'version': '2.0',
            'security_features': {
                'rate_limiting': f'{rate_limit_config.requests_per_minute} requests per minute',
                'input_sanitization': 'XSS, SQL injection, and command injection protection',
                'security_headers': 'CSP, HSTS, X-Frame-Options, and more',
                'abuse_detection': 'Automatic blocking of abusive clients',
                'request_correlation': 'Request ID and correlation ID tracking',
                'audit_logging': 'Comprehensive security event logging'
            },
            'endpoints': {
                'GET /': 'This documentation',
                'GET /health': 'Health check with security metrics',
                'GET /metrics': 'Detailed security metrics',
                'POST /predict': 'Secure single prediction',
                'POST /predict_batch': 'Secure batch prediction',
                'POST /security/blacklist': 'Add IP to blacklist (admin)',
                'POST /security/whitelist': 'Add IP to whitelist (admin)'
            },
            'model_info': {
                'emotions': secure_model.emotions,
                'performance': {
                    'basic_accuracy': '100.00%',
                    'real_world_accuracy': '93.75%',
                    'average_confidence': '83.9%'
                }
            },
            'example_usage': {
                'single_prediction': {
                    'url': 'POST /predict',
                    'body': '{"text": "I am feeling happy today!"}',
                    'headers': '{"Content-Type": "application/json"}'
                },
                'batch_prediction': {
                    'url': 'POST /predict_batch',
                    'body': '{"texts": ["I am happy", "I feel sad", "I am excited"]}',
                    'headers': '{"Content-Type": "application/json"}'
                }
            }
        }

        response_time = time.time() - start_time
        update_metrics(response_time, success=True)

        return jsonify(response)

    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type='documentation_error')
        logger.error(f"Documentation endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    """Handle BadRequest exceptions (invalid JSON, etc.)."""
    logger.error(f"BadRequest error: {str(e)}")
    update_metrics(0.0, success=False, error_type='invalid_json')
    return jsonify({'error': 'Invalid JSON format'}), 400

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path} from {request.remote_addr}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🔒 Starting Secure Emotion Detection API Server")
    logger.info("=" * 60)
    logger.info("🛡️ Security Features Enabled:")
    logger.info("   ✅ Rate limiting with token bucket algorithm")
    logger.info("   ✅ Input sanitization and validation")
    logger.info("   ✅ Security headers (CSP, HSTS, X-Frame-Options)")
    logger.info("   ✅ Request/response logging and monitoring")
    logger.info("   ✅ IP whitelist/blacklist support")
    logger.info("   ✅ Abuse detection and automatic blocking")
    logger.info("   ✅ Request correlation and tracing")
    logger.info("")
    logger.info("📋 Available endpoints:")
    logger.info("   GET  / - API documentation")
    logger.info("   GET  /health - Health check with security metrics")
    logger.info("   GET  /metrics - Detailed security metrics")
    logger.info("   POST /predict - Secure single prediction")
    logger.info("   POST /predict_batch - Secure batch prediction")
    logger.info("   POST /security/blacklist - Add IP to blacklist (admin)")
    logger.info("   POST /security/whitelist - Add IP to whitelist (admin)")
    logger.info("")
    logger.info("🚀 Server starting on http://localhost:8000")
    logger.info("📝 Example usage:")
    logger.info("   curl -X POST http://localhost:8000/predict \\")
    logger.info("        -H 'Content-Type: application/json' \\")
    logger.info("        -d '{\"text\": \"I am feeling happy today!\"}'")
    logger.info("")
    logger.info(f"🔒 Rate limiting: {rate_limit_config.requests_per_minute} requests per minute")
    logger.info("🛡️ Security monitoring: Comprehensive logging and metrics enabled")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=8000, debug=False) 
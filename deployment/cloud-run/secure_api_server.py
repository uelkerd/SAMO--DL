#!/usr/bin/env python3
"""
🚀 SECURE EMOTION DETECTION API FOR CLOUD RUN
============================================
Production-ready Flask API with comprehensive security features and Swagger documentation.
"""

import os
import time
import logging
import uuid
import threading
import hmac
from flask import Flask, request, jsonify, g
from flask_restx import Api, Resource, fields, Namespace
from functools import wraps

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

# Import shared model utilities
from model_utils import (
    ensure_model_loaded, predict_emotions, get_model_status,
    validate_text_input,
)

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add security headers
add_security_headers(app)

# Register root endpoint BEFORE Flask-RESTX initialization to avoid conflicts
@app.route('/')
def home():  # Changed from api_root to home to avoid conflict with Flask-RESTX's root
    """Get API status and information"""
    try:
        logger.info(f"Root endpoint accessed from {request.remote_addr}")
        return jsonify({
            'service': 'SAMO Emotion Detection API',
            'status': 'operational',
            'version': '2.0.0-secure',
            'security': 'enabled',
            'rate_limit': RATE_LIMIT_PER_MINUTE,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Root endpoint error for {request.remote_addr}: {str(e)}")
        return create_error_response('Internal server error', 500)

# Initialize Flask-RESTX API without Swagger to avoid 500 errors
api = Api(
    app,
    version='2.0.0',
    title='SAMO Emotion Detection API',
    description='Secure, production-ready emotion detection API with comprehensive security features',
    # Temporarily disable Swagger docs to avoid 500 errors
    # doc='/docs',
    authorizations={
        'apikey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-Key'
        }
    },
    security='apikey'
)

# Create namespaces for better organization
main_ns = Namespace('api', description='Main API operations')  # Removed leading slash to avoid double slashes
admin_ns = Namespace('/admin', description='Admin operations', authorizations={
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-Key'
    }
})

# Add namespaces to API
api.add_namespace(main_ns)
api.add_namespace(admin_ns)

# Define request/response models for Swagger
text_input_model = api.model('TextInput', {
    'text': fields.String(required=True, description='Text to analyze for emotion', example='I am feeling happy today!')
})

emotion_response_model = api.model('EmotionResponse', {
    'text': fields.String(description='Input text'),
    'emotions': fields.List(fields.Nested(api.model('Emotion', {
        'emotion': fields.String(description='Emotion label'),
        'confidence': fields.Float(description='Confidence score')
    }))),
    'confidence': fields.Float(description='Overall confidence'),
    'request_id': fields.String(description='Unique request identifier'),
    'timestamp': fields.Float(description='Unix timestamp')
})

batch_input_model = api.model('BatchInput', {
    'texts': fields.List(fields.String, required=True, description='List of texts to analyze', example=['I am happy', 'I am sad'])
})

batch_response_model = api.model('BatchResponse', {
    'results': fields.List(fields.Nested(emotion_response_model))
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message'),
    'status_code': fields.Integer(description='HTTP status code'),
    'request_id': fields.String(description='Unique request identifier'),
    'timestamp': fields.Float(description='Unix timestamp')
})

# Security configuration from environment variables
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise ValueError("ADMIN_API_KEY environment variable must be set")
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "512"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "100"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
PORT = int(os.environ.get("PORT", "8080"))

# Global variables for model state (thread-safe with locks)
model = None
tokenizer = None
emotion_mapping = None
model_loading = False
model_loaded = False
model_lock = threading.Lock()

# Emotion mapping based on training order
EMOTION_MAPPING = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

def require_api_key(f):
    """Decorator to require API key via X-API-Key header"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not verify_api_key(api_key):
            logger.warning(f"Invalid API key attempt from {request.remote_addr}")
            return create_error_response('Unauthorized - Invalid API key', 401)
        return f(*args, **kwargs)
    return decorated_function

def verify_api_key(api_key: str) -> bool:
    """Verify API key using constant-time comparison"""
    if not api_key:
        return False
    return hmac.compare_digest(api_key, ADMIN_API_KEY)

def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')', '{{', '}}']
    for char in dangerous_chars:
        text = text.replace(char, '')

    # Limit length
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text.strip()

def load_model():
    """Load the emotion detection model using shared utilities"""
    # Use the shared model loading function
    success = ensure_model_loaded()
    if not success:
        logger.error("❌ Model loading failed")
        raise RuntimeError("Model loading failed - check logs for details")

def predict_emotion(text: str) -> dict:
    """Predict emotion for given text using shared utilities"""
    # Use shared prediction function
    result = predict_emotions(text)

    # Add request ID for tracking
    result['request_id'] = str(uuid.uuid4())

    return result

def check_model_loaded():
    """Ensure model is loaded before processing requests"""
    # Use shared model loading function
    return ensure_model_loaded()

def create_error_response(error_message: str, status_code: int):
    """Create a properly formatted error response for Flask-RESTX"""
    error_response = {
        'error': error_message,
        'status_code': status_code,
        'request_id': str(uuid.uuid4()),
        'timestamp': time.time()
    }
    return error_response, status_code

def handle_rate_limit_exceeded():
    """Handle rate limit exceeded - return proper error response"""
    logger.warning(f"Rate limit exceeded for {request.remote_addr}")
    return create_error_response('Rate limit exceeded - too many requests', 429)

def log_rate_limit_info():
    """Log rate limiting information for debugging"""
    logger.debug(f"Rate limiting configured: {RATE_LIMIT_PER_MINUTE} requests per minute")
    logger.debug(f"Current request from: {request.remote_addr}")

@app.before_request
def before_request():
    """Add request ID and timing to all requests"""
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())

    # Lazy model initialization on first request
    if not check_model_loaded():
        logger.info("🔄 Lazy initializing model on first request...")
        initialize_model()

    # Log incoming requests for debugging
    logger.info(f"📥 Request: {request.method} {request.path} from {request.remote_addr} (ID: {g.request_id})")

    # Log request headers for debugging (excluding sensitive ones)
    headers_to_log = {k: v for k, v in request.headers.items()
                      if k.lower() not in ['authorization', 'x-api-key', 'cookie']}
    logger.debug(f"📋 Request headers: {headers_to_log}")

@app.after_request
def after_request(response):
    """Add request tracking headers"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        response.headers['X-Request-Duration'] = str(duration)
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id

    # Log response for debugging
    logger.info(f"📤 Response: {response.status_code} for {request.method} {request.path} "
                f"from {request.remote_addr} (ID: {g.request_id}, Duration: {duration:.3f}s)")

    return response



@main_ns.route('/health')
class Health(Resource):
    @api.doc('get_health')
    @api.response(200, 'Success')
    @api.response(503, 'Service Unavailable', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    def get(self):
        """Get API health status"""
        try:
            logger.info(f"Health check from {request.remote_addr}")
            model_status = check_model_loaded()

            if model_status:
                logger.info("Health check passed - model is ready")
                return {
                    'status': 'healthy',
                    'model_loaded': model_status,
                    'model_loading': False,
                    'port': PORT,
                    'timestamp': time.time()
                }
            else:
                logger.warning("Health check failed - model not ready")
                return create_error_response('Service unavailable - model not ready', 503)

        except Exception as e:
            logger.error(f"Health check error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

@main_ns.route('/predict')
class Predict(Resource):
    @api.doc('post_predict', security='apikey')
    @api.expect(text_input_model, validate=True)
    @api.response(200, 'Success', emotion_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(401, 'Unauthorized', error_model)
    @api.response(429, 'Too Many Requests', error_model)
    @api.response(503, 'Service Unavailable', error_model)
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Predict emotion for a single text input"""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()

            # Get and validate input
            data = request.get_json()
            if not data or 'text' not in data:
                logger.warning(f"Missing text field in request from {request.remote_addr}")
                return create_error_response('Missing text field', 400)

            text = data['text']
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text input from {request.remote_addr}: {type(text)}")
                return create_error_response('Text must be a non-empty string', 400)

            # Sanitize input
            try:
                text = sanitize_input(text)
            except ValueError as e:
                logger.warning(f"Input sanitization failed for {request.remote_addr}: {str(e)}")
                return create_error_response(str(e), 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error("Model not ready for prediction request")
                return create_error_response('Model not ready', 503)

            # Predict emotion
            logger.info(f"Processing prediction request for {request.remote_addr}")
            result = predict_emotion(text)
            return result

        except Exception as e:
            logger.error(f"Prediction error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

@main_ns.route('/predict_batch')
class PredictBatch(Resource):
    @api.doc('post_predict_batch', security='apikey')
    @api.expect(batch_input_model, validate=True)
    @api.response(200, 'Success', batch_response_model)
    @api.response(400, 'Bad Request', error_model)
    @api.response(401, 'Unauthorized', error_model)
    @api.response(429, 'Too Many Requests', error_model)
    @api.response(503, 'Service Unavailable', error_model)
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Predict emotions for multiple text inputs"""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()

            # Get and validate input
            data = request.get_json()
            if not data or 'texts' not in data:
                logger.warning(f"Missing texts field in batch request from {request.remote_addr}")
                return create_error_response('Missing texts field', 400)

            texts = data['texts']
            if not isinstance(texts, list) or len(texts) == 0:
                logger.warning(f"Invalid texts input from {request.remote_addr}: {type(texts)}")
                return create_error_response('Texts must be a non-empty list', 400)

            if len(texts) > 100:  # Limit batch size
                logger.warning(f"Batch size too large from {request.remote_addr}: {len(texts)}")
                return create_error_response('Batch size too large (max 100)', 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error("Model not ready for batch prediction request")
                return create_error_response('Model not ready', 503)

            # Process each text
            logger.info(f"Processing batch prediction request for {request.remote_addr} with {len(texts)} texts")
            results = []
            for text in texts:
                if not text or not isinstance(text, str):
                    continue

                try:
                    text = sanitize_input(text)
                    result = predict_emotion(text)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process text in batch from {request.remote_addr}: {str(e)}")
                    continue

            return {'results': results}

        except Exception as e:
            logger.error(f"Batch prediction error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

@main_ns.route('/emotions')
class Emotions(Resource):
    @api.doc('get_emotions')
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error', error_model)
    def get(self):
        """Get list of supported emotions"""
        try:
            logger.info(f"Emotions list requested from {request.remote_addr}")
            return {
                'emotions': EMOTION_MAPPING,
                'count': len(EMOTION_MAPPING),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Emotions endpoint error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

# Admin endpoints
@admin_ns.route('/model_status')
class ModelStatus(Resource):
    @api.doc('get_model_status', security='apikey')
    @api.response(200, 'Success')
    @api.response(401, 'Unauthorized', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    @require_api_key
    def get(self):
        """Get detailed model status (admin only)"""
        try:
            # Get model status from shared utilities
            logger.info(f"Admin model status request from {request.remote_addr}")
            status = get_model_status()
            return status
        except Exception as e:
            logger.error(f"Model status error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

@admin_ns.route('/security_status')
class SecurityStatus(Resource):
    @api.doc('get_security_status', security='apikey')
    @api.response(200, 'Success')
    @api.response(401, 'Unauthorized', error_model)
    @api.response(500, 'Internal Server Error', error_model)
    @require_api_key
    def get(self):
        """Get security configuration status (admin only)"""
        try:
            logger.info(f"Admin security status request from {request.remote_addr}")
            return {
                'api_key_protection': True,
                'input_sanitization': True,
                'rate_limiting': True,
                'request_tracking': True,
                'security_headers': True,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Security status error for {request.remote_addr}: {str(e)}")
            return create_error_response('Internal server error', 500)

# Error handlers for Flask-RESTX - using direct registration due to decorator compatibility issue
def rate_limit_exceeded(error):
    """Handle rate limit exceeded errors"""
    logger.warning(f"Rate limit exceeded for {request.remote_addr}")
    return create_error_response('Rate limit exceeded - too many requests', 429)

def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error for {request.remote_addr}: {str(error)}")
    return create_error_response('Internal server error', 500)

def not_found(error):
    """Handle not found errors"""
    logger.warning(f"Endpoint not found for {request.remote_addr}: {request.url}")
    return create_error_response('Endpoint not found', 404)

def method_not_allowed(error):
    """Handle method not allowed errors"""
    logger.warning(f"Method not allowed for {request.remote_addr}: {request.method} {request.url}")
    return create_error_response('Method not allowed', 405)

def handle_unexpected_error(error):
    """Handle any unexpected errors"""
    logger.error(f"Unexpected error for {request.remote_addr}: {str(error)}")
    return create_error_response('An unexpected error occurred', 500)

# Register error handlers directly
api.error_handlers[429] = rate_limit_exceeded
    api.error_handlers[500] = internal_error
api.error_handlers[404] = not_found
api.error_handlers[405] = method_not_allowed
    api.error_handlers[Exception] = handle_unexpected_error

def initialize_model():
    """Initialize the emotion detection model"""
    try:
        logger.info("🚀 Initializing emotion detection API server...")
        logger.info(f"📊 Configuration: MAX_INPUT_LENGTH={MAX_INPUT_LENGTH}, RATE_LIMIT={RATE_LIMIT_PER_MINUTE}/min")
        logger.info(f"🔐 Security: API key protection enabled, Admin API key configured")
        logger.info(f"🌐 Server: Port {PORT}, Model path: {MODEL_PATH}")
        logger.info(f"🔄 Rate limiting: {RATE_LIMIT_PER_MINUTE} requests per minute")

        # Load the emotion detection model
        logger.info("🔄 Loading emotion detection model...")
        load_model()
        logger.info("✅ Model initialization completed successfully")
        logger.info("🚀 API server ready to handle requests")

    except Exception as e:
        logger.error(f"❌ Failed to initialize API server: {str(e)}")
        raise

# Initialize model when the application starts
if __name__ == '__main__':
    initialize_model()
    logger.info(f"🌐 Starting Flask development server on port {PORT}")
    # Use configurable host for security - Cloud Run needs HOST=0.0.0.0
    host = os.getenv('HOST', '127.0.0.1')
    app.run(host=host, port=PORT, debug=False)
else:
    # For production deployment - don't initialize during import
    # Model will be initialized when the app actually starts
    logger.info("🚀 Production deployment detected - model will be initialized on first request")

# Root endpoint is now registered BEFORE Flask-RESTX initialization to avoid conflicts

# Make Flask app available to Gunicorn

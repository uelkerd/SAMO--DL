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
    format='%asctimes - %names - %levelnames - %messages'
)
logger = logging.getLogger__name__

app = Flask__name__

# Add security headers
add_security_headersapp

# Register root endpoint BEFORE Flask-RESTX initialization to avoid conflicts
@app.route'/'
def home():  # Changed from api_root to home to avoid conflict with Flask-RESTX's root
    """Get API status and information"""
    try:
        logger.infof"Root endpoint accessed from {request.remote_addr}"
        return jsonify({
            'service': 'SAMO Emotion Detection API',
            'status': 'operational',
            'version': '2.0.0-secure',
            'security': 'enabled',
            'rate_limit': RATE_LIMIT_PER_MINUTE,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Root endpoint error for {request.remote_addr}: {stre}")
        return create_error_response'Internal server error', 500

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
main_ns = Namespace'api', description='Main API operations'  # Removed leading slash to avoid double slashes
admin_ns = Namespace('/admin', description='Admin operations', authorizations={
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-Key'
    }
})

# Add namespaces to API
api.add_namespacemain_ns
api.add_namespaceadmin_ns

# Define request/response models for Swagger
text_input_model = api.model('TextInput', {
    'text': fields.Stringrequired=True, description='Text to analyze for emotion', example='I am feeling happy today!'
})

emotion_response_model = api.model('EmotionResponse', {
    'text': fields.Stringdescription='Input text',
    'emotions': fields.List(fields.Nested(api.model('Emotion', {
        'emotion': fields.Stringdescription='Emotion label',
        'confidence': fields.Floatdescription='Confidence score'
    }))),
    'confidence': fields.Floatdescription='Overall confidence',
    'request_id': fields.Stringdescription='Unique request identifier',
    'timestamp': fields.Floatdescription='Unix timestamp'
})

batch_input_model = api.model('BatchInput', {
    'texts': fields.Listfields.String, required=True, description='List of texts to analyze', example=['I am happy', 'I am sad']
})

batch_response_model = api.model('BatchResponse', {
    'results': fields.List(fields.Nestedemotion_response_model)
})

error_model = api.model('Error', {
    'error': fields.Stringdescription='Error message',
    'status_code': fields.Integerdescription='HTTP status code',
    'request_id': fields.Stringdescription='Unique request identifier',
    'timestamp': fields.Floatdescription='Unix timestamp'
})

# Security configuration from environment variables
ADMIN_API_KEY = os.environ.get"ADMIN_API_KEY"
if not ADMIN_API_KEY:
    raise ValueError"ADMIN_API_KEY environment variable must be set"
MAX_INPUT_LENGTH = int(os.environ.get"MAX_INPUT_LENGTH", "512")
RATE_LIMIT_PER_MINUTE = int(os.environ.get"RATE_LIMIT_PER_MINUTE", "100")
MODEL_PATH = os.environ.get"MODEL_PATH", "/app/model"
PORT = int(os.environ.get"PORT", "8080")

# Global variables for model state thread-safe with locks
model = None
tokenizer = None
emotion_mapping = None
model_loading = False
model_loaded = False
model_lock = threading.Lock()

# Emotion mapping based on training order
EMOTION_MAPPING = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

def require_api_keyf:
    """Decorator to require API key via X-API-Key header"""
    @wrapsf
    def decorated_function*args, **kwargs:
        api_key = request.headers.get'X-API-Key'
        if not verify_api_keyapi_key:
            logger.warningf"Invalid API key attempt from {request.remote_addr}"
            return create_error_response'Unauthorized - Invalid API key', 401
        return f*args, **kwargs
    return decorated_function

def verify_api_keyapi_key: str -> bool:
    """Verify API key using constant-time comparison"""
    if not api_key:
        return False
    return hmac.compare_digestapi_key, ADMIN_API_KEY

def sanitize_inputtext: str -> str:
    """Sanitize input text"""
    if not isinstancetext, str:
        raise ValueError"Input must be a string"

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '', '', '{{', '}}']
    for char in dangerous_chars:
        text = text.replacechar, ''

    # Limit length
    if lentext > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text.strip()

def load_model():
    """Load the emotion detection model using shared utilities"""
    # Use the shared model loading function
    success = ensure_model_loaded()
    if not success:
        logger.error"❌ Model loading failed"
        raise RuntimeError"Model loading failed - check logs for details"

def predict_emotiontext: str -> dict:
    """Predict emotion for given text using shared utilities"""
    # Use shared prediction function
    result = predict_emotionstext

    # Add request ID for tracking
    result['request_id'] = str(uuid.uuid4())

    return result

def check_model_loaded():
    """Ensure model is loaded before processing requests"""
    # Use shared model loading function
    return ensure_model_loaded()

def create_error_responseerror_message: str, status_code: int:
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
    logger.warningf"Rate limit exceeded for {request.remote_addr}"
    return create_error_response'Rate limit exceeded - too many requests', 429

def log_rate_limit_info():
    """Log rate limiting information for debugging"""
    logger.debugf"Rate limiting configured: {RATE_LIMIT_PER_MINUTE} requests per minute"
    logger.debugf"Current request from: {request.remote_addr}"

@app.before_request
def before_request():
    """Add request ID and timing to all requests"""
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())
    
    # Lazy model initialization on first request
    if not check_model_loaded():
        logger.info"🔄 Lazy initializing model on first request..."
        initialize_model()
    
    # Log incoming requests for debugging
    logger.info(f"📥 Request: {request.method} {request.path} from {request.remote_addr} ID: {g.request_id}")
    
    # Log request headers for debugging excluding sensitive ones
    headers_to_log = {k: v for k, v in request.headers.items() 
                      if k.lower() not in ['authorization', 'x-api-key', 'cookie']}
    logger.debugf"📋 Request headers: {headers_to_log}"

@app.after_request
def after_requestresponse:
    """Add request tracking headers"""
    if hasattrg, 'start_time':
        duration = time.time() - g.start_time
        response.headers['X-Request-Duration'] = strduration
    if hasattrg, 'request_id':
        response.headers['X-Request-ID'] = g.request_id
    
    # Log response for debugging
    logger.info(f"📤 Response: {response.status_code} for {request.method} {request.path} "
                f"from {request.remote_addr} ID: {g.request_id}, Duration: {duration:.3f}s")
    
    return response



@main_ns.route'/health'
class HealthResource:
    @api.doc'get_health'
    @api.response200, 'Success'
    @api.response503, 'Service Unavailable', error_model
    @api.response500, 'Internal Server Error', error_model
    def getself:
        """Get API health status"""
        try:
            logger.infof"Health check from {request.remote_addr}"
            model_status = check_model_loaded()
            
            if model_status:
                logger.info"Health check passed - model is ready"
                return {
                    'status': 'healthy',
                    'model_loaded': model_status,
                    'model_loading': False,
                    'port': PORT,
                    'timestamp': time.time()
                }
            else:
                logger.warning"Health check failed - model not ready"
                return create_error_response'Service unavailable - model not ready', 503
                
        except Exception as e:
            logger.error(f"Health check error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

@main_ns.route'/predict'
class PredictResource:
    @api.doc'post_predict', security='apikey'
    @api.expecttext_input_model, validate=True
    @api.response200, 'Success', emotion_response_model
    @api.response400, 'Bad Request', error_model
    @api.response401, 'Unauthorized', error_model
    @api.response429, 'Too Many Requests', error_model
    @api.response503, 'Service Unavailable', error_model
    @rate_limitRATE_LIMIT_PER_MINUTE
    @require_api_key
    def postself:
        """Predict emotion for a single text input"""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()
            
            # Get and validate input
            data = request.get_json()
            if not data or 'text' not in data:
                logger.warningf"Missing text field in request from {request.remote_addr}"
                return create_error_response'Missing text field', 400

            text = data['text']
            if not text or not isinstancetext, str:
                logger.warning(f"Invalid text input from {request.remote_addr}: {typetext}")
                return create_error_response'Text must be a non-empty string', 400

            # Sanitize input
            try:
                text = sanitize_inputtext
            except ValueError as e:
                logger.warning(f"Input sanitization failed for {request.remote_addr}: {stre}")
                return create_error_response(stre, 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error"Model not ready for prediction request"
                return create_error_response'Model not ready', 503

            # Predict emotion
            logger.infof"Processing prediction request for {request.remote_addr}"
            result = predict_emotiontext
            return result

        except Exception as e:
            logger.error(f"Prediction error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

@main_ns.route'/predict_batch'
class PredictBatchResource:
    @api.doc'post_predict_batch', security='apikey'
    @api.expectbatch_input_model, validate=True
    @api.response200, 'Success', batch_response_model
    @api.response400, 'Bad Request', error_model
    @api.response401, 'Unauthorized', error_model
    @api.response429, 'Too Many Requests', error_model
    @api.response503, 'Service Unavailable', error_model
    @rate_limitRATE_LIMIT_PER_MINUTE
    @require_api_key
    def postself:
        """Predict emotions for multiple text inputs"""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()
            
            # Get and validate input
            data = request.get_json()
            if not data or 'texts' not in data:
                logger.warningf"Missing texts field in batch request from {request.remote_addr}"
                return create_error_response'Missing texts field', 400

            texts = data['texts']
            if not isinstancetexts, list or lentexts == 0:
                logger.warning(f"Invalid texts input from {request.remote_addr}: {typetexts}")
                return create_error_response'Texts must be a non-empty list', 400

            if lentexts > 100:  # Limit batch size
                logger.warning(f"Batch size too large from {request.remote_addr}: {lentexts}")
                return create_error_response('Batch size too large max 100', 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error"Model not ready for batch prediction request"
                return create_error_response'Model not ready', 503

            # Process each text
            logger.info(f"Processing batch prediction request for {request.remote_addr} with {lentexts} texts")
            results = []
            for text in texts:
                if not text or not isinstancetext, str:
                    continue
                
                try:
                    text = sanitize_inputtext
                    result = predict_emotiontext
                    results.appendresult
                except Exception as e:
                    logger.warning(f"Failed to process text in batch from {request.remote_addr}: {stre}")
                    continue

            return {'results': results}

        except Exception as e:
            logger.error(f"Batch prediction error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

@main_ns.route'/emotions'
class EmotionsResource:
    @api.doc'get_emotions'
    @api.response200, 'Success'
    @api.response500, 'Internal Server Error', error_model
    def getself:
        """Get list of supported emotions"""
        try:
            logger.infof"Emotions list requested from {request.remote_addr}"
            return {
                'emotions': EMOTION_MAPPING,
                'count': lenEMOTION_MAPPING,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Emotions endpoint error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

# Admin endpoints
@admin_ns.route'/model_status'
class ModelStatusResource:
    @api.doc'get_model_status', security='apikey'
    @api.response200, 'Success'
    @api.response401, 'Unauthorized', error_model
    @api.response500, 'Internal Server Error', error_model
    @require_api_key
    def getself:
        """Get detailed model status admin only"""
        try:
            # Get model status from shared utilities
            logger.infof"Admin model status request from {request.remote_addr}"
            status = get_model_status()
            return status
        except Exception as e:
            logger.error(f"Model status error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

@admin_ns.route'/security_status'
class SecurityStatusResource:
    @api.doc'get_security_status', security='apikey'
    @api.response200, 'Success'
    @api.response401, 'Unauthorized', error_model
    @api.response500, 'Internal Server Error', error_model
    @require_api_key
    def getself:
        """Get security configuration status admin only"""
        try:
            logger.infof"Admin security status request from {request.remote_addr}"
            return {
                'api_key_protection': True,
                'input_sanitization': True,
                'rate_limiting': True,
                'request_tracking': True,
                'security_headers': True,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Security status error for {request.remote_addr}: {stre}")
            return create_error_response'Internal server error', 500

# Error handlers for Flask-RESTX - using direct registration due to decorator compatibility issue
def rate_limit_exceedederror:
    """Handle rate limit exceeded errors"""
    logger.warningf"Rate limit exceeded for {request.remote_addr}"
    return create_error_response'Rate limit exceeded - too many requests', 429

def internal_errorerror:
    """Handle internal server errors"""
    logger.error(f"Internal server error for {request.remote_addr}: {strerror}")
    return create_error_response'Internal server error', 500

def not_founderror:
    """Handle not found errors"""
    logger.warningf"Endpoint not found for {request.remote_addr}: {request.url}"
    return create_error_response'Endpoint not found', 404

def method_not_allowederror:
    """Handle method not allowed errors"""
    logger.warningf"Method not allowed for {request.remote_addr}: {request.method} {request.url}"
    return create_error_response'Method not allowed', 405

def handle_unexpected_errorerror:
    """Handle any unexpected errors"""
    logger.error(f"Unexpected error for {request.remote_addr}: {strerror}")
    return create_error_response'An unexpected error occurred', 500

# Register error handlers directly
api.error_handlers[429] = rate_limit_exceeded
api.error_handlers[500] = internal_error
api.error_handlers[404] = not_found
api.error_handlers[405] = method_not_allowed
api.error_handlers[Exception] = handle_unexpected_error

def initialize_model():
    """Initialize the emotion detection model"""
    try:
        logger.info"🚀 Initializing emotion detection API server..."
        logger.infof"📊 Configuration: MAX_INPUT_LENGTH={MAX_INPUT_LENGTH}, RATE_LIMIT={RATE_LIMIT_PER_MINUTE}/min"
        logger.info"🔐 Security: API key protection enabled, Admin API key configured"
        logger.infof"🌐 Server: Port {PORT}, Model path: {MODEL_PATH}"
        logger.infof"🔄 Rate limiting: {RATE_LIMIT_PER_MINUTE} requests per minute"
        
        # Load the emotion detection model
        logger.info"🔄 Loading emotion detection model..."
        load_model()
        logger.info"✅ Model initialization completed successfully"
        logger.info"🚀 API server ready to handle requests"
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API server: {stre}")
        raise

# Initialize model when the application starts
if __name__ == '__main__':
    initialize_model()
    logger.infof"🌐 Starting Flask development server on port {PORT}"
    app.runhost='0.0.0.0', port=PORT, debug=False
else:
    # For production deployment - don't initialize during import
    # Model will be initialized when the app actually starts
    logger.info"🚀 Production deployment detected - model will be initialized on first request"

# Root endpoint is now registered BEFORE Flask-RESTX initialization to avoid conflicts

# Make Flask app available to Gunicorn

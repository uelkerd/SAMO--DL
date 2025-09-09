#!/usr/bin/env python3
"""🚀 SECURE EMOTION DETECTION API FOR CLOUD RUN.
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
from werkzeug.datastructures import FileStorage
from functools import wraps

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

# Import shared model utilities
from model_utils import (
    ensure_model_loaded, predict_emotions, get_model_status,
)

# Import T5 and Whisper models
T5_AVAILABLE = False
WHISPER_AVAILABLE = False

# Set up logger for import errors
# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.models.summarization.t5_summarizer import create_t5_summarizer
    T5_AVAILABLE = True
except ImportError as e:
    logger.warning("T5 summarization not available: %s", e)
    T5_AVAILABLE = False

try:
    from src.models.voice_processing.whisper_transcriber import create_whisper_transcriber
    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning("Whisper transcription not available: %s", e)
    WHISPER_AVAILABLE = False

# Temporary file cleanup utility
def cleanup_temp_file(file_path) -> None:
    """Safely delete temporary file with error logging."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Successfully deleted temporary file: %s", file_path)
    except OSError:
        logger.exception("Failed to delete temporary file %s", file_path)

def normalize_emotion_results(raw_emotion):
    """Convert raw emotion prediction results to normalized format."""
    if not raw_emotion or 'emotions' not in raw_emotion:
        return {
            'emotions': {'neutral': 1.0},
            'primary_emotion': 'neutral',
            'confidence': 1.0,
            'emotional_intensity': 'neutral'
        }

    emotions = raw_emotion.get('emotions', [])
    if not emotions:
        return {
            'emotions': {'neutral': 1.0},
            'primary_emotion': 'neutral',
            'confidence': 1.0,
            'emotional_intensity': 'neutral'
        }

    # Convert list format to dict format
    emotion_dict = {}
    for emotion in emotions:
        emotion_dict[emotion['emotion']] = emotion['confidence']

    # Get primary emotion (highest confidence)
    primary_emotion = max(emotions, key=lambda e: e['confidence'])['emotion'] if emotions else 'neutral'
    confidence = raw_emotion.get('confidence', 0.0)

    # Determine emotional intensity
    if confidence > 0.8:
        intensity = 'high'
    elif confidence > 0.5:
        intensity = 'medium'
    else:
        intensity = 'low'

    return {
        'emotions': emotion_dict,
        'primary_emotion': primary_emotion,
        'confidence': confidence,
        'emotional_intensity': intensity
    }

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constants
MAX_AUDIO_FILE_SIZE_MB = 45

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024

# Add security headers
add_security_headers(app)

# Global model instances for T5 and Whisper
t5_summarizer = None
whisper_transcriber = None

def initialize_advanced_models() -> None:
    """Initialize T5 and Whisper models if available (only if not already loaded)."""
    global t5_summarizer, whisper_transcriber, T5_AVAILABLE, WHISPER_AVAILABLE

    # Initialize T5 model
    if T5_AVAILABLE and t5_summarizer is None:
        try:
            logger.info("Loading T5 summarization model (fallback)...")
            t5_summarizer = create_t5_summarizer("t5-small")
            logger.info("✅ T5 summarization model loaded")
        except Exception:
            logger.exception("❌ Failed to load T5 summarizer")
            T5_AVAILABLE = False

    # Initialize Whisper model
    if WHISPER_AVAILABLE and whisper_transcriber is None:
        try:
            logger.info("Loading Whisper transcription model (fallback)...")
            whisper_transcriber = create_whisper_transcriber("base")
            logger.info("✅ Whisper transcription model loaded")
        except Exception:
            logger.exception("❌ Failed to load Whisper transcriber")
            WHISPER_AVAILABLE = False

def load_all_models() -> None:
    """Consolidated model loading function for all AI models."""
    global t5_summarizer, whisper_transcriber, T5_AVAILABLE, WHISPER_AVAILABLE

    logger.info("🔄 Loading all AI models...")

    # Load emotion detection model
    try:
        load_model()
        logger.info("✅ Emotion detection model loaded")
    except Exception:
        logger.exception("❌ Failed to load emotion detection model")
        raise

    # Load T5 summarization model
    if T5_AVAILABLE and t5_summarizer is None:
        try:
            logger.info("🔄 Loading T5 summarization model...")
            t5_summarizer = create_t5_summarizer("t5-small")
            logger.info("✅ T5 summarization model loaded")
        except Exception:
            logger.exception("❌ Failed to load T5 summarizer")
            T5_AVAILABLE = False

    # Load Whisper transcription model
    if WHISPER_AVAILABLE and whisper_transcriber is None:
        try:
            logger.info("🔄 Loading Whisper transcription model...")
            whisper_transcriber = create_whisper_transcriber("base")
            logger.info("✅ Whisper transcription model loaded")
        except Exception:
            logger.exception("❌ Failed to load Whisper transcriber")
            WHISPER_AVAILABLE = False

    logger.info("✅ All available models loaded successfully")

# Load all models at startup
load_all_models()

# Register root endpoint BEFORE Flask-RESTX initialization to avoid conflicts
@app.route('/')
def home():  # Changed from api_root to home to avoid conflict with Flask-RESTX's root
    """Get API status and information."""
    try:
        logger.info("Root endpoint accessed from %s", request.remote_addr)
        return jsonify({
            'service': 'SAMO Emotion Detection API',
            'status': 'operational',
            'version': '2.0.0-secure',
            'security': 'enabled',
            'rate_limit': RATE_LIMIT_PER_MINUTE,
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error("Root endpoint error for %s: %s", request.remote_addr, e)
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
main_ns = Namespace('api', description='Main API operations')
admin_ns = Namespace('admin', description='Admin operations', authorizations={
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
    logger.warning("ADMIN_API_KEY environment variable not set - using default for development")
    ADMIN_API_KEY = "dev-admin-key-123"  # Default for development
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "512"))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "5000"))
MAX_AUDIO_FILE_SIZE_MB = int(os.environ.get("MAX_AUDIO_FILE_SIZE_MB", "45"))
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
    """Decorator to require API key via X-API-Key header."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not verify_api_key(api_key):
            logger.warning("Invalid API key attempt from %s", request.remote_addr)
            return create_error_response('Unauthorized - Invalid API key', 401)
        return f(*args, **kwargs)
    return decorated_function

def verify_api_key(api_key: str) -> bool:
    """Verify API key using constant-time comparison."""
    if not api_key:
        return False
    return hmac.compare_digest(api_key, ADMIN_API_KEY)

def sanitize_input(text: str) -> str:
    """Sanitize input text."""
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

def load_model() -> None:
    """Load the emotion detection model using shared utilities."""
    # Use the shared model loading function
    success = ensure_model_loaded()
    if not success:
        logger.error("❌ Model loading failed")
        raise RuntimeError("Model loading failed - check logs for details")

def predict_emotion(text: str) -> dict:
    """Predict emotion for given text using shared utilities."""
    # Use shared prediction function
    result = predict_emotions(text)

    # Add request ID for tracking
    result['request_id'] = str(uuid.uuid4())

    return result

def check_model_loaded():
    """Ensure model is loaded before processing requests."""
    # Use shared model loading function
    return ensure_model_loaded()

def create_error_response(error_message: str, status_code: int):
    """Create a properly formatted error response for Flask-RESTX."""
    error_response = {
        'error': error_message,
        'status_code': status_code,
        'request_id': str(uuid.uuid4()),
        'timestamp': time.time()
    }
    return error_response, status_code

def handle_rate_limit_exceeded():
    """Handle rate limit exceeded - return proper error response."""
    logger.warning("Rate limit exceeded for %s", request.remote_addr)
    return create_error_response('Rate limit exceeded - too many requests', 429)

def log_rate_limit_info() -> None:
    """Log rate limiting information for debugging."""
    logger.debug("Rate limiting configured: %s requests per minute", RATE_LIMIT_PER_MINUTE)
    logger.debug("Current request from: %s", request.remote_addr)

@app.before_request
def before_request() -> None:
    """Add request ID and timing to all requests."""
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())
    
    # Lazy model initialization on first request
    if not check_model_loaded():
        logger.info("🔄 Lazy initializing model on first request...")
        initialize_model()
    
    # Log incoming requests for debugging
    logger.info("📥 Request: %s %s from %s (ID: %s)", request.method, request.path, request.remote_addr, g.request_id)
    
    # Log request headers for debugging (excluding sensitive ones)
    headers_to_log = {k: v for k, v in request.headers.items()
                      if k.lower() not in ['authorization', 'x-api-key', 'cookie']}
    logger.debug(f"📋 Request headers: {headers_to_log}")

@app.after_request
def after_request(response):
    """Add request tracking headers."""
    duration = 0.0
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        response.headers['X-Request-Duration'] = str(duration)
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    # Log response for debugging
    summary = getattr(request, 'summary', None)
    logger.info("📤 Response: %s for %s %s from %s (ID: %s, Duration: %.3fs, Summary: %s)", response.status_code, request.method, request.path, request.remote_addr, g.request_id, duration, summary if summary else 'None')
    
    return response



@main_ns.route('/health')
class Health(Resource):
    @api.doc('get_health')
    @api.response(200, 'Success')
    @api.response(503, 'Service Unavailable')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Get API health status."""
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
            logger.error("Health check error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

@main_ns.route('/predict')
class Predict(Resource):
    @api.doc('post_predict', security='apikey')
    @api.expect(text_input_model, validate=True)
    @api.response(200, 'Success', emotion_response_model)
    @api.response(400, 'Bad Request')
    @api.response(401, 'Unauthorized')
    @api.response(429, 'Too Many Requests')
    @api.response(503, 'Service Unavailable')
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post():
        """Predict emotion for a single text input."""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()
            
            # Get and validate input
            data = request.get_json()
            if not data or 'text' not in data:
                logger.warning("Missing text field in request from %s", request.remote_addr)
                return create_error_response('Missing text field', 400)

            text = data['text']
            if not text or not isinstance(text, str):
                logger.warning("Invalid text input from %s: %s", request.remote_addr, type(text))
                return create_error_response('Text must be a non-empty string', 400)

            # Sanitize input
            try:
                text = sanitize_input(text)
            except ValueError as e:
                logger.warning("Input sanitization failed for %s: %s", request.remote_addr, e)
                return create_error_response(str(e), 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error("Model not ready for prediction request")
                return create_error_response('Model not ready', 503)

            # Predict emotion
            logger.info("Processing prediction request for %s", request.remote_addr)
            result = predict_emotion(text)
            return result

        except Exception as e:
            logger.error("Prediction error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

@main_ns.route('/predict_batch')
class PredictBatch(Resource):
    @api.doc('post_predict_batch', security='apikey')
    @api.expect(batch_input_model, validate=True)
    @api.response(200, 'Success', batch_response_model)
    @api.response(400, 'Bad Request')
    @api.response(401, 'Unauthorized')
    @api.response(429, 'Too Many Requests')
    @api.response(503, 'Service Unavailable')
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Predict emotions for multiple text inputs."""
        try:
            # Log rate limiting info for debugging
            log_rate_limit_info()
            
            # Get and validate input
            data = request.get_json()
            if not data or 'texts' not in data:
                logger.warning("Missing texts field in batch request from %s", request.remote_addr)
                return create_error_response('Missing texts field', 400)

            texts = data['texts']
            if not isinstance(texts, list) or len(texts) == 0:
                logger.warning("Invalid texts input from %s: %s", request.remote_addr, type(texts))
                return create_error_response('Texts must be a non-empty list', 400)

            if len(texts) > 100:  # Limit batch size
                logger.warning("Batch size too large from %s: %s", request.remote_addr, len(texts))
                return create_error_response('Batch size too large (max 100)', 400)

            # Ensure model is loaded
            if not check_model_loaded():
                logger.error("Model not ready for batch prediction request")
                return create_error_response('Model not ready', 503)

            # Process each text
            logger.info("Processing batch prediction request for %s with %s texts", request.remote_addr, len(texts))
            results = []
            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                
                try:
                    text = sanitize_input(text)
                    result = predict_emotion(text)
                    results.append(result)
                except Exception as e:
                    logger.warning("Failed to process text in batch from %s: %s", request.remote_addr, e)
                    continue

            return {'results': results}

        except Exception as e:
            logger.error("Batch prediction error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

@main_ns.route('/emotions')
class Emotions(Resource):
    @api.doc('get_emotions')
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Get list of supported emotions."""
        try:
            logger.info("Emotions list requested from %s", request.remote_addr)
            return {
                'emotions': EMOTION_MAPPING,
                'count': len(EMOTION_MAPPING),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error("Emotions endpoint error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

# Admin endpoints
@admin_ns.route('/model_status')
class ModelStatus(Resource):
    @api.doc('get_model_status', security='apikey')
    @api.response(200, 'Success')
    @api.response(401, 'Unauthorized')
    @api.response(500, 'Internal Server Error')
    @require_api_key
    def get(self):
        """Get detailed model status (admin only)."""
        try:
            # Get model status from shared utilities
            logger.info("Admin model status request from %s", request.remote_addr)
            status = get_model_status()
            return status
        except Exception as e:
            logger.error("Model status error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

@admin_ns.route('/security_status')
class SecurityStatus(Resource):
    @api.doc('get_security_status', security='apikey')
    @api.response(200, 'Success')
    @api.response(401, 'Unauthorized')
    @api.response(500, 'Internal Server Error')
    @require_api_key
    def get(self):
        """Get security configuration status (admin only)."""
        try:
            logger.info("Admin security status request from %s", request.remote_addr)
            return {
                'api_key_protection': True,
                'input_sanitization': True,
                'rate_limiting': True,
                'request_tracking': True,
                'security_headers': True,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error("Security status error for %s: %s", request.remote_addr, e)
            return create_error_response('Internal server error', 500)

# Error handlers for Flask-RESTX - using direct registration due to decorator compatibility issue
def rate_limit_exceeded(error):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded for %s", request.remote_addr)
    return create_error_response('Rate limit exceeded - too many requests', 429)

def internal_error(error):
    """Handle internal server errors."""
    logger.error("Internal server error for %s: %s", request.remote_addr, error)
    return create_error_response('Internal server error', 500)

def not_found(error):
    """Handle not found errors."""
    logger.warning("Endpoint not found for %s: %s", request.remote_addr, request.url)
    return create_error_response('Endpoint not found', 404)

def method_not_allowed(error):
    """Handle method not allowed errors."""
    logger.warning("Method not allowed for %s: %s %s", request.remote_addr, request.method, request.url)
    return create_error_response('Method not allowed', 405)

def handle_unexpected_error(error):
    """Handle any unexpected errors."""
    logger.error("Unexpected error for %s: %s", request.remote_addr, error)
    return create_error_response('An unexpected error occurred', 500)

# Register error handlers directly
api.error_handlers[429] = rate_limit_exceeded
api.error_handlers[500] = internal_error
api.error_handlers[404] = not_found
api.error_handlers[405] = method_not_allowed
api.error_handlers[Exception] = handle_unexpected_error

# ===== ADVANCED ENDPOINTS: Summarization and Transcription =====





@api.route('/summarize')
class Summarize(Resource):
    """Text summarization endpoint."""

    @api.doc('summarize_text')
    @api.expect(api.model('SummarizeRequest', {
        'text': fields.String(required=True, description='Text to summarize', example='This is a long text that needs to be summarized...'),
        'max_length': fields.Integer(default=150, description='Maximum summary length'),
        'min_length': fields.Integer(default=30, description='Minimum summary length')
    }))
    # Temporarily removed @api.marshal_with to debug
    # @api.marshal_with(api.model('SummarizeResponse', {
    #     'summary': fields.String(description='Generated summary'),
    #     'original_length': fields.Integer(description='Original text length'),
    #     'summary_length': fields.Integer(description='Summary length'),
    #     'compression_ratio': fields.Float(description='Compression ratio'),
    #     'processing_time': fields.Float(description='Processing time in seconds')
    # }))
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Summarize text using T5 model."""
        logger.info("📥 Summarization request received")
        logger.info("T5_AVAILABLE: %s, t5_summarizer: %s", T5_AVAILABLE, t5_summarizer is not None)

        if not T5_AVAILABLE or t5_summarizer is None:
            logger.error("T5 summarization service unavailable")
            api.abort(503, "Text summarization service unavailable")

        start_time = time.time()
        data = request.get_json()
        logger.info("Request data: %s", data)

        if not data or 'text' not in data:
            api.abort(400, "Text field is required")

        text = data['text'].strip()
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 30)
        logger.info(
            f"Text length: {len(text)}, max_length: {max_length}, "
            f"min_length: {min_length}"
        )

        if not text:
            api.abort(400, "Text cannot be empty")

        if len(text) > MAX_TEXT_LENGTH:
            api.abort(400, f"Text too long (max {MAX_TEXT_LENGTH} characters)")

        try:
            logger.info("🔄 Starting T5 summarization...")
            summary = t5_summarizer.generate_summary(
                text, max_length=max_length, min_length=min_length
            )
            logger.info("✅ T5 summarization completed: %s", summary[:100])

            original_length = len(text.split())
            summary_length = len(summary.split()) if summary else 0
            compression_ratio = (
                1 - (summary_length / original_length)
                if original_length > 0 else 0
            )

            result = {
                'summary': summary,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'processing_time': time.time() - start_time
            }
            logger.info("📤 Summarization result: %s", result)
            return result

        except Exception:
            logger.exception("❌ Summarization failed")
            api.abort(500, "Summarization failed")


@api.route('/transcribe')
class Transcribe(Resource):
    """Voice transcription endpoint."""

    @api.doc('transcribe_audio')
    @api.expect(api.parser()
        .add_argument(
            'audio', type=FileStorage, location='files', required=True,
            help='Audio file to transcribe (MP3, WAV, M4A)'
        )
        .add_argument(
            'language', type=str, location='form',
            help='Language code (optional)'
        )
        .add_argument(
            'model_size', type=str, location='form', default='base',
            help='Whisper model size (tiny, base, small, medium, large)'
        ))
    @api.marshal_with(api.model('TranscriptionResponse', {
        'text': fields.String(description='Transcribed text'),
        'language': fields.String(description='Detected language'),
        'confidence': fields.Float(description='Transcription confidence'),
        'duration': fields.Float(description='Audio duration in seconds'),
        'processing_time': fields.Float(description='Processing time in seconds'),
        'word_count': fields.Integer(description='Number of words'),
        'speaking_rate': fields.Float(description='Words per minute')
    }))
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Transcribe audio file to text using Whisper."""
        if not WHISPER_AVAILABLE or whisper_transcriber is None:
            api.abort(503, "Voice transcription service unavailable")

        start_time = time.time()

        # Parse form data
        if 'audio' not in request.files:
            api.abort(400, "Audio file is required")

        audio_file = request.files['audio']
        if not audio_file.filename:
            api.abort(400, "No audio file selected")

        # Validate file type with logging
        allowed_extensions = {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'}
        if '.' not in audio_file.filename:
            logger.warning("No extension in filename %s, rejecting", audio_file.filename)
            api.abort(400, "File must have a valid audio extension")
        ext = audio_file.filename.rsplit('.', 1)[1].lower()
        if ext not in allowed_extensions:
            logger.warning("Unsupported extension %s in filename %s, rejecting", ext, audio_file.filename)
            api.abort(
                400,
                f"Unsupported file type: .{ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        logger.info("File validation passed for %s (ext: %s)", audio_file.filename, ext)

        # Check file size (max 45MB)
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        if file_size > MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024:
            api.abort(400, f"File too large (max {MAX_AUDIO_FILE_SIZE_MB}MB)")

        try:
            import tempfile
            allowed_extensions = {'mp3','wav','m4a','aac','ogg','flac'}
            # Select only from allowlisted extensions, ignoring user-provided value if not allowed.
            ext = 'wav'  # default
            _, ext_candidate = os.path.splitext(audio_file.filename)
            if ext_candidate in allowed_extensions:
                ext = ext_candidate
            else:
                logger.warning("Extension '%s' in filename '%s' not in allowed set %s; defaulting to .wav", ext_candidate, audio_file.filename, allowed_extensions)
            logger.info("Using validated extension: .%s for temp file", ext)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f'.{ext}'
            ) as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # Transcribe
                language = request.form.get('language')
                result = whisper_transcriber.transcribe(temp_path, language=language)

                # Extract result data
                transcription_text = (
                    result.text if hasattr(result, 'text') else str(result)
                )
                language_detected = getattr(result, 'language', 'unknown')
                confidence = getattr(result, 'confidence', 0.0)
                duration = getattr(result, 'duration', 0.0)
                word_count = len(transcription_text.split())
                speaking_rate = word_count / (duration / 60) if duration > 0 else 0

                return {
                    'text': transcription_text,
                    'language': language_detected,
                    'confidence': confidence,
                    'duration': duration,
                    'processing_time': time.time() - start_time,
                    'word_count': word_count,
                    'speaking_rate': speaking_rate
                }

            finally:
                # Cleanup temporary file
                cleanup_temp_file(temp_path)

        except (OSError, RuntimeError, ValueError) as e:
            logger.exception(f"Transcription failed: {e}")
            api.abort(500, "Transcription failed")


@api.route('/analyze/complete')
class CompleteAnalysis(Resource):
    """Complete analysis endpoint combining all AI models."""

    def _process_transcription(self, audio_file):
        """Process audio transcription if provided."""
        logger.info("🔄 Processing audio transcription...")
        import tempfile
        allowed_extensions = {'mp3': 'mp3', 'wav': 'wav', 'm4a': 'm4a', 'aac': 'aac', 'ogg': 'ogg', 'flac': 'flac'}
        # Extract extension safely using os.path.splitext
        _, ext_candidate = os.path.splitext(audio_file.filename)
        ext = 'wav'  # default
        if ext_candidate:
            ext_candidate_clean = ext_candidate.lstrip('.').lower()
            ext = allowed_extensions.get(ext_candidate_clean, 'wav')
            if ext_candidate_clean != ext:
                logger.warning(f"Extension '{ext_candidate_clean}' in filename '{audio_file.filename}' not in allowed set; defaulting to .{ext}")
        else:
            logger.warning("No extension in filename %s, defaulting to .wav", audio_file.filename)
        logger.info("Using validated extension: .%s for temp file in complete analysis", ext)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f'.{ext}'
        ) as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            language = request.form.get('language')
            transcription_result = whisper_transcriber.transcribe(temp_path, language=language)
            text_to_analyze = (
                transcription_result.text
                if hasattr(transcription_result, 'text')
                else str(transcription_result)
            )
            logger.info("✅ Transcription completed: %s", text_to_analyze[:100])
            return {
                'text': text_to_analyze,
                'language': getattr(transcription_result, 'language', 'en'),
                'confidence': getattr(transcription_result, 'confidence', 0.95),
                'duration': getattr(transcription_result, 'duration', 0.0)
            }
        finally:
            cleanup_temp_file(temp_path)

    def _process_emotion(self, text_to_analyze):
        """Process emotion analysis."""
        logger.info("🔄 Processing emotion analysis...")
        try:
            raw_emotion = predict_emotions(text_to_analyze)
            emotion_result = normalize_emotion_results(raw_emotion)
            logger.info("✅ Emotion analysis: %s (%.2f)", emotion_result['primary_emotion'], emotion_result['confidence'])
            return emotion_result
        except Exception as e:
            logger.warning("Emotion analysis failed: %s", e)
            return {
                'emotions': {'neutral': 1.0},
                'primary_emotion': 'neutral',
                'confidence': 1.0,
                'emotional_intensity': 'neutral'
            }

    def _process_summary(self, text_to_analyze, emotion_result, generate_summary):
        """Process text summarization if requested."""
        logger.info("🔄 Processing text summarization...")
        summary_result = {}
        if generate_summary and T5_AVAILABLE and t5_summarizer is not None:
            try:
                summary_text = t5_summarizer.generate_summary(text_to_analyze)
                original_length = len(text_to_analyze.split())
                summary_length = len(summary_text.split())
                compression_ratio = (
                    1 - (summary_length / original_length)
                    if original_length > 0 else 0
                )

                # Determine emotional tone
                tone = "neutral"
                if emotion_result.get('primary_emotion') in [
                    'joy', 'gratitude', 'excitement'
                ]:
                    tone = "positive"
                elif emotion_result.get('primary_emotion') in [
                    'sadness', 'anger', 'fear'
                ]:
                    tone = "negative"

                summary_result = {
                    'summary': summary_text,
                    'compression_ratio': compression_ratio,
                    'emotional_tone': tone
                }
                logger.info("✅ Summarization completed: %.2f ratio", compression_ratio)
            except Exception as e:
                logger.warning("Summarization failed: %s", e)
        return summary_result

    @api.doc('analyze_complete')
    @api.expect(api.parser()
        .add_argument(
            'text', type=str, location='form',
            help='Text to analyze (optional if audio provided)'
        )
        .add_argument(
            'audio', type=FileStorage, location='files',
            help='Audio file to transcribe (optional if text provided)'
        )
        .add_argument(
            'language', type=str, location='form',
            help='Language code for transcription'
        )
        .add_argument(
            'generate_summary', type=bool, location='form', default=True,
            help='Whether to generate summary'
        ))
    @api.marshal_with(api.model('CompleteAnalysisResponse', {
        'transcription': fields.Nested(api.model('TranscriptionData', {
            'text': fields.String(),
            'language': fields.String(),
            'confidence': fields.Float(),
            'duration': fields.Float()
        })),
        'emotion_analysis': fields.Nested(api.model('EmotionData', {
            'emotions': fields.Raw(),
            'primary_emotion': fields.String(),
            'confidence': fields.Float(),
            'emotional_intensity': fields.String()
        })),
        'summary': fields.Nested(api.model('SummaryData', {
            'summary': fields.String(),
            'compression_ratio': fields.Float(),
            'emotional_tone': fields.String()
        })),
        'processing_time': fields.Float(),
        'pipeline_status': fields.Raw()
    }))
    @rate_limit(RATE_LIMIT_PER_MINUTE)
    @require_api_key
    def post(self):
        """Complete analysis pipeline: transcription + emotion + summarization."""
        start_time = time.time()
        pipeline_status = {
            'emotion_detection': True,
            'text_summarization': T5_AVAILABLE and t5_summarizer is not None,
            'voice_processing': WHISPER_AVAILABLE and whisper_transcriber is not None
        }

        text_to_analyze = request.form.get('text', '').strip()
        generate_summary = request.form.get('generate_summary', 'true').lower() == 'true'

        # Handle transcription if audio provided
        transcription_data = None
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                transcription_data = self._process_transcription(audio_file)
                text_to_analyze = transcription_data['text']

        if not text_to_analyze:
            api.abort(400, "Either text or audio file must be provided")

        # Process emotion and summary sequentially
        emotion_result = self._process_emotion(text_to_analyze)
        summary_result = self._process_summary(text_to_analyze, emotion_result, generate_summary)

        return {
            'transcription': transcription_data,
            'emotion_analysis': emotion_result,
            'summary': summary_result,
            'processing_time': time.time() - start_time,
            'pipeline_status': pipeline_status
        }


def initialize_model() -> None:
    """Initialize the emotion detection model."""
    try:
        logger.info("🚀 Initializing emotion detection API server...")
        logger.info("📊 Configuration: MAX_INPUT_LENGTH=%s, RATE_LIMIT=%s/min", MAX_INPUT_LENGTH, RATE_LIMIT_PER_MINUTE)
        logger.info("🔐 Security: API key protection enabled, Admin API key configured")
        logger.info("🌐 Server: Port %s, Model path: %s", PORT, MODEL_PATH)
        logger.info("🔄 Rate limiting: %s requests per minute", RATE_LIMIT_PER_MINUTE)

        # Load all models using consolidated function
        if os.environ.get("PRELOAD_MODELS", "1") == "1":
            try:
                load_all_models()
            except Exception as e:
                logger.exception(f"Failed to preload models: {e}")
                logger.info("Continuing without preloaded models")
        else:
            logger.info("Model preloading skipped (PRELOAD_MODELS=0)")

        logger.info("✅ Model initialization completed successfully")
        logger.info("🚀 API server ready to handle requests")

    except Exception:
        logger.exception("❌ Failed to initialize API server")
        raise

# Initialize models immediately when module is imported
logger.info("🚀 Initializing models during module import...")
try:
    initialize_model()
    logger.info("✅ Models loaded successfully during module import")
    MODELS_LOADED_AT_STARTUP = True
except Exception:
    logger.exception("❌ Failed to load models during module import")
    # Continue anyway - models will be loaded on first request if startup fails
    logger.info("⚠️ Continuing without pre-loaded models - will load on first request")
    MODELS_LOADED_AT_STARTUP = False

if __name__ == '__main__':
    logger.info("🌐 Starting Flask development server on port %s", PORT)
    app.run(host='0.0.0.0', port=PORT, debug=False)

# Root endpoint is now registered BEFORE Flask-RESTX initialization to avoid conflicts

# Make Flask app available to Gunicorn

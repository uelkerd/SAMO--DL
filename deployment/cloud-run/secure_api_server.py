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
from werkzeug.datastructures import FileStorage
from functools import wraps

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

# Import shared model utilities
from model_utils import (
    ensure_model_loaded, predict_emotions, get_model_status,
    validate_text_input,
)

# Import T5 and Whisper models
T5_AVAILABLE = False
WHISPER_AVAILABLE = False

try:
    from src.models.summarization.t5_summarizer import create_t5_summarizer
    T5_AVAILABLE = True
except ImportError as e:
    logger.warning(f"T5 summarization not available: {e}")

try:
    from src.models.voice_processing.whisper_transcriber import create_whisper_transcriber
    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Whisper transcription not available: {e}")

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add security headers
add_security_headers(app)

# Global model instances for T5 and Whisper
t5_summarizer = None
whisper_transcriber = None

def initialize_advanced_models():
    """Initialize T5 and Whisper models if available"""
    global t5_summarizer, whisper_transcriber

    if T5_AVAILABLE and t5_summarizer is None:
        try:
            logger.info("Loading T5 summarization model...")
            t5_summarizer = create_t5_summarizer("t5-small")
            logger.info("✅ T5 summarization model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load T5 summarizer: {e}")

    if WHISPER_AVAILABLE and whisper_transcriber is None:
        try:
            logger.info("Loading Whisper transcription model...")
            whisper_transcriber = create_whisper_transcriber("base")
            logger.info("✅ Whisper transcription model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper transcriber: {e}")

# Initialize advanced models at startup
initialize_advanced_models()

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

# ===== ADVANCED ENDPOINTS: Summarization and Transcription =====

@api.route('/summarize')
class Summarize(Resource):
    """Text summarization endpoint"""

    @api.doc('summarize_text')
    @api.expect(api.model('SummarizeRequest', {
        'text': fields.String(required=True, description='Text to summarize', example='This is a long text that needs to be summarized...'),
        'max_length': fields.Integer(default=150, description='Maximum summary length'),
        'min_length': fields.Integer(default=30, description='Minimum summary length')
    }))
    @api.marshal_with(api.model('SummarizeResponse', {
        'summary': fields.String(description='Generated summary'),
        'original_length': fields.Integer(description='Original text length'),
        'summary_length': fields.Integer(description='Summary length'),
        'compression_ratio': fields.Float(description='Compression ratio'),
        'processing_time': fields.Float(description='Processing time in seconds')
    }))
    @rate_limit
    @require_api_key
    def post(self):
        """Summarize text using T5 model"""
        if not T5_AVAILABLE or t5_summarizer is None:
            api.abort(503, "Text summarization service unavailable")

        start_time = time.time()
        data = request.get_json()

        if not data or 'text' not in data:
            api.abort(400, "Text field is required")

        text = data['text'].strip()
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 30)

        if not text:
            api.abort(400, "Text cannot be empty")

        if len(text) > 5000:
            api.abort(400, "Text too long (max 5000 characters)")

        try:
            summary = t5_summarizer.generate_summary(
                text, max_length=max_length, min_length=min_length
            )

            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0

            return {
                'summary': summary,
                'original_length': original_length,
                'summary_length': summary_length,
                'compression_ratio': compression_ratio,
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            api.abort(500, "Summarization failed")


@api.route('/transcribe')
class Transcribe(Resource):
    """Voice transcription endpoint"""

    @api.doc('transcribe_audio')
    @api.expect(api.parser()
        .add_argument('audio', type=FileStorage, location='files', required=True,
                     help='Audio file to transcribe (MP3, WAV, M4A)')
        .add_argument('language', type=str, location='form', help='Language code (optional)')
        .add_argument('model_size', type=str, location='form', default='base',
                     help='Whisper model size (tiny, base, small, medium, large)'))
    @api.marshal_with(api.model('TranscriptionResponse', {
        'text': fields.String(description='Transcribed text'),
        'language': fields.String(description='Detected language'),
        'confidence': fields.Float(description='Transcription confidence'),
        'duration': fields.Float(description='Audio duration in seconds'),
        'processing_time': fields.Float(description='Processing time in seconds'),
        'word_count': fields.Integer(description='Number of words'),
        'speaking_rate': fields.Float(description='Words per minute')
    }))
    @rate_limit
    @require_api_key
    def post(self):
        """Transcribe audio file to text using Whisper"""
        if not WHISPER_AVAILABLE or whisper_transcriber is None:
            api.abort(503, "Voice transcription service unavailable")

        start_time = time.time()

        # Parse form data
        if 'audio' not in request.files:
            api.abort(400, "Audio file is required")

        audio_file = request.files['audio']
        if not audio_file.filename:
            api.abort(400, "No audio file selected")

        # Validate file type
        allowed_extensions = {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'}
        if '.' not in audio_file.filename:
            api.abort(400, "File must have an extension")
        ext = audio_file.filename.rsplit('.', 1)[1].lower()
        if ext not in allowed_extensions:
            api.abort(400, f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}")

        # Check file size (max 45MB)
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        if file_size > 45 * 1024 * 1024:
            api.abort(400, "File too large (max 45MB)")

        try:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # Transcribe
                language = request.form.get('language')
                result = whisper_transcriber.transcribe(temp_path, language=language)

                # Extract result data
                transcription_text = result.text if hasattr(result, 'text') else str(result)
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
                import os
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            api.abort(500, "Transcription failed")


@api.route('/analyze/complete')
class CompleteAnalysis(Resource):
    """Complete analysis endpoint combining all AI models"""

    @api.doc('analyze_complete')
    @api.expect(api.parser()
        .add_argument('text', type=str, location='form', help='Text to analyze (optional if audio provided)')
        .add_argument('audio', type=FileStorage, location='files', help='Audio file to transcribe (optional if text provided)')
        .add_argument('language', type=str, location='form', help='Language code for transcription')
        .add_argument('generate_summary', type=bool, location='form', default=True, help='Whether to generate summary')
        .add_argument('emotion_threshold', type=float, location='form', default=0.1, help='Emotion detection threshold'))
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
    @rate_limit
    @require_api_key
    def post(self):
        """Complete analysis pipeline: transcription + emotion + summarization"""
        start_time = time.time()
        pipeline_status = {
            'emotion_detection': True,
            'text_summarization': T5_AVAILABLE and t5_summarizer is not None,
            'voice_processing': WHISPER_AVAILABLE and whisper_transcriber is not None
        }

        text_to_analyze = request.form.get('text', '').strip()
        generate_summary = request.form.get('generate_summary', 'true').lower() == 'true'
        emotion_threshold = float(request.form.get('emotion_threshold', 0.1))

        # Handle transcription if audio provided
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                # Use transcription endpoint logic
                import tempfile
                import os

                ext = audio_file.filename.rsplit('.', 1)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_file:
                    audio_file.save(temp_file.name)
                    temp_path = temp_file.name

                try:
                    language = request.form.get('language')
                    transcription_result = whisper_transcriber.transcribe(temp_path, language=language)
                    text_to_analyze = transcription_result.text if hasattr(transcription_result, 'text') else str(transcription_result)
                finally:
                    os.unlink(temp_path)

        if not text_to_analyze:
            api.abort(400, "Either text or audio file must be provided")

        # Emotion Analysis
        emotion_result = {}
        try:
            raw_emotion = predict_emotions(text_to_analyze, threshold=emotion_threshold)
            emotion_result = normalize_emotion_results(raw_emotion)
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            emotion_result = {
                'emotions': {'neutral': 1.0},
                'primary_emotion': 'neutral',
                'confidence': 1.0,
                'emotional_intensity': 'neutral'
            }

        # Text Summarization
        summary_result = {}
        if generate_summary and T5_AVAILABLE and t5_summarizer is not None:
            try:
                summary_text = t5_summarizer.generate_summary(text_to_analyze)
                original_length = len(text_to_analyze.split())
                summary_length = len(summary_text.split())
                compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0

                # Determine emotional tone
                tone = "neutral"
                if emotion_result.get('primary_emotion') in ['joy', 'gratitude', 'excitement']:
                    tone = "positive"
                elif emotion_result.get('primary_emotion') in ['sadness', 'anger', 'fear']:
                    tone = "negative"

                summary_result = {
                    'summary': summary_text,
                    'compression_ratio': compression_ratio,
                    'emotional_tone': tone
                }
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        return {
            'transcription': {
                'text': text_to_analyze,
                'language': 'en',  # Default assumption
                'confidence': 1.0 if 'audio' not in request.files else 0.95,
                'duration': 0.0  # Would need audio metadata
            } if 'audio' in request.files else None,
            'emotion_analysis': emotion_result,
            'summary': summary_result,
            'processing_time': time.time() - start_time,
            'pipeline_status': pipeline_status
        }


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
    app.run(host='0.0.0.0', port=PORT, debug=False)
else:
    # For production deployment - don't initialize during import
    # Model will be initialized when the app actually starts
    logger.info("🚀 Production deployment detected - model will be initialized on first request")

# Root endpoint is now registered BEFORE Flask-RESTX initialization to avoid conflicts

# Make Flask app available to Gunicorn

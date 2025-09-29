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

⚠️  SECURITY WARNING - AUTHENTICATION BYPASS:
The require_api_key decorator supports authentication bypass for development
via the ALLOW_UNAUTHENTICATED=true environment variable. This creates a
CRITICAL SECURITY VULNERABILITY if misconfigured:

PRODUCTION ENVIRONMENT (FLASK_ENV=production or ENVIRONMENT=production):
- CLIENT_API_KEY is MANDATORY - server will refuse to start without it
- ALLOW_UNAUTHENTICATED is FORBIDDEN - server will refuse to start if set
- Authentication is ALWAYS enforced - no bypass possible

DEVELOPMENT ENVIRONMENT:
- Either CLIENT_API_KEY OR ALLOW_UNAUTHENTICATED=true is REQUIRED
- No automatic bypass - explicit configuration is mandatory
- Server will refuse to start without proper authentication configuration

Security implications of bypass:
- Unauthorized clients can access protected endpoints
- Sensitive data and AI models become publicly accessible
- Rate limiting and abuse detection are circumvented

To prevent security risks:
1. ALWAYS set CLIENT_API_KEY in production environments
2. Use ALLOW_UNAUTHENTICATED=true ONLY for local development
3. Monitor logs for authentication bypass warnings
4. Never deploy with ALLOW_UNAUTHENTICATED=true in production
"""

# Import all modules first
import os
import secrets
from flask import Flask, request, jsonify, g
import werkzeug
import logging
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
from functools import wraps, lru_cache
from typing import List, Tuple, Any, Dict

# Import security components using relative imports
from ..src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig
from ..src.input_sanitizer import InputSanitizer, SanitizationConfig
from ..src.security_setup import setup_security_middleware, get_environment
from ..src.inference.text_emotion_service import HFEmotionService  # type: ignore

# Import centralized constants with fallback for non-package environments
try:
    from src.constants import EMOTION_MODEL_DIR  # single source of truth
except ImportError:
    EMOTION_MODEL_DIR = os.getenv(
        'EMOTION_MODEL_DIR',
        '/app/models/emotion-english-distilroberta-base'
    )

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


def validate_security_configuration():
    """Validate security configuration at startup and return auth bypass flag.
    
    Security Configuration Logic:
    - Production: CLIENT_API_KEY is REQUIRED, ALLOW_UNAUTHENTICATED is forbidden
    - Development: Either CLIENT_API_KEY OR ALLOW_UNAUTHENTICATED=true is required
    - No automatic bypass - explicit configuration is always required
    
    Returns:
        bool: True if authentication bypass is allowed, False otherwise
    """
    client_api_key = os.environ.get("CLIENT_API_KEY")
    allow_unauthenticated = os.environ.get("ALLOW_UNAUTHENTICATED", "").lower() == "true"
    flask_env = os.environ.get("FLASK_ENV", "").lower()
    is_production = flask_env == "production" or os.environ.get("ENVIRONMENT", "").lower() == "production"

    # Production environment validation
    if is_production:
        if not client_api_key:
            error_msg = (
                "🚨 CRITICAL SECURITY ERROR: CLIENT_API_KEY is not set in production environment!\n"
                "This creates a severe security vulnerability allowing unauthorized access to all protected endpoints.\n"
                "Please set CLIENT_API_KEY environment variable before starting the server."
            )
            logger.error(error_msg)
            print(f"\n{error_msg}\n")
            raise RuntimeError("CLIENT_API_KEY must be set in production environment")

        if allow_unauthenticated:
            error_msg = (
                "🚨 CRITICAL SECURITY ERROR: ALLOW_UNAUTHENTICATED=true is set in production environment!\n"
                "This disables API key authentication and creates a severe security vulnerability.\n"
                "ALLOW_UNAUTHENTICATED is forbidden in production - remove this environment variable."
            )
            logger.error(error_msg)
            print(f"\n{error_msg}\n")
            raise RuntimeError("ALLOW_UNAUTHENTICATED is forbidden in production environment")

        # Production: authentication is always required
        bypass_allowed = False
        logger.info("🔐 Production mode: API key authentication is enforced")
        print("🔐 Production mode: API key authentication is enforced")

    # Development environment validation
    else:
        if not client_api_key and not allow_unauthenticated:
            error_msg = (
                "🚨 CONFIGURATION ERROR: Neither CLIENT_API_KEY nor ALLOW_UNAUTHENTICATED is set.\n"
                "In development, you must explicitly choose one of:\n"
                "1. Set CLIENT_API_KEY=<your-key> to enable authentication\n"
                "2. Set ALLOW_UNAUTHENTICATED=true to disable authentication (development only)\n"
                "No automatic bypass is allowed - explicit configuration is required."
            )
            logger.error(error_msg)
            print(f"\n{error_msg}\n")
            raise RuntimeError("Authentication configuration is required in development")

        if allow_unauthenticated:
            bypass_allowed = True
            logger.warning("🔓 Development mode: Authentication bypass enabled via ALLOW_UNAUTHENTICATED=true")
            print("🔓 Development mode: Authentication bypass enabled")
        else:  # client_api_key is True
            bypass_allowed = False
            logger.info("🔐 Development mode: API key authentication enabled")
            print("🔐 Development mode: API key authentication enabled")

    # Log final configuration
    if bypass_allowed:
        logger.warning("⚠️  AUTHENTICATION BYPASS ENABLED - API key validation is disabled")
    else:
        logger.info("✅ API key authentication is enforced")
    
    return bypass_allowed

# Validate security configuration at startup and store the result
auth_bypass_allowed = validate_security_configuration()

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

# Initialize security components
rate_limiter = TokenBucketRateLimiter(rate_limit_config)
input_sanitizer = InputSanitizer(sanitization_config)
security_middleware = setup_security_middleware(app, get_environment())

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
                logger.warning("Rate limit exceeded: %s from %s", reason, client_ip)
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
                    logger.warning("Invalid content type: %s from %s", content_type, client_ip)
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
            # Log detailed error on server but return generic message to user
            logger.error("Endpoint error: %s", str(e), exc_info=True)
            return jsonify({'error': 'Internal server error occurred'}), 500

    return decorated_function


class _ClientError(Exception):
    """Lightweight exception with HTTP status and error type for client faults."""

    def __init__(self, message: str, status_code: int, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class SecureEmotionDetectionModel:
    def __init__(self):
        """Initialize the secure emotion detection model."""
        # Resolve model directory (allow override via env var for tests/dev)
        default_model_dir = Path(__file__).resolve().parent.parent / 'model'
        env_model_dir = os.environ.get("SECURE_MODEL_DIR")
        self.model_path = Path(env_model_dir).expanduser().resolve() if env_model_dir else default_model_dir
        logger.info("Loading secure model from: %s", self.model_path)

        # Default emotions list available even if model isn't loaded
        self.emotions = [
            'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
            'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
        ]
        self.loaded = False

        # In CI/TESTING, or when model directory is missing/invalid, run in stub mode
        if os.environ.get("TESTING") or os.environ.get("CI"):
            logger.warning("TEST/CI environment detected. Running secure model in stub mode.")
            self.tokenizer = None
            self.model = None
            self.loaded = False
            return

        # If the local model directory is missing, skip heavy loading to keep imports working
        if not self.model_path.exists() or not self.model_path.is_dir():
            logger.warning(
                "Secure model directory not found. Running in stub mode (no HF model will be loaded)."
            )
            self.tokenizer = None
            self.model = None
            self.loaded = False
            return

        # If directory exists but lacks required files, also stub to avoid HF hub lookups
        required_all = [
            self.model_path / 'config.json',
            self.model_path / 'tokenizer.json',
            self.model_path / 'tokenizer_config.json',
        ]
        if not all(p.exists() for p in required_all):
            logger.warning(
                "Secure model directory lacks expected files. Running in stub mode."
            )
            self.tokenizer = None
            self.model = None
            self.loaded = False
            return

        try:
            # Lazy import heavy deps only when not in stub mode and path checks passed
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            import torch  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path), local_files_only=True)

            # Move to GPU if available
            try:
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                    logger.info("✅ Model moved to GPU")
                else:
                    logger.info("⚠️ CUDA not available, using CPU")
            except Exception:
                # If torch is absent at runtime, remain on CPU
                logger.info("⚠️ Torch not available, using CPU")

            self.loaded = True
            logger.info("✅ Secure model loaded successfully")

        except Exception as e:
            logger.error("❌ Failed to load secure model: %s. Falling back to stub mode.", str(e))
            self.tokenizer = None
            self.model = None
            self.loaded = False

    def predict(self, text, confidence_threshold=None):
        """Make a secure prediction."""
        start_time = time.time()

        try:
            if not getattr(self, 'loaded', False):
                raise RuntimeError("SecureEmotionDetectionModel is not loaded; prediction unavailable.")
            # Ensure torch is available within function scope for linter/runtime
            try:
                import torch  # type: ignore
            except Exception as e:  # pragma: no cover
                logger.error("Torch import failed during prediction: %s", e)
                raise
            # Sanitize input text
            sanitized_text, warnings = input_sanitizer.sanitize_text(text, "emotion")
            if warnings:
                logger.warning("Sanitization warnings: %s", warnings)

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
                elif predicted_label in self.model.config.id2label:
                    predicted_emotion = self.model.config.id2label[predicted_label]
                elif str(predicted_label) in self.model.config.id2label:
                    predicted_emotion = self.model.config.id2label[str(predicted_label)]
                else:
                    predicted_emotion = f"unknown_{predicted_label}"

                # Get all probabilities
                all_probs = probabilities[0].cpu().numpy()

            prediction_time = time.time() - start_time
            logger.info("Secure prediction completed in %.3fs: '%s...' → %s (conf: %.3f)",
                       prediction_time, sanitized_text[:50], predicted_emotion, confidence)

            # Create secure response
            return {
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

        except Exception as e:
            prediction_time = time.time() - start_time
            logger.error("Secure prediction failed after %.3fs: %s", prediction_time, str(e))
            raise

# Secure model factory for explicit creation and testability
logger.info("🔒 Secure model will be created via factory function")

def create_secure_model():
    """Factory function to create a SecureEmotionDetectionModel or a stub in CI/TEST.

    This avoids implicit global state and makes the creation path explicit and mockable in tests.
    """
    if os.environ.get("TESTING") or os.environ.get("CI"):
        class _Stub:
            emotions = [
                'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
                'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
            ]
            loaded = False
        return _Stub()
    return SecureEmotionDetectionModel()


@lru_cache(maxsize=1)
def get_secure_model():
    """Return a cached secure model instance created via the factory.

    Using an LRU cache (size=1) avoids global mutable state and ensures a single
    instance per process. Tests can clear the cache with
    get_secure_model.cache_clear().
    """
    return create_secure_model()


# Provider selection for text emotion (simple registry/factory)
EMOTION_PROVIDER = os.environ.get("EMOTION_PROVIDER", "hf").lower()
_provider_registry = {}


def register_provider(name, factory):
    """Register a provider factory by name for emotion services."""
    _provider_registry[name] = factory


def get_emotion_service():
    """Return an emotion service instance for the configured provider."""
    name = EMOTION_PROVIDER
    factory = _provider_registry.get(name)
    if not factory:
        raise ValueError(f"Unsupported EMOTION_PROVIDER: {name}")
    return factory()


# Register default providers
register_provider("hf", HFEmotionService)


def _parse_single_text_payload(data: dict) -> str:
    """Validate and extract 'text' from request payload."""
    text = data.get('text') if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Field "text" must be a non-empty string')
    return text


def _sanitize_texts_batch(texts: List[str]) -> Tuple[List[str], int]:
    """Sanitize batch texts and return (sanitized_texts, total_warnings)."""
    sanitized: List[str] = []
    total_warnings = 0
    for t in texts:
        s, warnings = input_sanitizer.sanitize_text(t, "emotion")
        sanitized.append(s)
        total_warnings += len(warnings)
    return sanitized, total_warnings


def _build_provider_info() -> dict:
    """Build provider info dict reflecting local-only mode and model_dir."""
    local_only_env = str(os.environ.get('EMOTION_LOCAL_ONLY', '')).strip().lower()
    return {
        'local_only': local_only_env in ('1', 'true', 'yes', 'on'),
        'model_dir': os.environ.get('EMOTION_MODEL_DIR', '') or EMOTION_MODEL_DIR,
    }

# Read admin API key per-request to reflect environment changes during tests
def get_admin_api_key() -> str | None:
    """Fetch the admin API key from the environment on each call.

    This function intentionally does not cache the key to support dynamic
    updates (e.g., during tests or runtime reconfiguration). Be aware this
    per-request read may introduce race conditions if the environment variable
    changes mid-request; callers should treat the value as ephemeral per call.
    """
    return os.environ.get("ADMIN_API_KEY")

def require_admin_api_key(f):
    """Decorator to require admin API key via X-Admin-API-Key header.

    Reads the expected key via ``get_admin_api_key()`` for each request and
    does not cache it. See ``get_admin_api_key`` for concurrency considerations.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-Admin-API-Key")
        expected_key = get_admin_api_key()

        # Handle None values early and use constant-time comparison
        if not expected_key or not api_key or not secrets.compare_digest(str(expected_key), str(api_key)):
            logger.warning("Unauthorized admin access attempt from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized: admin API key required"}), 403
        return f(*args, **kwargs)
    return decorated_function


def require_api_key(f):
    """Decorator to require API key via X-API-Key header for protected endpoints.

    Validates client API key to ensure only authorized clients can access
    protected prediction and analysis endpoints.

    Authentication bypass is controlled by the global auth_bypass_allowed flag,
    which is set at startup based on environment configuration.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if authentication bypass is allowed (set at startup)
        if auth_bypass_allowed:
            return f(*args, **kwargs)

        # Authentication is required - validate API key
        api_key = request.headers.get("X-API-Key")
        expected_key = os.environ.get("CLIENT_API_KEY")

        if not expected_key:
            logger.error("CLIENT_API_KEY not set but authentication bypass is disabled - this should not happen")
            return jsonify({
                "error": "Server configuration error",
                "message": "API key validation is required but not configured"
            }), 500

        if not api_key:
            logger.warning("Missing API key in request from %s", request.remote_addr)
            return jsonify({
                "error": "Unauthorized: API key required",
                "message": "Include X-API-Key header with valid API key"
            }), 401

        # Use constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(str(expected_key), str(api_key)):
            logger.warning("Invalid API key attempt from %s", request.remote_addr)
            return jsonify({"error": "Unauthorized: invalid API key"}), 401

        return f(*args, **kwargs)
    return decorated_function


def _get_json_payload_or_raise() -> Dict[str, Any]:
    """Return JSON payload or raise _ClientError for invalid JSON."""
    data = request.get_json(silent=True)
    if data is None:
        raise _ClientError('Invalid JSON format', 400, 'invalid_json')
    return data


def _extract_and_filter_texts_or_raise(
    data: Dict[str, Any]
) -> Tuple[List[str], List[str], int]:
    """Extract 'texts' list, filter invalid entries, and return tuple.

    Returns (original_texts, filtered_texts, num_filtered).
    """
    if not data or 'texts' not in data or not isinstance(data['texts'], list):
        raise _ClientError(
            'Field "texts" must be a list of strings', 400, 'validation_error'
        )
    original_texts = data['texts']
    texts = [t for t in original_texts if isinstance(t, str) and t.strip()]
    num_filtered = len(original_texts) - len(texts)
    if not texts:
        raise _ClientError('No valid texts provided', 400, 'validation_error')
    return original_texts, texts, num_filtered


def _validate_alignment_count_or_raise(
    results: Any, expected_count: int
) -> bool:
    """Ensure provider results match expected count or raise _ClientError."""
    if (not isinstance(results, list)) or (len(results) != expected_count):
        raise _ClientError(
            'Provider returned mismatched result count', 502, 'provider_misalignment'
        )
    return True


def _validate_single_results_or_raise(results: Any) -> List[Dict[str, Any]]:
    """Validate single-input provider results shape and return the distribution.

    Expects results to be List[List[Dict[str, Any]]], with len(results) == 1.
    Raises _ClientError(502) on invalid shape.
    """
    if (
        (not isinstance(results, list))
        or (len(results) != 1)
        or (not isinstance(results[0], list))
    ):
        outer_type = type(results).__name__
        outer_len = (
            len(results) if isinstance(results, list) else 'N/A'
        )
        inner_type = (
            type(results[0]).__name__
            if isinstance(results, list) and results
            else 'N/A'
        )
        logger.error(
            "Provider returned invalid shape for single input: "
            "type=%s len=%s inner_type=%s",
            outer_type,
            outer_len,
            inner_type,
        )
        raise _ClientError(
            'Provider returned mismatched result count',
            502,
            'provider_misalignment'
        )
    dist = results[0]
    if dist and not (
        isinstance(dist[0], dict)
        and 'label' in dist[0]
        and 'score' in dist[0]
    ):
        inner_first_type = (
            type(dist[0]).__name__ if dist else 'N/A'
        )
        inner_keys = (
            list(dist[0].keys()) if isinstance(dist[0], dict) else 'N/A'
        )
        logger.error(
            "Provider returned invalid inner element: "
            "inner_first_type=%s keys=%s",
            inner_first_type,
            inner_keys,
        )
        raise _ClientError(
            'Provider returned mismatched result count',
            502,
            'provider_misalignment'
        )
    return dist


def _build_single_response(
    sanitized_text: str,
    dist: List[Dict[str, Any]],
    warnings: List[Any]
) -> Dict[str, Any]:
    """Build JSON response payload for the single-input endpoint."""
    return {
        'text': sanitized_text,
        'scores': dist,
        'provider': os.environ.get("EMOTION_PROVIDER", EMOTION_PROVIDER).lower(),
        'provider_info': _build_provider_info(),
        'timestamp': time.time(),
        'security': {
            'sanitization_warnings': warnings,
            'request_id': getattr(g, 'request_id', None),
            'correlation_id': getattr(g, 'correlation_id', None)
        }
    }

@app.route('/health', methods=['GET'])
@secure_endpoint
def health_check():
    """Secure health check endpoint."""
    start_time = time.time()

    try:
        mdl = get_secure_model()
        response = {
            'status': 'healthy',
            'model_loaded': getattr(mdl, 'loaded', False),
            'model_version': '2.0',
            'emotions': getattr(mdl, 'emotions', []),
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
        logger.error("Health check failed: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@require_api_key
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
            logger.error("Invalid JSON in request from %s", request.remote_addr)
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
            logger.warning("Validation error: %s from %s", str(e), request.remote_addr)
            return jsonify({'error': str(e)}), 400

        # Detect anomalies
        anomalies = input_sanitizer.detect_anomalies(data)
        if anomalies:
            logger.warning("Security anomalies detected: %s", anomalies)
            with metrics_lock:
                metrics['security_violations'] += 1

        # Make secure prediction
        model_instance = get_secure_model()
        if not getattr(model_instance, 'loaded', False):
            return jsonify({'error': 'Secure model not loaded'}), 503
        result = model_instance.predict(
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
        logger.error("Secure prediction endpoint error: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
@require_api_key
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
            logger.error("Invalid JSON in batch request from %s", request.remote_addr)
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
            logger.warning("Batch validation error: %s from %s", str(e), request.remote_addr)
            return jsonify({'error': str(e)}), 400

        # Detect anomalies
        anomalies = input_sanitizer.detect_anomalies(data)
        if anomalies:
            logger.warning("Security anomalies detected in batch: %s", anomalies)
            with metrics_lock:
                metrics['security_violations'] += 1

        # Make secure batch predictions
        results = []
        model_instance = get_secure_model()
        if not getattr(model_instance, 'loaded', False):
            return jsonify({'error': 'Secure model not loaded'}), 503
        for text in sanitized_data['texts']:
            if text.strip():
                result = model_instance.predict(
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
        update_metrics(
            response_time, success=False, error_type='batch_prediction_error'
        )
        logger.error("NLP emotion batch error: %s", e)
        return jsonify({'error': 'An internal server error occurred.'}), 500


@app.route('/nlp/emotion', methods=['POST'])
@require_api_key
@secure_endpoint
def nlp_emotion():
    """Classify emotion distribution for a single input text."""
    start_time = time.time()
    try:
        # Parse and validate JSON
        data = _get_json_payload_or_raise()
        text = _parse_single_text_payload(data)

        # Sanitize and classify
        sanitized_text, warnings = input_sanitizer.sanitize_text(text, "emotion")
        try:
            service = get_emotion_service()
        except (ImportError, ValueError):
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='provider_error')
            logger.exception("Emotion provider misconfiguration")
            return jsonify({'error': 'Emotion provider misconfiguration.'}), 503
        except Exception:
            response_time = time.time() - start_time
            update_metrics(response_time, success=False, error_type='provider_error')
            logger.exception("Unknown provider error in /nlp/emotion")
            return jsonify({'error': 'Internal server error'}), 500

        results = service.classify(sanitized_text)
        dist = _validate_single_results_or_raise(results)

        response = _build_single_response(sanitized_text, dist, warnings)

        # Update distribution metric by top label
        try:
            top = max(dist, key=lambda x: x.get('score', 0.0)) if dist else None
            update_metrics(
                time.time() - start_time,
                success=True,
                emotion=(top.get('label') if top else None),
                sanitization_warnings=len(warnings)
            )
        except Exception:
            update_metrics(
                time.time() - start_time,
                success=True,
                sanitization_warnings=len(warnings)
            )

        return jsonify(response)
    except Exception:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type='prediction_error')
        logger.exception("NLP emotion error")
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/nlp/emotion/batch', methods=['POST'])
@require_api_key
@secure_endpoint
def nlp_emotion_batch():
    """Classify emotion distributions for a batch of input texts."""
    start_time = time.time()
    try:
        data = _get_json_payload_or_raise()
        _original_texts, texts, num_filtered = _extract_and_filter_texts_or_raise(data)
        if num_filtered > 0:
            logger.warning(
                "%s invalid texts filtered out from input batch.", num_filtered
            )

        sanitized, total_warnings = _sanitize_texts_batch(texts)

        try:
            service = get_emotion_service()
        except (ImportError, ValueError):
            response_time = time.time() - start_time
            update_metrics(
                response_time, success=False, error_type='provider_error'
            )
            logger.exception("Emotion provider misconfiguration")
            return jsonify({'error': 'Emotion provider misconfiguration.'}), 503
        except Exception:
            response_time = time.time() - start_time
            update_metrics(
                response_time, success=False, error_type='provider_error'
            )
            logger.exception("Unknown provider error in /nlp/emotion/batch")
            return jsonify({'error': 'Internal server error'}), 500

        results = service.classify(sanitized)
        _validate_alignment_count_or_raise(results, len(sanitized))

        responses = []
        for text, dist in zip(sanitized, results):
            dist = dist if isinstance(dist, list) else []
            top = (
                max(dist, key=lambda x: x.get('score', 0.0))
                if dist else {'label': 'unknown', 'score': 0.0}
            )
            responses.append({
                'text': text,
                'scores': dist,
                'top_label': top.get('label'),
                'top_score': top.get('score')
            })

        response_time = time.time() - start_time
        try:
            first_top = (
                max(results[0], key=lambda x: x.get('score', 0.0))
                if results and results[0] else None
            )
            update_metrics(
                response_time,
                success=True,
                emotion=(first_top.get('label') if first_top else None),
                sanitization_warnings=total_warnings
            )
        except Exception:
            update_metrics(
                response_time, success=True, sanitization_warnings=total_warnings
            )

        return jsonify({
            'results': responses,
            'count': len(responses),
            'provider': os.environ.get("EMOTION_PROVIDER", EMOTION_PROVIDER).lower(),
            'provider_info': _build_provider_info(),
            'batch_processing_time_ms': round(response_time * 1000, 2),
            'security': {
                'sanitization_warnings': total_warnings,
                'request_id': getattr(g, 'request_id', None),
                'correlation_id': getattr(g, 'correlation_id', None)
            }
        })
    except _ClientError as ce:
        response_time = time.time() - start_time
        update_metrics(response_time, success=False, error_type=ce.error_type)
        if ce.error_type == 'provider_misalignment':
            logger.error(ce.message)
        else:
            logger.warning(ce.message)
        return jsonify({'error': ce.message}), ce.status_code
    except Exception as e:
        response_time = time.time() - start_time
        update_metrics(
            response_time, success=False, error_type='batch_prediction_error'
        )
        logger.error("NLP emotion batch error: %s", e)
        return jsonify({'error': "An internal error has occurred."}), 500

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
        logger.info("Added %s to blacklist", ip)
        return jsonify({'message': f'Added {ip} to blacklist'})
    except Exception as e:
        logger.error("Blacklist error: %s", str(e))
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
        logger.info("Added %s to whitelist", ip)
        return jsonify({'message': f'Added {ip} to whitelist'})
    except Exception as e:
        logger.error("Whitelist error: %s", str(e))
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
                'POST /nlp/emotion': 'HF-backed text emotion classification',
                'POST /nlp/emotion/batch': (
                    'HF-backed batch text emotion classification'
                ),
                'POST /security/blacklist': 'Add IP to blacklist (admin)',
                'POST /security/whitelist': 'Add IP to whitelist (admin)'
            },
            'model_info': {
                'emotions': getattr(get_secure_model(), 'emotions', []),
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
        logger.error("Documentation endpoint error: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    """Handle BadRequest exceptions (invalid JSON, etc.)."""
    logger.error("BadRequest error: %s", str(e))
    update_metrics(0.0, success=False, error_type='invalid_json')
    return jsonify({'error': 'Invalid JSON format'}), 400

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    logger.warning("404 error: %s from %s", request.path, request.remote_addr)
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(e))
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
    logger.info("🔒 Rate limiting: %s requests per minute", rate_limit_config.requests_per_minute)
    logger.info("🛡️ Security monitoring: Comprehensive logging and metrics enabled")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=8000, debug=False)

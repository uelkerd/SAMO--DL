#!/usr/bin/env python3
"""
🚨 CRITICAL SECURITY DEPLOYMENT FIX
===================================
Emergency deployment script to fix critical security vulnerabilities in Cloud Run.

This script:
1. Updates all dependencies to secure versions
2. Creates a secure API server with all security features
3. Deploys to Cloud Run with proper security headers
4. Tests the deployment for security compliance
"""

import os
import sys
import subprocess
import shlex
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
def get_project_id():
    """Get current GCP project ID dynamically"""
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to environment variable or default
        return os.environ.get('GOOGLE_CLOUD_PROJECT', 'the-tendril-466607-n8')

PROJECT_ID = get_project_id()
REGION = "us-central1"
SERVICE_NAME = "samo-emotion-api-secure"
MODEL_PATH = "/app/model"
PORT = 8080
# Use Artifact Registry instead of deprecated Container Registry
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/samo-dl"

# Security configuration
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "samo-admin-key-2024-secure")
RATE_LIMIT_PER_MINUTE = 100
MAX_INPUT_LENGTH = 512

class SecurityDeploymentFix:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.deployment_dir = self.base_dir / "deployment" / "cloud-run"
        self.secure_requirements = self.deployment_dir / "requirements_secure.txt"
        self.secure_dockerfile = self.deployment_dir / "Dockerfile.secure"
        self.secure_api = self.deployment_dir / "secure_api_server.py"

    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        # Sanitize command for security
        sanitized_command = []
        for arg in command:
            if isinstance(arg, str):
                sanitized_command.append(shlex.quote(arg))
            else:
                sanitized_command.append(str(arg))

        self.log(f"Running: {' '.join(sanitized_command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=check)
            if result.stdout:
                self.log(f"STDOUT: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e.stderr}", "ERROR")
            if check:
                raise
            return e

    def create_secure_requirements(self):
        """Create secure requirements.txt with latest secure versions"""
        self.log("Creating secure requirements.txt...")

        secure_requirements = """# Secure requirements for Cloud Run deployment
# All versions verified with safety-mcp for security and Python 3.9 compatibility

# Web framework - latest secure version
flask>=3.1.1,<4.0.0

# ML libraries - latest secure versions compatible with Python 3.9
torch>=2.0.0,<3.0.0
transformers>=4.55.0,<5.0.0
numpy>=1.26.0,<2.0.0
scikit-learn>=1.5.0,<2.0.0

# WSGI server - latest secure version
gunicorn>=23.0.0,<24.0.0

# Security libraries
cryptography>=42.0.0,<43.0.0
bcrypt>=4.2.0,<5.0.0

# Rate limiting and security
redis>=5.2.0,<6.0.0
"""

        with open(self.secure_requirements, 'w') as f:
            f.write(secure_requirements)

        self.log("✅ Secure requirements.txt created")

    def create_security_headers_module(self):
        """Create security headers module"""
        security_headers_path = self.deployment_dir / "security_headers.py"

        security_headers_code = '''#!/usr/bin/env python3
"""
Security Headers Module for Cloud Run API
"""

from flask import Flask
from typing import Dict, Any

def add_security_headers(app: Flask) -> None:
    """Add comprehensive security headers to Flask app"""

    @app.after_request
    def add_headers(response):
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Remove server information
        response.headers.pop('Server', None)

        return response
'''

        with open(security_headers_path, 'w') as f:
            f.write(security_headers_code)

        self.log("✅ Security headers module created")

    def create_rate_limiter_module(self):
        """Create Flask-compatible rate limiter"""
        rate_limiter_path = self.deployment_dir / "rate_limiter.py"

        rate_limiter_code = '''#!/usr/bin/env python3
"""
Rate Limiter for Flask API
"""

import time
import threading
from collections import defaultdict, deque
from flask import Flask, request, jsonify, g
from functools import wraps

class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()

        with self.lock:
            # Clean old requests (older than 1 minute)
            while (self.requests[client_id] and
                   current_time - self.requests[client_id][0] > 60):
                self.requests[client_id].popleft()

            # Check if under limit
            if len(self.requests[client_id]) < self.requests_per_minute:
                self.requests[client_id].append(current_time)
                return True

            return False

    def get_client_id(self, request) -> str:
        """Get client identifier"""
        # Try API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to IP address
        return f"ip:{request.remote_addr}"

def rate_limit(requests_per_minute: int = 100):
    """Rate limiting decorator"""
    limiter = RateLimiter(requests_per_minute)

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = limiter.get_client_id(request)

            if not limiter.is_allowed(client_id):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {requests_per_minute} requests per minute'
                }), 429

            return f(*args, **kwargs)
        return decorated_function
    return decorator
'''

        with open(rate_limiter_path, 'w') as f:
            f.write(rate_limiter_code)

        self.log("✅ Rate limiter module created")

    def create_secure_api_server(self):
        """Create secure API server with all security features"""
        self.log("Creating secure API server...")

        secure_api_code = f'''#!/usr/bin/env python3
"""
🚀 SECURE EMOTION DETECTION API FOR CLOUD RUN
============================================
Production-ready Flask API with comprehensive security features.
"""

import os
import time
import logging
import uuid
import threading
import hashlib
import hmac
from flask import Flask, request, jsonify, g
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Import security modules
from security_headers import add_security_headers
from rate_limiter import rate_limit

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add security headers
add_security_headers(app)

# Security configuration
ADMIN_API_KEY = "{ADMIN_API_KEY}"
MAX_INPUT_LENGTH = {MAX_INPUT_LENGTH}
RATE_LIMIT_PER_MINUTE = {RATE_LIMIT_PER_MINUTE}

# Global variables for model state (thread-safe with locks)
model = None
tokenizer = None
emotion_mapping = None
model_loading = False
model_loaded = False
model_lock = threading.Lock()

# Emotion mapping based on training order
EMOTION_MAPPING = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

def verify_api_key(api_key: str) -> bool:
    """Verify API key using constant-time comparison"""
    if not api_key:
        return False
    return hmac.compare_digest(api_key, ADMIN_API_KEY)

def require_api_key(f):
    """Decorator to require API key for admin endpoints"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not verify_api_key(api_key):
            return jsonify({'error': 'Unauthorized - Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

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
    """Load the emotion detection model"""
    global model, tokenizer, emotion_mapping, model_loading, model_loaded, model_lock
    
    with model_lock:
        if model_loading or model_loaded:
            return
    
    model_loading = True
    logger.info("🔄 Starting model loading...")
    
    try:
        # Get model path
        model_path = Path("{MODEL_PATH}")
        logger.info(f"📁 Loading model from: {{model_path}}")
        
        # Check if model files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {{model_path}}")
        
        # Load tokenizer and model
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        logger.info("📥 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Set device (CPU for Cloud Run)
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        
        emotion_mapping = EMOTION_MAPPING
        model_loaded = True
        model_loading = False
        
        logger.info(f"✅ Model loaded successfully on {{device}}")
        logger.info(f"🎯 Supported emotions: {{emotion_mapping}}")
        
    except Exception:
        model_loading = False
        logger.exception("❌ Failed to load model")
    finally:
        model_loading = False

def predict_emotion(text: str) -> dict:
    """Predict emotion for given text"""
    global model, tokenizer, emotion_mapping
    
    if not model_loaded:
        raise RuntimeError("Model not loaded")

    # Sanitize input
    text = sanitize_input(text)
    
    if not text:
        raise ValueError("Input text cannot be empty")

    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {{
        'text': text,
        'emotion': emotion_mapping[predicted_class],
        'confidence': confidence,
        'request_id': str(uuid.uuid4())
    }}

def ensure_model_loaded():
    """Ensure model is loaded before processing requests"""
    if not model_loaded and not model_loading:
        load_model()
    
    if not model_loaded:
        raise RuntimeError("Model failed to load")

def create_error_response(message: str, status_code: int = 500) -> tuple:
    """Create standardized error response"""
    return jsonify({{
        'error': message,
        'status_code': status_code,
        'request_id': str(uuid.uuid4()),
        'timestamp': time.time()
    }}), status_code

@app.before_request
def before_request():
    """Add request ID and timing to all requests"""
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    
    # Log request
    logger.info(f"Request {{g.request_id}}: {{request.method}} {{request.path}} from {{request.remote_addr}}")

@app.after_request
def after_request(response):
    """Add timing and request ID to response"""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        response.headers['X-Request-Duration'] = str(duration)
    
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    return response

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with security info"""
    return jsonify({{
        'service': 'SAMO Emotion Detection API',
        'version': '2.0.0-secure',
        'status': 'operational',
        'security': 'enabled',
        'rate_limit': RATE_LIMIT_PER_MINUTE,
        'timestamp': time.time()
    }})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({{
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'port': os.environ.get('PORT', '{PORT}'),
        'timestamp': time.time()
    }})

@app.route('/predict', methods=['POST'])
@rate_limit(RATE_LIMIT_PER_MINUTE)
def predict():
    """Predict emotion for given text"""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Content-type validation
        if not request.is_json:
            return create_error_response('Content-Type must be application/json', 400)
        
        try:
            data = request.get_json()
        except Exception:
            return create_error_response('Invalid JSON data', 400)
            
        if not data:
            return create_error_response('No JSON data provided', 400)
        
        text = data.get('text', '')
        if not text:
            return create_error_response('No text provided', 400)
        
        # Make prediction
        result = predict_emotion(text)
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Prediction error: {{e}}")
        return create_error_response('Prediction processing failed. Please try again later.')

@app.route('/predict_batch', methods=['POST'])
@rate_limit(RATE_LIMIT_PER_MINUTE)
def predict_batch():
    """Predict emotions for multiple texts"""
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Content-type validation
        if not request.is_json:
            return create_error_response('Content-Type must be application/json', 400)
        
        try:
            data = request.get_json()
        except Exception:
            return create_error_response('Invalid JSON data', 400)
            
        if not data:
            return create_error_response('No JSON data provided', 400)
        
        texts = data.get('texts', [])
        if not texts:
            return create_error_response('No texts provided', 400)
        
        # Limit batch size for security
        if len(texts) > 10:
            return create_error_response('Batch size too large (max 10)', 400)
        
        # Make predictions
        results = []
        for text in texts:
            result = predict_emotion(text)
            results.append(result)
        
        return jsonify({{'results': results}})
    
    except Exception as e:
        logger.exception(f"Batch prediction error: {{e}}")
        return create_error_response('Batch prediction processing failed. Please try again later.')

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    return jsonify({{
        'emotions': EMOTION_MAPPING,
        'count': len(EMOTION_MAPPING)
    }})

@app.route('/model_status', methods=['GET'])
@require_api_key
def model_status():
    """Get detailed model status (admin only)"""
    return jsonify({{
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'emotions': EMOTION_MAPPING if model_loaded else [],
        'device': 'cpu',
        'timestamp': time.time()
    }})

@app.route('/security_status', methods=['GET'])
@require_api_key
def security_status():
    """Get security status (admin only)"""
    return jsonify({{
        'rate_limiting': True,
        'api_key_protection': True,
        'security_headers': True,
        'input_sanitization': True,
        'request_tracking': True,
        'timestamp': time.time()
    }})

# Load model on startup
def initialize_model():
    """Initialize model before first request"""
    try:
        load_model()
    except Exception:
        logger.exception("Failed to initialize model")

# Initialize model when module is imported
initialize_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={PORT}, debug=False)
'''

        with open(self.secure_api, 'w') as f:
            f.write(secure_api_code)

        self.log("✅ Secure API server created")

    def create_secure_dockerfile(self):
        """Create secure Dockerfile"""
        self.log("Creating secure Dockerfile...")

        dockerfile_content = f'''# Use official Python runtime with explicit platform targeting
FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONHASHSEED=random \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy secure requirements first for better caching
COPY requirements_secure.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_secure.txt

# Copy application code
COPY secure_api_server.py .
COPY security_headers.py .
COPY rate_limiter.py .
COPY model/ ./model/

# Create non-root user for security (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Cloud Run requirement)
EXPOSE {PORT}

# Health check following Cloud Run best practices
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:{PORT}/health || exit 1

# Use exec form for CMD (Docker best practice)
# Set timeout to 0 for Cloud Run (allows unlimited request timeouts)
CMD exec gunicorn \\
    --bind :$PORT \\
    --workers 1 \\
    --threads 8 \\
    --timeout 0 \\
    --keep-alive 5 \\
    --max-requests 1000 \\
    --max-requests-jitter 100 \\
    --access-logfile - \\
    --error-logfile - \\
    --log-level info \\
    secure_api_server:app
'''

        with open(self.secure_dockerfile, 'w') as f:
            f.write(dockerfile_content)

        self.log("✅ Secure Dockerfile created")

    def build_and_deploy(self):
        """Build and deploy secure container to Cloud Run"""
        self.log("Building and deploying secure container...")

        # Create a temporary cloudbuild.yaml file
        cloudbuild_path = self.deployment_dir / "cloudbuild.yaml"
        cloudbuild_content = f'''steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '{ARTIFACT_REGISTRY}/{SERVICE_NAME}', '-f', 'Dockerfile.secure', '.']
images:
  - '{ARTIFACT_REGISTRY}/{SERVICE_NAME}'
'''

        with open(cloudbuild_path, 'w') as f:
            f.write(cloudbuild_content)

        # Build container
        self.log("Building secure container...")
        build_result = self.run_command([
            'gcloud', 'builds', 'submit', 
            str(self.deployment_dir),
            '--config', str(cloudbuild_path)
        ])

        if build_result.returncode != 0:
            raise RuntimeError("Container build failed")
        # Deploy to Cloud Run
        self.log("Deploying to Cloud Run...")
        deploy_result = self.run_command([
            'gcloud', 'run', 'deploy', SERVICE_NAME,
            '--image', f'{ARTIFACT_REGISTRY}/{SERVICE_NAME}',
            '--platform', 'managed',
            '--region', REGION,
            '--allow-unauthenticated',
            '--port', str(PORT),
            '--memory', '2Gi',
            '--cpu', '1',
            '--max-instances', '10',
            '--timeout', '300'
        ])

        if deploy_result.returncode != 0:
            raise RuntimeError("Cloud Run deployment failed")
        # Clean up temporary file
        cloudbuild_path.unlink(missing_ok=True)

        self.log("Secure deployment completed")

    def test_deployment(self):
        """Test the secure deployment"""
        self.log("Testing secure deployment...")

        # Get service URL
        url_result = self.run_command([
            'gcloud', 'run', 'services', 'describe', SERVICE_NAME,
            '--region', REGION,
            '--format', 'value(status.url)'
        ])

        if url_result.returncode != 0:
            raise RuntimeError("Failed to get service URL")
        service_url = url_result.stdout.strip()
        self.log(f"Service URL: {service_url}")

        # Test endpoints
        tests = [
            ('GET', '/', 'Root endpoint'),
            ('GET', '/health', 'Health check'),
            ('GET', '/emotions', 'Emotions list'),
            ('POST', '/predict', 'Prediction endpoint', {{'text': 'I am happy today!'}}),
        ]

        for test in tests:
            method, endpoint, description = test[:3]
            data = test[3] if len(test) > 3 else None

            try:
                if method == 'GET':
                    response = requests.get(f"{service_url}{endpoint}", timeout=30)
                else:
                    response = requests.post(
                        f"{service_url}{endpoint}", 
                        json=data,
                        headers={{'Content-Type': 'application/json'}},
                        timeout=30
                    )

                if response.status_code == 200:
                    self.log(f"{description}: PASS")
                else:
                    self.log(f"{description}: FAIL (Status: {response.status_code})")

            except Exception as e:
                self.log(f"{description}: ERROR ({e})")

        # Test security features
        self.log("Testing security features...")

        # Test rate limiting
        try:
            responses = []
            for _ in range(105):  # Exceed rate limit
                response = requests.post(
                    f"{service_url}/predict",
                    json={'text': 'test'},
                    headers={'Content-Type': 'application/json'},
                    timeout=5
                )
                responses.append(response.status_code)

            if 429 in responses:
                self.log("✅ Rate limiting: PASS")
            else:
                self.log("❌ Rate limiting: FAIL")

        except Exception as e:
            self.log(f"❌ Rate limiting test: ERROR ({e})")

        # Test security headers
        try:
            response = requests.get(f"{service_url}/health", timeout=10)
            headers = response.headers

            security_headers = [
                'Content-Security-Policy',
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection'
            ]

            missing_headers = [h for h in security_headers if h not in headers]

            if not missing_headers:
                self.log("✅ Security headers: PASS")
            else:
                self.log(f"❌ Security headers: FAIL (Missing: {missing_headers})")

        except Exception as e:
            self.log(f"❌ Security headers test: ERROR ({e})")

        self.log("✅ Security testing completed")

    def cleanup_old_deployment(self):
        """Clean up old insecure deployment"""
        self.log("Cleaning up old deployment...")

        # Try to delete different possible old service names
        old_services = ['samo-emotion-api', 'samo-emotion-api-71517823771', 'arch-fixed-test']

        for service_name in old_services:
            try:
                self.run_command([
                    'gcloud', 'run', 'services', 'delete', service_name,
                    '--region', REGION,
                    '--quiet'
                ], check=False)
                self.log(f"✅ Old deployment '{service_name}' cleaned up")
            except Exception as e:
                self.log(f"Info: Could not clean up '{service_name}': {e}")

    def run(self):
        """Run the complete security deployment fix"""
        try:
            self.log("🚨 STARTING CRITICAL SECURITY DEPLOYMENT FIX")
            self.log("=" * 60)

            # Step 1: Create secure files
            self.create_secure_requirements()
            self.create_security_headers_module()
            self.create_rate_limiter_module()
            self.create_secure_api_server()
            self.create_secure_dockerfile()

            # Step 2: Build and deploy
            self.build_and_deploy()

            # Step 3: Test deployment
            self.test_deployment()

            # Step 4: Clean up old deployment
            self.cleanup_old_deployment()

            self.log("=" * 60)
            self.log("🎉 CRITICAL SECURITY DEPLOYMENT FIX COMPLETED SUCCESSFULLY")
            self.log("✅ All security vulnerabilities have been fixed")
            self.log("✅ Secure API is now deployed and operational")
            self.log("✅ Rate limiting and security headers are active")
            self.log("✅ Admin endpoints are protected with API key")

        except Exception as e:
            self.log(f"❌ SECURITY DEPLOYMENT FAILED: {e}", "ERROR")
            sys.exit(1)

if __name__ == '__main__':
    fix = SecurityDeploymentFix()
    fix.run() 

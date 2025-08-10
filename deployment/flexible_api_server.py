#!/usr/bin/env python3
"""
🚀 FLEXIBLE EMOTION DETECTION API SERVER
========================================
Supports multiple HuggingFace deployment strategies:
- Serverless Inference API (free)
- Inference Endpoints (paid)  
- Self-hosted (local)
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

import requests
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class DeploymentType(Enum):
    SERVERLESS = "serverless"
    ENDPOINT = "endpoint"
    LOCAL = "local"

class FlexibleEmotionDetector:
    """Flexible emotion detector supporting multiple HuggingFace deployment strategies."""

    def __init__(self):
        """Initialize based on environment configuration."""
        self.deployment_type = DeploymentType(os.getenv('DEPLOYMENT_TYPE', 'serverless'))
        self.model_name = os.getenv('MODEL_NAME', 'your-username/samo-dl-emotion-model')
        self.hf_token = os.getenv('HF_TOKEN')

        # Emotion labels for your custom model
        self.emotion_labels = [
            'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
            'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
        ]

        self.model = None
        self.tokenizer = None
        self.session = None

        # Initialize based on deployment type
        self._initialize()

    def _initialize(self):
        """Initialize the appropriate deployment strategy."""
        logger.info(f"🔄 Initializing {self.deployment_type.value} deployment...")

        if self.deployment_type == DeploymentType.SERVERLESS:
            self._initialize_serverless()
        elif self.deployment_type == DeploymentType.ENDPOINT:
            self._initialize_endpoint()  
        elif self.deployment_type == DeploymentType.LOCAL:
            self._initialize_local()

        logger.info("✅ Initialization complete!")

    def _initialize_serverless(self):
        """Initialize serverless inference API."""
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable required for serverless API")

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

        # Create session with retry strategy
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"📡 Serverless API: {self.api_url}")

    def _initialize_endpoint(self):
        """Initialize inference endpoints."""
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable required for inference endpoints")

        self.endpoint_url = os.getenv('INFERENCE_ENDPOINT_URL')
        if not self.endpoint_url:
            raise ValueError("INFERENCE_ENDPOINT_URL environment variable required")

        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

        # Create session
        self.session = requests.Session()

        logger.info(f"🚀 Inference Endpoint: {self.endpoint_url}")

    def _initialize_local(self):
        """Initialize local model."""
        try:
            logger.info(f"📥 Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Move to GPU if available
            device = os.getenv('DEVICE', 'cpu')
            if device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("🔥 Model moved to GPU")
            else:
                logger.info("💻 Model using CPU")

            self.model.eval()

        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict emotion using the configured deployment strategy."""
        try:
            if self.deployment_type == DeploymentType.SERVERLESS:
                return self._predict_serverless(text)
            if self.deployment_type == DeploymentType.ENDPOINT:
                return self._predict_endpoint(text)
            if self.deployment_type == DeploymentType.LOCAL:
                return self._predict_local(text)

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {
                "error": str(e),
                "text": text,
                "deployment_type": self.deployment_type.value
            }

    def _predict_serverless(self, text: str) -> Dict[str, Any]:
        """Predict using serverless inference API."""
        try:
            payload = {"inputs": text}

            # Add timeout and rate limit handling
            timeout = int(os.getenv('TIMEOUT_SECONDS', '30'))
            response = self.session.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                timeout=timeout
            )

            if response.status_code == 503:
                # Model is loading (cold start)
                logger.info("🔄 Model loading, waiting...")
                time.sleep(10)  # Wait for model to load
                response = self.session.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=timeout
                )

            response.raise_for_status()
            result = response.json()

            # Convert HuggingFace format to our format
            if isinstance(result, list) and len(result) > 0:
                # Standard classification output
                predictions = result[0] if isinstance(result[0], list) else result

                # Find highest scoring emotion
                best_prediction = max(predictions, key=lambda x: x['score'])

                # Create emotion probabilities dict
                all_emotions = {pred['label']: pred['score'] for pred in predictions}

                return {
                    "emotion": best_prediction['label'],
                    "confidence": best_prediction['score'],
                    "all_emotions": all_emotions,
                    "text": text,
                    "deployment_type": "serverless",
                    "model": self.model_name
                }
            return {
                "error": "Unexpected response format",
                "raw_response": result,
                "text": text,
                "deployment_type": "serverless"
            }
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout (model may be cold starting)",
                "suggestion": "Try again in a few seconds",
                "text": text,
                "deployment_type": "serverless"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {e}",
                "text": text,
                "deployment_type": "serverless"
            }

    def _predict_endpoint(self, text: str) -> Dict[str, Any]:
        """Predict using inference endpoints."""
        try:
            payload = {"inputs": text}
            timeout = int(os.getenv('TIMEOUT_SECONDS', '10'))

            response = self.session.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()
            result = response.json()

            # Same processing as serverless
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                best_prediction = max(predictions, key=lambda x: x['score'])
                all_emotions = {pred['label']: pred['score'] for pred in predictions}

                return {
                    "emotion": best_prediction['label'],
                    "confidence": best_prediction['score'],
                    "all_emotions": all_emotions,
                    "text": text,
                    "deployment_type": "endpoint",
                    "model": self.model_name
                }
            return {
                "error": "Unexpected response format",
                "raw_response": result,
                "text": text,
                "deployment_type": "endpoint"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Endpoint request failed: {e}",
                "text": text,
                "deployment_type": "endpoint"
            }

    def _predict_local(self, text: str) -> Dict[str, Any]:
        """Predict using local model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=int(os.getenv('MAX_LENGTH', '128'))
            )

            # Move to same device as model
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                # Model has no parameters, default to CPU
                device = torch.device('cpu')
                logger.warning("Model has no parameters, using CPU device")
            
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1)

            # Convert to CPU for processing
            probabilities = probabilities.cpu()
            predicted_class = predicted_class.cpu()

            # Get emotion label
            if hasattr(self.model.config, 'id2label'):
                emotion = self.model.config.id2label[predicted_class.item()]
            else:
                emotion = self.emotion_labels[predicted_class.item()]

            # Get confidence
            confidence = probabilities[0][predicted_class].item()

            # Get all emotion probabilities
            all_emotions = {}
            for i, prob in enumerate(probabilities[0]):
                if hasattr(self.model.config, 'id2label'):
                    label = self.model.config.id2label[i]
                else:
                    label = self.emotion_labels[i] if i < len(self.emotion_labels) else f"emotion_{i}"
                all_emotions[label] = prob.item()

            return {
                "emotion": emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
                "text": text,
                "deployment_type": "local",
                "model": self.model_name,
                "device": str(device)
            }

        except Exception as e:
            return {
                "error": f"Local prediction failed: {e}",
                "text": text,
                "deployment_type": "local"
            }

    def _get_model_device_str(self) -> Optional[str]:
        """Safely get the model device as string, handling models with no parameters."""
        if not self.model:
            return None
        
        try:
            device = next(self.model.parameters()).device
            return str(device)
        except StopIteration:
            # Model has no parameters, return None or default
            logger.warning("Model has no parameters, cannot determine device")
            return "unknown"

    def get_status(self) -> Dict[str, Any]:
        """Get detector status information."""
        return {
            "deployment_type": self.deployment_type.value,
            "model_name": self.model_name,
            "emotion_labels": self.emotion_labels,
            "ready": True,
            "config": {
                "serverless_api": self.api_url if hasattr(self, 'api_url') else None,
                "endpoint_url": self.endpoint_url if hasattr(self, 'endpoint_url') else None,
                "local_device": self._get_model_device_str() if self.model else None,
            }
        }

# Initialize detector
try:
    detector = FlexibleEmotionDetector()
    logger.info("✅ Flexible emotion detector initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize detector: {e}")
    detector = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Detector not initialized'
        }), 503

    status = detector.get_status()
    status['status'] = 'healthy'
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion for given text."""
    if detector is None:
        return jsonify({
            'error': 'Detector not initialized'
        }), 503

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # Make prediction
        result = detector.predict(text)

        # Return appropriate status code
        if 'error' in result:
            return jsonify(result), 500
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    if detector is None:
        return jsonify({
            'error': 'Detector not initialized'
        }), 503

    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400

        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list'}), 400

        results = []
        for text in texts:
            if text and text.strip():
                result = detector.predict(text.strip())
                results.append(result)

        return jsonify({
            'predictions': results,
            'count': len(results),
            'deployment_type': detector.deployment_type.value
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation."""
    if detector is None:
        return jsonify({
            'error': 'Detector not initialized'
        }), 503

    status = detector.get_status()

    return jsonify({
        'message': 'Flexible Emotion Detection API',
        'version': '3.0',
        'deployment_type': status['deployment_type'],
        'model': status['model_name'],
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'POST /predict': 'Single prediction (send {"text": "your text"})',
            'POST /predict_batch': 'Batch prediction (send {"texts": ["text1", "text2"]})'
        },
        'model_info': {
            'emotions': status['emotion_labels'],
            'deployment_strategies': {
                'serverless': 'Free HuggingFace Inference API (with rate limits)',
                'endpoint': 'Paid HuggingFace Inference Endpoints (consistent latency)',  
                'local': 'Self-hosted using local transformers (maximum control)'
            }
        },
        'configuration': status['config'],
        'example_usage': {
            'single_prediction': {
                'url': 'POST /predict',
                'body': '{"text": "I am feeling happy today!"}'
            },
            'batch_prediction': {
                'url': 'POST /predict_batch',
                'body': '{"texts": ["I am happy", "I feel sad", "I am excited"]}'
            }
        }
    })

if __name__ == '__main__':
    print("🌐 Starting Flexible Emotion Detection API...")
    print("=" * 60)

    if detector:
        status = detector.get_status()
        print(f"📋 Deployment Type: {status['deployment_type'].upper()}")
        print(f"🤖 Model: {status['model_name']}")
        print(f"🎭 Emotions: {len(status['emotion_labels'])} classes")

        if status['deployment_type'] == 'serverless':
            print("💰 Cost: FREE (with rate limits)")
            print("⚡ Cold Starts: Possible")
        elif status['deployment_type'] == 'endpoint':
            print("💰 Cost: PAID per usage")
            print("⚡ Cold Starts: None")
        elif status['deployment_type'] == 'local':
            print("💰 Cost: Your infrastructure")
            print("⚡ Performance: You control")

        print("\n📋 Available endpoints:")
        print("   GET  / - API documentation")
        print("   GET  /health - Health check")
        print("   POST /predict - Single prediction")
        print("   POST /predict_batch - Batch prediction")
    else:
        print("❌ Detector initialization failed - check your configuration")

    # Configure server binding with security considerations
    host = os.getenv('FLASK_HOST', '127.0.0.1')  # Default to localhost for security
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Security warning for production binding
    if host == '0.0.0.0':
        print("\n⚠️  SECURITY WARNING: Binding to all interfaces (0.0.0.0)")
        print("   This exposes the service to external networks!")
        print("   Only use this in production with proper security measures.")
        print("   For development, use FLASK_HOST=127.0.0.1 (default)")
    
    server_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    print(f"\n🚀 Server starting on {server_url}")
    
    print("📝 Example test:")
    print(f"   curl -X POST {server_url}/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"I am feeling really happy today!\"}'")
    
    print(f"\n🔧 Configuration:")
    print(f"   Host: {host} ({'SECURE - localhost only' if host == '127.0.0.1' else 'EXPOSED - all interfaces' if host == '0.0.0.0' else 'CUSTOM'})")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    
    if host != '127.0.0.1' and host != 'localhost':
        print(f"\n💡 Security Tips:")
        print(f"   • Use FLASK_HOST=127.0.0.1 for development (secure)")
        print(f"   • Use FLASK_HOST=0.0.0.0 only in production with firewall/proxy")
        print(f"   • Never expose debug=True to external networks")

    app.run(host=host, port=port, debug=debug)

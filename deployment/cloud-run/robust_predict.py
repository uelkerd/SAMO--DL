#!/usr/bin/env python3
"""
Robust Cloud Run Application for SAMO Emotion Detection
Following Google Cloud Run best practices and troubleshooting recommendations
"""

import os
import sys
import time
import logging
import signal
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure robust logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Cloud Run expects stdout
        logging.StreamHandler(sys.stderr)   # Also log to stderr
    ]
)
logger = logging.getLogger(__name__)

# Global variables
app = Flask(__name__)
model = None
tokenizer = None
label_mapping = None
model_loaded = False
model_loading = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def load_model():
    """Load the emotion detection model with robust error handling"""
    global model, tokenizer, label_mapping, model_loaded, model_loading
    
    if model_loading:
        logger.info("Model loading already in progress...")
        return False
    
    model_loading = True
    logger.info("Starting model loading process...")
    
    try:
        # Get model path - Cloud Run specific
        model_path = os.path.join(os.getcwd(), 'model')
        logger.info(f"Model path: {model_path}")
        
        # Verify model directory exists
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found: {model_path}")
            return False
        
        # List model files for debugging
        model_files = os.listdir(model_path)
        logger.info(f"Model files found: {model_files}")
        
        # Load tokenizer with timeout
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
        
        # Load model with timeout
        logger.info("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=7,
            ignore_mismatched_sizes=True
        )
        logger.info("Model loaded successfully")
        
        # Set label mapping
        label_mapping = {
            0: 'anger',
            1: 'disgust', 
            2: 'fear',
            3: 'joy',
            4: 'neutral',
            5: 'sadness',
            6: 'surprise'
        }
        
        model_loaded = True
        model_loading = False
        logger.info("Model loading completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loading = False
        return False

def predict_emotion(text):
    """Predict emotion with robust error handling"""
    if not model_loaded:
        logger.warning("Model not loaded, cannot make prediction")
        return None
    
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get emotion label
        emotion = label_mapping.get(predicted_class, 'unknown')
        
        return {
            'emotion': emotion,
            'confidence': round(confidence, 4),
            'class_id': predicted_class
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint following Cloud Run best practices"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'model_loading': model_loading,
            'timestamp': time.time(),
            'port': os.environ.get('PORT', '8080')
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'SAMO Emotion Detection API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with robust error handling"""
    try:
        # Check if model is still loading
        if model_loading:
            return jsonify({
                'error': 'Model is still loading, please try again in a few seconds'
            }), 503
        
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request body'}), 400
        
        text = data['text']
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Make prediction
        start_time = time.time()
        result = predict_emotion(text)
        prediction_time = time.time() - start_time
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Add metadata
        result['prediction_time_ms'] = round(prediction_time * 1000, 2)
        result['text_length'] = len(text)
        
        logger.info(f"Prediction successful: {result['emotion']} (confidence: {result['confidence']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main application entry point following Cloud Run best practices"""
    try:
        # Get port from environment (Cloud Run requirement)
        port = int(os.environ.get('PORT', 8080))
        logger.info(f"Starting SAMO Emotion Detection API on port {port}")
        
        # Load model in background (non-blocking)
        import threading
        def load_model_async():
            load_model()
        
        model_thread = threading.Thread(target=load_model_async)
        model_thread.daemon = True
        model_thread.start()
        
        # Start Flask app
        logger.info("Starting Flask application...")
        app.run(
            host='0.0.0.0',  # Cloud Run requirement
            port=port,
            debug=False,     # Production mode
            threaded=True    # Enable threading for concurrent requests
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
import os
import json
import logging
import threading
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report
import time

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
label_mapping = None
model_loading = False
model_loaded = False

def load_model():
    """Load the emotion detection model"""
    global model, tokenizer, label_mapping, model_loading, model_loaded
    
    if model_loading:
        return False
    
    model_loading = True
    
    try:
        model_path = os.path.join(os.getcwd(), 'model')
        logger.info(f"Loading model from: {model_path}")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        # List model files for debugging
        model_files = os.listdir(model_path)
        logger.info(f"Model files found: {model_files}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=7,
            ignore_mismatched_sizes=True
        )
        logger.info("Model loaded successfully")
        
        # Load label mapping
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
        logger.info("Model loading completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loading = False
        return False

def load_model_async():
    """Load model in background thread"""
    def _load():
        load_model()
    
    thread = threading.Thread(target=_load)
    thread.daemon = True
    thread.start()

def predict_emotion(text):
    """Predict emotion for given text"""
    if not model_loaded:
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
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
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
        
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request body'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Make prediction
        start_time = time.time()
        result = predict_emotion(text)
        prediction_time = time.time() - start_time
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed'
            }), 500
        
        # Add metadata
        result['prediction_time_ms'] = round(prediction_time * 1000, 2)
        result['text_length'] = len(text)
        
        logger.info(f"Prediction successful: {result['emotion']} (confidence: {result['confidence']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'SAMO Emotion Detection API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        },
        'model_loaded': model_loaded,
        'model_loading': model_loading
    })

if __name__ == '__main__':
    # Start model loading in background
    logger.info("Starting model loading in background...")
    load_model_async()
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    
    # Start the server immediately (don't wait for model)
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    ) 
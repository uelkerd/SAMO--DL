import os
import json
import logging
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

def load_model():
    """Load the emotion detection model"""
    global model, tokenizer, label_mapping
    
    try:
        model_path = os.path.join(os.getcwd(), 'model')
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=7,
            ignore_mismatched_sizes=True
        )
        
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
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_emotion(text):
    """Predict emotion for given text"""
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
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
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
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
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
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        # Get port from environment (Cloud Run sets PORT)
        port = int(os.environ.get('PORT', 8080))
        
        # Run with gunicorn for production
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1) 
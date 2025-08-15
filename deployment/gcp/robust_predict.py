#!/usr/bin/env python3
"""
Simple Flask prediction service for Alpine Linux (no PyTorch)
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None
label_encoder = None

def load_model():
    """Load the scikit-learn model and label encoder"""
    global model, label_encoder
    
    try:
        model_path = os.getenv('MODEL_PATH', '/app/model')
        
        # Try to load scikit-learn model
        model_file = os.path.join(model_path, 'model.joblib')
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info(f"Loaded scikit-learn model from {model_file}")
        else:
            logger.warning(f"No model.joblib found at {model_file}")
            
        # Try to load label encoder
        encoder_file = os.path.join(model_path, 'label_encoder.joblib')
        if os.path.exists(encoder_file):
            label_encoder = joblib.load(encoder_file)
            logger.info(f"Loaded label encoder from {encoder_file}")
        else:
            logger.warning(f"No label_encoder.joblib found at {encoder_file}")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        label_encoder = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
            
        # Get input data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
            
        text = data['text']
        
        # Simple feature extraction (TF-IDF-like)
        # In a real scenario, you'd use the same preprocessing as training
        features = np.array([len(text), len(text.split())]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features)[0]
            prob_dict = {}
            if label_encoder:
                for i, prob in enumerate(probabilities):
                    label = label_encoder.classes_[i]
                    prob_dict[label] = float(prob)
        except:
            prob_dict = {}
        
        # Decode prediction if encoder available
        if label_encoder:
            try:
                predicted_label = label_encoder.inverse_transform([prediction])[0]
            except:
                predicted_label = str(prediction)
        else:
            predicted_label = str(prediction)
        
        return jsonify({
            'prediction': predicted_label,
            'probabilities': prob_dict,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'message': 'SAMO Emotion Prediction Service (Alpine Linux)',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        },
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

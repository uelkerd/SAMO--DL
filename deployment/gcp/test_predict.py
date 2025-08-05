#!/usr/bin/env python3
"""
Minimal Test Prediction Server
==============================

This is a minimal test to isolate the Vertex AI container startup issue.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check."""
    return jsonify({
        'status': 'healthy',
        'message': 'Test server is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simple test prediction."""
    try:
        data = request.get_json()
        text = data.get('text', 'test')
        
        return jsonify({
            'text': text,
            'predicted_emotion': 'happy',
            'confidence': 0.95,
            'message': 'Test prediction successful'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Test Prediction Server',
        'status': 'running'
    })

def main():
    """Main function."""
    logger.info("🚀 Starting minimal test server...")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"📁 Environment variables: {dict(os.environ)}")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    main() 
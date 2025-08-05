#!/usr/bin/env python3
"""
Simple robust Cloud Run test following official documentation
Based on Cloud Run troubleshooting guide recommendations
"""

import os
import sys
import time
from flask import Flask, jsonify

# Configure logging to stdout (Cloud Run requirement)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    logger.info("Root endpoint called")
    return jsonify({
        'message': 'Hello from SAMO Emotion Detection API!',
        'status': 'running',
        'timestamp': time.time()
    })

@app.route('/health')
def health():
    """Health check endpoint following Cloud Run best practices."""
    logger.info("Health check called")
    try:
        return jsonify({
            'status': 'healthy',
            'port': os.environ.get('PORT', '8080'),
            'timestamp': time.time()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/ready')
def ready():
    """Readiness probe endpoint."""
    logger.info("Readiness probe called")
    return jsonify({'status': 'ready'}), 200

if __name__ == '__main__':
    # Get port from environment (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting simple robust test on port {port}")
    
    # Start the server
    app.run(
        host='0.0.0.0',  # Cloud Run requirement
        port=port,
        debug=False      # Production mode
    ) 
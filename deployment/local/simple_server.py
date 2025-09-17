#!/usr/bin/env python3
"""
Simple Local API Server for Development
========================================

A lightweight Flask server for local development testing.
Serves static files and provides basic CORS support.
"""

import os
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Serve static files from website directory
@app.route('/')
def index():
    website_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'website')
    return send_from_directory(website_dir, 'comprehensive-demo.html')

@app.route('/<path:filename>')
def static_files(filename):
    website_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'website')
    return send_from_directory(website_dir, filename)

# Mock API endpoints for testing
@app.route('/api/emotion', methods=['POST'])
def mock_emotion():
    import time
    time.sleep(0.5)  # Simulate processing time
    return jsonify({
        "emotions": [
            {"emotion": "joy", "confidence": 0.85},
            {"emotion": "optimism", "confidence": 0.72},
            {"emotion": "excitement", "confidence": 0.68},
            {"emotion": "love", "confidence": 0.45},
            {"emotion": "nervousness", "confidence": 0.32}
        ],
        "primary_emotion": "joy",
        "emotional_intensity": 0.85,
        "sentiment_score": 0.78,
        "confidence_range": "0.32-0.85",
        "model_details": "SAMO DeBERTa v3 Large - 28 emotion categories analyzed",
        "processing_time_ms": 520,
        "models_used": "DeBERTa-v3-large",
        "total_time": "520ms",
        "status": "completed"
    })

@app.route('/api/summarize', methods=['POST'])
def mock_summarize():
    import time
    time.sleep(0.3)  # Simulate processing time
    return jsonify({
        "summary": "This text expresses excitement and happiness about wonderful news and future opportunities, while also acknowledging some nervousness about upcoming challenges.",
        "original_length": 245,
        "summary_length": 142,
        "processing_time_ms": 380,
        "models_used": "SAMO-T5",
        "total_time": "380ms",
        "status": "completed"
    })

@app.route('/api/transcribe', methods=['POST'])
def mock_transcribe():
    import time
    time.sleep(0.4)  # Simulate processing time
    return jsonify({
        "transcription": "I am so excited and happy today! This is such wonderful news and I feel optimistic about the future.",
        "confidence": 0.95,
        "processing_time_ms": 420,
        "models_used": "SAMO-Whisper",
        "total_time": "420ms",
        "status": "completed"
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "server": "simple_local_dev"})

if __name__ == '__main__':
    print("🚀 SIMPLE LOCAL DEVELOPMENT SERVER")
    print("==================================")
    print("🌐 Server starting at: http://localhost:8000")
    print("📁 Serving website files with CORS enabled")
    print("🔧 Mock AI endpoints available for testing")
    print("Press Ctrl+C to stop the server")
    print("")

    app.run(host='127.0.0.1', port=8000, debug=False)

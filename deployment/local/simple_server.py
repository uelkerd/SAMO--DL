#!/usr/bin/env python3
"""
Simple Local API Server for Development
========================================

A lightweight Flask server for local development testing.
Serves static files and provides basic CORS support.
"""

import os
from flask import Flask, send_from_directory, jsonify, request
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

# CORS Proxy for Real API
import requests

@app.route('/api/emotion', methods=['POST'])
def proxy_emotion():
    try:
        # Accept JSON body or query param
        data = (request.get_json(silent=True) or {})
        text = (data.get('text') or request.args.get('text', '')).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API (requests will encode params)
        api_url = "https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app/analyze/emotion"
        response = requests.post(api_url, params={"text": text}, timeout=30)

        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def proxy_summarize():
    try:
        # Accept JSON body or query param
        data = (request.get_json(silent=True) or {})
        text = (data.get('text') or request.args.get('text', '')).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API (requests will encode params)
        api_url = "https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app/analyze/summarize"
        response = requests.post(api_url, params={"text": text}, timeout=30)

        if response.ok:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

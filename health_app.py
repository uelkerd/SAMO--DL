#!/usr/bin/env python3
"""Simple health check Flask application for minimal Docker testing."""

import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health")
def health():
    """Health check endpoint.
    
    Returns:
        dict: Health status response
    """
    return jsonify({"status": "healthy"})

@app.route("/")
def root():
    """Root endpoint.
    
    Returns:
        dict: Basic application information
    """
    return jsonify({
        "message": "SAMO Health Check Service",
        "status": "running"
    })

if __name__ == "__main__":
    # For production, use HOST environment variable. For security, default to localhost
    # Override with HOST=0.0.0.0 only when explicitly needed (e.g., Docker/Cloud Run)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
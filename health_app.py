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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
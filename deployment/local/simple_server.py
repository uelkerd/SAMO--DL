# (shebang removed; run via `python deployment/local/simple_server.py`)
"""
Simple Local API Server for Development
========================================

A lightweight Flask server for local development testing.
Serves static files and provides basic CORS support.
"""

import argparse
import logging
import os

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Resolve once
WEBSITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "website"))

# Environment-configurable upstream settings
UPSTREAM_BASE = os.getenv(
    "SAMO_UNIFIED_API_BASE", "https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app"
)
API_KEY = os.getenv("SAMO_API_KEY")  # optional
COMMON_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


# Serve static files from website directory
@app.route("/")
def index():
    return send_from_directory(WEBSITE_DIR, "comprehensive-demo.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEBSITE_DIR, filename)


# CORS Proxy for Real API
@app.route("/api/emotion", methods=["POST"])
def proxy_emotion():
    try:
        # Accept JSON body or query param
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or request.args.get("text", "")).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API (requests will encode params)
        api_url = f"{UPSTREAM_BASE}/analyze/emotion"
        response = requests.post(api_url, params={"text": text}, headers=COMMON_HEADERS, timeout=30)

        if response.ok:
            return jsonify(response.json())
        return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

    except Exception:
        logging.exception("Unhandled exception in /api/emotion")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/summarize", methods=["POST"])
def proxy_summarize():
    try:
        # Accept JSON body or query param
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or request.args.get("text", "")).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API (requests will encode params)
        api_url = f"{UPSTREAM_BASE}/analyze/summarize"
        response = requests.post(api_url, params={"text": text}, headers=COMMON_HEADERS, timeout=30)

        if response.ok:
            return jsonify(response.json())
        return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

    except Exception:
        logging.exception("Unhandled exception in /api/summarize")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "server": "simple_local_dev"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Local Development Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    print("🚀 SIMPLE LOCAL DEVELOPMENT SERVER")
    print("==================================")
    print(f"🌐 Server starting at: http://{args.host}:{args.port}")
    print("📁 Serving website files with CORS enabled")
    print("🔧 Proxy AI endpoints available for testing")
    print("Press Ctrl+C to stop the server")
    print("")

    app.run(host=args.host, port=args.port, debug=False)

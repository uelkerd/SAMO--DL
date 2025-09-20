#!/usr/bin/env python3
"""
Simple Web Server for Local Development
======================================

A lightweight Flask server that serves static website files with CORS enabled
for testing against deployed Cloud Run APIs.
"""
import sys
import argparse
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Get the project root directory (two levels up from this script)
# Script is at deployment/local/simple_server.py, so we go up 2 levels to get to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEBSITE_DIR = PROJECT_ROOT / "website"


app = Flask(__name__)

# Enable CORS for all domains and all routes
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

@app.route('/')
def index():
    """Serve the main index page."""
    if (WEBSITE_DIR / "index.html").exists():
        return send_from_directory(WEBSITE_DIR, "index.html")
    if (WEBSITE_DIR / "demo.html").exists():
        return send_from_directory(WEBSITE_DIR, "demo.html")
    return jsonify({
        "error": "No index.html or demo.html found",
        "website_dir": str(WEBSITE_DIR),
        "available_files": [f.name for f in WEBSITE_DIR.glob("*.html")] if WEBSITE_DIR.exists() else []
    }), 404

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from the website directory."""
    try:
        return send_from_directory(WEBSITE_DIR, filename)
    except FileNotFoundError:
        return jsonify({
            "error": f"File '{filename}' not found",
            "website_dir": str(WEBSITE_DIR)
        }), 404

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "server": "simple_server.py",
        "website_dir": str(WEBSITE_DIR),
        "website_exists": WEBSITE_DIR.exists(),
        "available_html_files": [f.name for f in WEBSITE_DIR.glob("*.html")] if WEBSITE_DIR.exists() else []
    })

@app.errorhandler(404)
def not_found(error):
    """Custom 404 handler."""
    return jsonify({
        "error": "File not found",
        "message": "The requested file was not found in the website directory",
        "website_dir": str(WEBSITE_DIR)
    }), 404

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple web server for local development")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("🌐 SAMO Simple Web Server")
    print("=" * 30)
    print(f"📁 Website directory: {WEBSITE_DIR}")
    print(f"🔗 Server URL: http://{args.host}:{args.port}")
    print("✅ CORS enabled for Cloud Run APIs")
    print("🎯 Cloud Run API: https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app")
    print("")

    if not WEBSITE_DIR.exists():
        print(f"⚠️  Warning: Website directory not found at {WEBSITE_DIR}")
        print("   Make sure you're running this from the correct directory")
        print("")

    print("Press Ctrl+C to stop the server")
    print("")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()

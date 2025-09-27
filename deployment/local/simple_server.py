#!/usr/bin/env python3
"""Simple Web Server for Local Development
======================================

A lightweight Flask server that serves static website files with CORS enabled
for testing against deployed Cloud Run APIs.

Usage:
    python simple_server.py [--port PORT]

Environment Variables:
    PORT: Server port (default: 8000)
    ENV: Environment mode ('prod' for production CORS, default: development)
    ALLOWED_ORIGINS: Comma-separated list of allowed origins for CORS
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from flask import Flask, send_from_directory, send_file, jsonify
    from flask_cors import CORS
except ImportError:
    print("❌ Missing required dependencies!")
    print("Please install with: pip install -r requirements-simple.txt")
    sys.exit(1)

# Get the project root directory (two levels up from this script)
# Script is at deployment/local/simple_server.py, so we go up 2 levels
# to get to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEBSITE_DIR = PROJECT_ROOT / "website"

app = Flask(__name__)

# Configure CORS based on environment
is_production = os.getenv("ENV", "").lower() == "prod"
allowed_origins = os.getenv("ALLOWED_ORIGINS", "")

if is_production:
    # Production: Use environment variable or default to localhost regex
    if allowed_origins:
        origins = [origin.strip() for origin in allowed_origins.split(",")]
    else:
        origins = [r"http://localhost:\d+", r"http://127\.0\.0\.1:\d+"]

    CORS(app, origins=origins, supports_credentials=True)
    print("🔒 Production CORS: Restricted origins")
else:
    # Development: Allow all origins for easier testing
    CORS(app, origins="*", supports_credentials=True)
    print("🔓 Development CORS: All origins allowed")


@app.route('/', defaults={'filename': 'index.html'})
@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files from the website directory."""
    return send_from_directory(WEBSITE_DIR, filename)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    files_available = []
    if WEBSITE_DIR.exists():
        files_available = [f.name for f in WEBSITE_DIR.glob("*.html")]

    return jsonify({
        "status": "healthy",
        "server": "SAMO Local Development Server",
        "website_dir": str(WEBSITE_DIR),
        "cors_mode": "production" if is_production else "development",
        "files_available": files_available
    })


@app.errorhandler(404)
def not_found(_error):
    """Custom 404 handler."""
    available_files = []
    if WEBSITE_DIR.exists():
        available_files = [f.name for f in WEBSITE_DIR.glob("*.html")]

    return jsonify({
        "error": "Not found",
        "message": "The requested resource was not found on this server",
        "available_files": available_files
    }), 404


@app.errorhandler(500)
def server_error(_error):
    """Custom 500 handler."""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


def validate_environment():
    """Validate that the environment is set up correctly."""
    if not WEBSITE_DIR.exists():
        print(f"❌ Website directory not found: {WEBSITE_DIR}")
        print(f"   Make sure you're running this script from: {PROJECT_ROOT}")
        return False

    index_file = WEBSITE_DIR / "index.html"
    if not index_file.exists():
        print(f"❌ index.html not found: {index_file}")
        return False

    print(f"✅ Website directory found: {WEBSITE_DIR}")
    print(f"✅ Found {len(list(WEBSITE_DIR.glob('*.html')))} HTML files")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAMO Local Development Server")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    args = parser.parse_args()

    # Handle PORT environment variable with graceful error handling
    if args.port is None:
        port_env = os.getenv("PORT", "8000")
        try:
            args.port = int(port_env)
        except ValueError:
            print(f"❌ Invalid PORT value: {port_env!r}. Please provide an integer.")
            sys.exit(1)

    print("🚀 SAMO Local Development Server")
    print("=" * 40)

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    print(f"📁 Serving files from: {WEBSITE_DIR}")
    print(f"🌐 Server URL: http://{args.host}:{args.port}")
    print("🔗 Direct links:")
    print(f"   • Main page: http://{args.host}:{args.port}/")

    # List available HTML files
    html_files = list(WEBSITE_DIR.glob("*.html"))
    for html_file in html_files:
        if html_file.name != "index.html":
            url = f"http://{args.host}:{args.port}/{html_file.name}"
            print(f"   • {html_file.stem.title()}: {url}")

    print(f"🏥 Health check: http://{args.host}:{args.port}/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

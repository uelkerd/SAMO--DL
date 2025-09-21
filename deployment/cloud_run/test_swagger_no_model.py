#!/usr/bin/env python3
"""Test Swagger docs without model dependencies"""

import os

from flask import Flask, jsonify
from flask_restx import Api, Namespace, Resource

# Environment variables will be set in main block to avoid import-time side effects

# Create Flask app
app = Flask(__name__)


# Register root endpoint first
@app.route("/")
def home():
    return jsonify({"message": "Root endpoint"})


# Initialize Flask-RESTX API
api = Api(
    app,
    version="1.0.0",
    title="Test API",
    description="Test for Swagger docs issue",
    doc="/docs",
)

# Create namespace
main_ns = Namespace("api", description="Main operations")
api.add_namespace(main_ns)


# Test endpoint
@main_ns.route("/health")
class Health(Resource):
    @staticmethod
    def get():
        return {"status": "healthy"}


if __name__ == "__main__":
    # Set required environment variables with defaults (only when running as main)
    os.environ.setdefault("ADMIN_API_KEY", "test-key-123")
    os.environ.setdefault("MAX_INPUT_LENGTH", "512")
    os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100")
    os.environ.setdefault("MODEL_PATH", "/app/model")
    os.environ.setdefault("PORT", "8083")
    
    print("=== Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")

    print("\n=== Starting test server ===")
    print("Test these endpoints:")
    print("- http://localhost:8083/ (should work)")
    print("- http://localhost:8083/docs (should work)")
    print("- http://localhost:8083/api/health (should work)")

    app.run(
        host="0.0.0.0", port=int(os.environ.get("PORT", 8083)), debug=False
    )  # Debug mode disabled for security

#!/usr/bin/env python3
"""Minimal test to isolate Swagger docs issue"""

import os
from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Create Flask app
app = Flask(__name__)


# Register root endpoint first
@app.route("/")
def root():
    return jsonify({"message": "Root endpoint"})


# Initialize Flask-RESTX API
api = Api(
    app,
    version="1.0.0",
    title="Test API",
    description="Minimal test for Swagger docs",
    doc="/docs",
)

# Create namespace
main_ns = Namespace("api", description="Main operations")
api.add_namespace(main_ns)


# Test endpoint
@main_ns.route("/health")
class Health(Resource):
    def get(self):
        return {"status": "healthy"}


if __name__ == "__main__":
    print("=== Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")

    print("\n=== Starting test server ===")
    print("Test these endpoints:")
    print("- http://localhost:5003/ (should work)")
    print("- http://localhost:5003/docs (should work)")
    print("- http://localhost:5003/api/health (should work)")

    # Use secure host binding for test server
    try:
        from src.security.host_binding import (
            get_secure_host_binding,
            validate_host_binding,
        )

        host, port = get_secure_host_binding(int(os.environ.get("PORT", 5003)))
        validate_host_binding(host, port)
        app.run(host=host, port=port, debug=False)
    except ImportError:
        # Fallback for test environment
        app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5003)), debug=False)

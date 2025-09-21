#!/usr/bin/env python3
"""Minimal test to isolate Swagger docs issue"""

import os

from flask import Flask, jsonify
from flask_restx import Api, Namespace, Resource

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
    @staticmethod
    def get():
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

    # Default to loopback interface for security, allow opt-in to bind all interfaces
    bind_all = os.environ.get("BIND_ALL", "").lower() in ("1", "true")
    host = "0.0.0.0" if bind_all else "127.0.0.1"

    app.run(
        host=host, port=int(os.environ.get("PORT", 5003)), debug=False
    )  # Debug mode disabled for security

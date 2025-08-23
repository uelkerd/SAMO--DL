#!/usr/bin/env python3
"""Test script to debug Swagger docs 500 error."""

from flask import Flask, jsonify
from flask_restx import Api, Namespace, Resource

# Create Flask app
app = Flask(__name__)

# Initialize Flask-RESTX API
api = Api(
    app,
    version="1.0.0",
    title="Test API",
    description="Minimal test to isolate routing issues",
    doc="/docs",
)

# Create namespace
main_ns = Namespace("/api", description="Main operations")
api.add_namespace(main_ns)


# Test endpoint in namespace
@main_ns.route("/health")
class Health(Resource):
    def get(self):
        return {"status": "healthy"}


# Override the root route with a different endpoint name
@app.route("/")
def api_root():  # Different function name to avoid conflict
    return jsonify({"message": "Root endpoint"})


if __name__ == "__main__":
    print("=== Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")

    print("\n=== Starting test server ===")
    print("Test these endpoints:")
    print("- http://localhost:5001/ (should work)")
    print("- http://localhost:5001/docs (should work)")
    print("- http://localhost:5001/api/health (should work)")

    app.run(host="0.0.0.0", port=5001, debug=True)

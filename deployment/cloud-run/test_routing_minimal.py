#!/usr/bin/env python3
"""
Minimal test script to isolate Flask-RESTX routing issues
"""

import os
from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Create Flask app
app = Flask(__name__)

# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0.0',
    title='Test API',
    description='Minimal test to isolate routing issues',
    doc='/docs'
)

# Create namespace with a different path to avoid conflicts
main_ns = Namespace('/api', description='Main operations')  # Changed from '/' to '/api'
api.add_namespace(main_ns)

# Test endpoint in namespace
@main_ns.route('/health')
class Health(Resource):
    def get(self):
        return {'status': 'healthy'}

# Test direct Flask route BEFORE API setup
@app.route('/test_before')
def test_before():
    return jsonify({'message': 'This route was added before API setup'})

# Test direct Flask route AFTER API setup
@app.route('/test_after')
def test_after():
    return jsonify({'message': 'This route was added after API setup'})

# Test root endpoint - this should work now
@app.route('/')
def root():
    return jsonify({'message': 'Root endpoint'})

if __name__ == '__main__':
    print("=== Flask App Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"App: {rule.rule} -> {rule.endpoint}")

    print("\n=== Flask-RESTX API Routes ===")
    for rule in api.url_map.iter_rules():
        print(f"API: {rule.rule} -> {rule.endpoint}")

    print("\n=== Starting test server ===")
    # Use secure host binding for test server
    try:
        from src.security.host_binding import get_secure_host_binding, validate_host_binding
        host, port = get_secure_host_binding(int(os.environ.get('PORT', 5000)))
        validate_host_binding(host, port)
        app.run(host=host, port=port, debug=False)
    except ImportError:
        # Fallback for test environment
        app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=False)
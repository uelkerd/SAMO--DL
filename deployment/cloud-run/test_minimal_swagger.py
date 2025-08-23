#!/usr/bin/env python3
"""
Minimal test to isolate Swagger docs issue
"""

from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Create Flask app
app = Flask(__name__)

# Register root endpoint first
@app.route('/')
def root():
    return jsonify({'message': 'Root endpoint'})

# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0.0',
    title='Test API',
    description='Minimal test for Swagger docs',
    doc='/docs'
)

# Create namespace
main_ns = Namespace('api', description='Main operations')
api.add_namespace(main_ns)

# Test endpoint
@main_ns.route('/health')
class Health(Resource):
    def get(self):
        return {'status': 'healthy'}

if __name__ == '__main__':
    print("=== Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")
    
    print("\n=== Starting test server ===")
    print("Test these endpoints:")
    print("- http://localhost:5003/ (should work)")
    print("- http://localhost:5003/docs (should work)")
    print("- http://localhost:5003/api/health (should work)")
    
    app.run(host='0.0.0.0', port=5003, debug=True)   # Allow all interfaces - review for production
#!/usr/bin/env python3
"""Minimal test to isolate Swagger docs issue."""

import os
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
    for _rule in app.url_map.iter_rules():
        pass
    
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)), debug=False)  # Debug mode disabled for security

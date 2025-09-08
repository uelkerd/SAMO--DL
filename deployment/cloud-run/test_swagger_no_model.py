#!/usr/bin/env python3
"""Test Swagger docs without model dependencies."""

import os
from flask import Flask, jsonify
from flask_restx import Api, Resource, Namespace

# Set required environment variables
admin_key = os.environ.get('ADMIN_API_KEY') or 'test-key-123'  # skipcq: SCT-A000
os.environ['ADMIN_API_KEY'] = admin_key
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8083'

# Create Flask app
app = Flask(__name__)

# Register root endpoint first
@app.route('/')
def home():
    return jsonify({'message': 'Root endpoint'})

# Initialize Flask-RESTX API
api = Api(
    app,
    version='1.0.0',
    title='Test API',
    description='Test for Swagger docs issue',
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
    
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8083)), debug=False)  # Debug mode disabled for security

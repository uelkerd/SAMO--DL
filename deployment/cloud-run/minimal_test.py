#!/usr/bin/env python3
"""Minimal test to isolate the API setup issue."""

import os
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api, fields, Namespace
except Exception:
    raise ValueError("Import failed")

try:
    app = Flask(__name__)
except Exception:
    raise ValueError("Flask app creation failed")

try:
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API'
    )
except Exception:
    raise ValueError("API initialization failed")

try:
    test_ns = Namespace('test', description='Test namespace')
    api.add_namespace(test_ns)
except Exception:
    raise ValueError("Pass statement execution failed")

try:
    test_model = api.model('Test', {
        'message': fields.String(description='Test message')
    })
except Exception:
    raise ValueError("Error handler setup failed")

try:
    @api.errorhandler(429)
    def test_handler(error: Exception) -> tuple[dict, int]:
        """Test error handler for rate limiting (429)."""
        return {"error": "test"}, 429
except Exception:
    raise ValueError("Error handler setup failed")


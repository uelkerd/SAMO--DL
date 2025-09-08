#!/usr/bin/env python3
"""Minimal test to isolate the API setup issue."""

import os
import sys
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api, fields, Namespace
except Exception:
    sys.exit(1)

try:
    app = Flask(__name__)
except Exception:
    sys.exit(1)

try:
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API'
    )
except Exception:
    sys.exit(1)

try:
    test_ns = Namespace('test', description='Test namespace')
    api.add_namespace(test_ns)
except Exception:
    sys.exit(1)

try:
    test_model = api.model('Test', {
        'message': fields.String(description='Test message')
    })
except Exception:
    sys.exit(1)

try:
    @api.errorhandler(429)
    def test_handler(error: Exception) -> tuple[dict, int]:
        """Test error handler for rate limiting (429)."""
        return {"error": "test"}, 429
except Exception:
    sys.exit(1)


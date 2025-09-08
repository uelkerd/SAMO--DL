#!/usr/bin/env python3
"""
Minimal test to isolate the API setup issue
"""

import os
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))

print("🔍 Starting minimal API setup test...")

try:
    print("1. Importing modules...")
    from flask import Flask
    from flask_restx import Api, Resource, fields, Namespace
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Imports failed: {e}")
    raise RuntimeError(f"Imports failed: {e}") from e

try:
    print("2. Creating Flask app...")
    app = Flask(__name__)
    print("✅ Flask app created")
except Exception as e:
    print(f"❌ Flask app creation failed: {e}")
    raise RuntimeError(f"Flask app creation failed: {e}") from e

try:
    print("3. Creating API object...")
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API'
    )
    print(f"✅ API object created: {type(api)}")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    raise RuntimeError(f"API creation failed: {e}") from e

try:
    print("4. Creating namespace...")
    test_ns = Namespace('test', description='Test namespace')
    api.add_namespace(test_ns)
    print("✅ Namespace added")
except Exception as e:
    print(f"❌ Namespace creation failed: {e}")
    raise RuntimeError(f"Namespace creation failed: {e}") from e

try:
    print("5. Creating model...")
    test_model = api.model('Test', {
        'message': fields.String(description='Test message')
    })
    print("✅ Model created")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    raise RuntimeError(f"Model creation failed: {e}") from e

try:
    print("6. Testing errorhandler...")
    from werkzeug.exceptions import TooManyRequests
    @api.errorhandler(TooManyRequests)
    def test_handler(error):
        """Return a canned 429 for debug validation."""
        return {"error": "test"}, 429
    print("✅ Error handler created")
except Exception as e:
    print(f"❌ Error handler creation failed: {e}")
    print(f"API type at this point: {type(api)}")
    print(f"API errorhandler type: {type(api.errorhandler)}")
    raise RuntimeError(f"Error handler creation failed: {e}") from e

print("🎉 All tests passed!") 
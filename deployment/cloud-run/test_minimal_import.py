#!/usr/bin/env python3
"""
Minimal test to isolate the API issue
"""

import os
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))

print("🔍 Starting minimal import test...")

try:
    print("1. Importing Flask and Flask-RESTX...")
    from flask import Flask
    from flask_restx import Api
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    raise RuntimeError(f"Basic imports failed: {e}") from e

try:
    print("2. Creating Flask app...")
    app = Flask(__name__)
    print("✅ Flask app created")
except Exception as e:
    print(f"❌ Flask app creation failed: {e}")
    raise RuntimeError(f"Flask app creation failed: {e}") from e

try:
    print("3. Creating API object...")
    api = Api(app, version='1.0.0', title='Test')
    print(f"✅ API object created: {type(api)}")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    raise RuntimeError(f"API creation failed: {e}") from e

try:
    print("4. Testing API methods...")
    print(f"API type: {type(api)}")
    print(f"Has errorhandler: {'errorhandler' in dir(api)}")
    print(f"errorhandler type: {type(api.errorhandler)}")
    print("✅ API methods check successful")
except Exception as e:
    print(f"❌ API methods check failed: {e}")
    raise RuntimeError(f"API methods check failed: {e}") from e

try:
    print("5. Testing errorhandler call...")
    from werkzeug.exceptions import TooManyRequests
    result = api.errorhandler(TooManyRequests)
    assert callable(result), "Expected a decorator (callable) from api.errorhandler"
    print("✅ errorhandler(TooManyRequests) call returned a callable")
except Exception as e:
    print(f"❌ errorhandler(TooManyRequests) call failed: {e}")
    print(f"Error type: {type(e)}")
    raise RuntimeError(f"errorhandler(TooManyRequests) call failed: {e}") from e

print("🎉 All tests passed!") 
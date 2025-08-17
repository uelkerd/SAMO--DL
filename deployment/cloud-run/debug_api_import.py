#!/usr/bin/env python3
"""
Debug script to isolate the 'int' object is not callable error
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Starting API import debug...")

try:
    print("1. Importing Flask...")
    from flask import Flask
    print("✅ Flask imported successfully")
except Exception as e:
    print(f"❌ Flask import failed: {e}")
    sys.exit(1)

try:
    print("2. Importing Flask-RESTX...")
    from flask_restx import Api, Resource, fields, Namespace
    print("✅ Flask-RESTX imported successfully")
except Exception as e:
    print(f"❌ Flask-RESTX import failed: {e}")
    sys.exit(1)

try:
    print("3. Creating Flask app...")
    app = Flask(__name__)
    print("✅ Flask app created successfully")
except Exception as e:
    print(f"❌ Flask app creation failed: {e}")
    sys.exit(1)

try:
    print("4. Creating API object...")
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API for debugging'
    )
    print(f"✅ API object created successfully: {type(api)}")
    print(f"API object: {api}")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    sys.exit(1)

try:
    print("5. Testing API decorator...")
    @api.errorhandler(429)
    def test_handler(error):
        return {"error": "test"}, 429
    print("✅ API decorator test successful")
except Exception as e:
    print(f"❌ API decorator test failed: {e}")
    print(f"API type at this point: {type(api)}")
    print(f"API value at this point: {api}")
    sys.exit(1)

try:
    print("6. Testing namespace creation...")
    test_ns = Namespace('test', description='Test namespace')
    api.add_namespace(test_ns)
    print("✅ Namespace test successful")
except Exception as e:
    print(f"❌ Namespace test failed: {e}")
    sys.exit(1)

print("🎉 All tests passed! The issue is not with basic Flask-RESTX functionality.")

# Now let's test the actual imports from secure_api_server.py
try:
    print("\n7. Testing security_headers import...")
    from security_headers import add_security_headers
    print("✅ security_headers imported successfully")
except Exception as e:
    print(f"❌ security_headers import failed: {e}")

try:
    print("8. Testing rate_limiter import...")
    from rate_limiter import rate_limit
    print("✅ rate_limiter imported successfully")
except Exception as e:
    print(f"❌ rate_limiter import failed: {e}")

try:
    print("9. Testing model_utils import...")
    from model_utils import    from model_utils import ensure_model_loaded,
         predict_emotions,
         get_model_status,
         validate_text_input
    print("✅ model_utils imported successfully")
except Exception as e:
    print(f"❌ model_utils import failed: {e}")

print("\n🔍 Debug complete. Check above for any import issues.")  # noqa: T201

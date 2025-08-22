#!/usr/bin/env python3
"""
Minimal test to isolate the API issue
"""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print("🔍 Starting minimal import test...")

try:
    print("1. Importing Flask and Flask-RESTX...")
    from flask import Flask
    from flask_restx import Api
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

try:
    print("2. Creating Flask app...")
    app = Flask(__name__)
    print("✅ Flask app created")
except Exception as e:
    print(f"❌ Flask app creation failed: {e}")
    exit(1)

try:
    print("3. Creating API object...")
    api = Api(app, version='1.0.0', title='Test')
    print(f"✅ API object created: {type(api)}")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    exit(1)

try:
    print("4. Testing API methods...")
    print(f"API type: {type(api)}")
    print(f"Has errorhandler: {'errorhandler' in dir(api)}")
    print(f"errorhandler type: {type(api.errorhandler)}")
    print("✅ API methods check successful")
except Exception as e:
    print(f"❌ API methods check failed: {e}")
    exit(1)

try:
    print("5. Testing errorhandler call...")
    result = api.errorhandler(429)
    print(f"✅ errorhandler(429) call successful: {type(result)}")
except Exception as e:
    print(f"❌ errorhandler(429) call failed: {e}")
    print(f"Error type: {type(e)}")
    exit(1)

print("🎉 All tests passed!")

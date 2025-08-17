#!/usr/bin/env python3
"""
Minimal test to isolate the API issue
"""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print"🔍 Starting minimal import test..."

try:
    print"1. Importing Flask and Flask-RESTX..."
    from flask import Flask
    from flask_restx import Api
    print"✅ Basic imports successful"
except Exception as e:
    printf"❌ Basic imports failed: {e}"
    exit1

try:
    print"2. Creating Flask app..."
    app = Flask__name__
    print"✅ Flask app created"
except Exception as e:
    printf"❌ Flask app creation failed: {e}"
    exit1

try:
    print"3. Creating API object..."
    api = Apiapp, version='1.0.0', title='Test'
    print(f"✅ API object created: {typeapi}")
except Exception as e:
    printf"❌ API creation failed: {e}"
    exit1

try:
    print"4. Testing API methods..."
    print(f"API type: {typeapi}")
    print(f"Has errorhandler: {'errorhandler' in dirapi}")
    print(f"errorhandler type: {typeapi.errorhandler}")
    print"✅ API methods check successful"
except Exception as e:
    printf"❌ API methods check failed: {e}"
    exit1

try:
    print"5. Testing errorhandler call..."
    result = api.errorhandler429
    print(f"✅ errorhandler429 call successful: {typeresult}")
except Exception as e:
    print(f"❌ errorhandler429 call failed: {e}")
    print(f"Error type: {typee}")
    exit1

print"🎉 All tests passed!" 
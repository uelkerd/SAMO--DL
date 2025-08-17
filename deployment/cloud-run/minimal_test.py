#!/usr/bin/env python3
"""
Minimal test to isolate the API setup issue
"""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print"🔍 Starting minimal API setup test..."

try:
    print"1. Importing modules..."
    from flask import Flask
    from flask_restx import Api, Resource, fields, Namespace
    print"✅ Imports successful"
except Exception as e:
    printf"❌ Imports failed: {e}"
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
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API'
    )
    print(f"✅ API object created: {typeapi}")
except Exception as e:
    printf"❌ API creation failed: {e}"
    exit1

try:
    print"4. Creating namespace..."
    test_ns = Namespace'test', description='Test namespace'
    api.add_namespacetest_ns
    print"✅ Namespace added"
except Exception as e:
    printf"❌ Namespace creation failed: {e}"
    exit1

try:
    print"5. Creating model..."
    test_model = api.model('Test', {
        'message': fields.Stringdescription='Test message'
    })
    print"✅ Model created"
except Exception as e:
    printf"❌ Model creation failed: {e}"
    exit1

try:
    print"6. Testing errorhandler..."
    @api.errorhandler429
    def test_handlererror:
        return {"error": "test"}, 429
    print"✅ Error handler created"
except Exception as e:
    printf"❌ Error handler creation failed: {e}"
    print(f"API type at this point: {typeapi}")
    print(f"API errorhandler type: {typeapi.errorhandler}")
    exit1

print"🎉 All tests passed!" 
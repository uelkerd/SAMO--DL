#!/usr/bin/env python3
"""
Test direct error handler registration
"""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print"🔍 Testing direct error handler registration..."

try:
    from flask import Flask
    from flask_restx import Api
    print"✅ Imports successful"
except Exception as e:
    printf"❌ Import failed: {e}"
    exit1

try:
    app = Flask__name__
    api = Apiapp, version='1.0.0', title='Test'
    print"✅ API object created"
except Exception as e:
    printf"❌ API creation failed: {e}"
    exit1

# Let's try to register error handlers directly
try:
    print"1. Testing direct error handler registration..."
    
    def rate_limit_handlererror:
        return {"error": "Rate limit exceeded"}, 429
    
    def internal_error_handlererror:
        return {"error": "Internal server error"}, 500
    
    # Try to register directly
    api.error_handlers[429] = rate_limit_handler
    api.error_handlers[500] = internal_error_handler
    
    print"✅ Direct registration successful"
    printf"Error handlers: {api.error_handlers}"
    
except Exception as e:
    printf"❌ Direct registration failed: {e}"

# Let's also try using the Flask app's error handler
try:
    print"\n2. Testing Flask app error handler..."
    
    @app.errorhandler429
    def flask_rate_limit_handlererror:
        return {"error": "Rate limit exceeded"}, 429
    
    @app.errorhandler500
    def flask_internal_error_handlererror:
        return {"error": "Internal server error"}, 500
    
    print"✅ Flask app error handlers registered"
    
except Exception as e:
    printf"❌ Flask app error handler failed: {e}"

print"\n�� Test complete." 
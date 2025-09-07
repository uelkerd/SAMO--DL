#!/usr/bin/env python3
"""
Test direct error handler registration
"""

import os
os.environ['ADMIN_API_KEY'] = os.getenv('ADMIN_API_KEY', 'test123')

print("🔍 Testing direct error handler registration...")

try:
    from flask import Flask
    from flask_restx import Api
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

try:
    app = Flask(__name__)
    api = Api(app, version='1.0.0', title='Test')
    print("✅ API object created")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    exit(1)

# Let's try to register error handlers directly
try:
    print("1. Testing direct error handler registration...")
    
    def rate_limit_handler(error):
        return {"error": "Rate limit exceeded"}, 429
    
    def internal_error_handler(error):
        return {"error": "Internal server error"}, 500
    
    # Try to register directly
    api.error_handlers[429] = rate_limit_handler
    api.error_handlers[500] = internal_error_handler
    
    print("✅ Direct registration successful")
    print(f"Error handlers: {api.error_handlers}")
    
except Exception as e:
    print(f"❌ Direct registration failed: {e}")

# Let's also try using the Flask app's error handler
try:
    print("\n2. Testing Flask app error handler...")
    
    @app.errorhandler(429)
    def flask_rate_limit_handler(error):
        return {"error": "Rate limit exceeded"}, 429
    
    @app.errorhandler(500)
    def flask_internal_error_handler(error):
        return {"error": "Internal server error"}, 500
    
    print("✅ Flask app error handlers registered")
    
except Exception as e:
    print(f"❌ Flask app error handler failed: {e}")

print("\n�� Test complete.") 
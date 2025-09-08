"""Test direct error handler registration."""

import os
import sys
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api
except Exception:
    raise ValueError("Import failed")

try:
    app = Flask(__name__)
    api = Api(app, version='1.0.0', title='Test')
except Exception:
    raise ValueError("Flask app creation failed")

# Let's try to register error handlers directly
try:
    
    def rate_limit_handler(error):
        return {"error": "Rate limit exceeded"}, 429
    
    def internal_error_handler(error):
        return {"error": "Internal server error"}, 500
    
    # Try to register directly
    
    
except Exception:
    pass

# Let's also try using the Flask app's error handler
try:
    
    @app.errorhandler(429)
    def flask_rate_limit_handler(error):
        return {"error": "Rate limit exceeded"}, 429
    
    @app.errorhandler(500)
    def flask_internal_error_handler(error):
        return {"error": "Internal server error"}, 500
    
    
except Exception:
    pass


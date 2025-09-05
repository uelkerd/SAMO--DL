#!/usr/bin/env python3
"""
Test direct error handler registration
"""

import os
import logging

os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🔍 Testing direct error handler registration...")

try:
    from flask import Flask
    from flask_restx import Api
    logger.info("✅ Imports successful")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")
    exit(1)

try:
    app = Flask(__name__)
    api = Api(app, version='1.0.0', title='Test')
    logger.info("✅ API object created")
except Exception as e:
    logger.error(f"❌ API creation failed: {e}")
    exit(1)

# Let's try to register error handlers with decorators
try:
    logger.info("1. Testing error handler registration with decorators...")

    @api.errorhandler(429)
    def rate_limit_handler(error) -> tuple:
        return {"error": "Rate limit exceeded"}, 429

    @api.errorhandler(500)
    def internal_error_handler(error) -> tuple:
        return {"error": "Internal server error"}, 500

    logger.info("✅ Decorator registration successful")
    logger.info(f"Error handlers: {api.error_handlers}")

except Exception as e:
    logger.error(f"❌ Decorator registration failed: {e}")

# Let's also try using the Flask app's error handler
try:
    logger.info("2. Testing Flask app error handler...")
    
    @app.errorhandler(429)
    def flask_rate_limit_handler(error):
        return {"error": "Rate limit exceeded"}, 429
    
    @app.errorhandler(500)
    def flask_internal_error_handler(error):
        return {"error": "Internal server error"}, 500
    
    logger.info("✅ Flask app error handlers registered")
    
except Exception as e:
    logger.error(f"❌ Flask app error handler failed: {e}")

print("\n�� Test complete.") 
#!/usr/bin/env python3
"""Debug test server to validate Flask-RESTX hypotheses"""

import os
import logging
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, Namespace
import sys

# Set up environment variables
os.environ['ADMIN_API_KEY'] = 'test123'

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Test 1: Register root endpoint BEFORE Flask-RESTX initialization
logger.info("🔍 Test 1: Registering root endpoint BEFORE Flask-RESTX initialization...")
@app.route('/')
def home():
    """Get API status and information"""
    logger.info("Root endpoint accessed from %s", request.remote_addr)
    return jsonify({
        'service': 'Test API',
        'status': 'operational',
        'timestamp': 1234567890
    })

# Test 2: Initialize Flask-RESTX API
logger.info("🔍 Test 2: Initializing Flask-RESTX API...")
try:
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Debug test for Flask-RESTX issues',
        doc='/docs'  # Enable docs to test for 500 errors
    )
    logger.info("✅ Flask-RESTX API initialized successfully")
except Exception as e:
    logger.error("❌ Flask-RESTX API initialization failed: %s", str(e))
    sys.exit(1)

# Test 3: Create namespaces - test with and without leading slashes
logger.info("🔍 Test 3: Creating namespaces...")
main_ns = Namespace('api', description='Main operations')  # No leading slash
admin_ns = Namespace('admin', description='Admin operations')  # No leading slash - fixed

logger.info("🔍 Adding namespaces to API...")
api.add_namespace(main_ns)
api.add_namespace(admin_ns)
logger.info("✅ Namespaces added successfully")

# Test 4: Register routes in namespaces
@main_ns.route('/health')
class Health(Resource):
    @staticmethod
    def get():
        """Return the health status of the service."""
        return {'status': 'healthy'}

@admin_ns.route('/status')
class AdminStatus(Resource):
    @staticmethod
    def get():
        """Return the admin status of the service."""
        return {'admin_status': 'ok'}

# Test 5: Register error handlers
logger.info("🔍 Test 5: Registering error handlers...")
def test_error_handler(error):
    """Handle test errors and return error response."""
    logger.error("Test error handler: %s", str(error))
    return {'error': 'Test error'}, 500

def exception_error_handler(error):
    """Handle general exceptions and return error response."""
    logger.error("Exception error handler: %s", str(error))
    return {'error': 'Exception occurred'}, 500

try:
    api.error_handlers[500] = test_error_handler
    api.error_handlers[Exception] = exception_error_handler  # Add exception-level error handler
    logger.info("✅ Error handlers registered successfully")
except Exception as e:
    logger.error("❌ Error handler registration failed: %s", str(e))

# Test 6: Log final route state
logger.info("🔍 Test 6: Final route registration check:")
for rule in app.url_map.iter_rules():
    logger.info("  Route: %s -> %s (methods: %s)", rule.rule, rule.endpoint, list(rule.methods))

if __name__ == '__main__':
    logger.info("🚀 Starting debug test server...")
    logger.info("Test endpoints:")
    logger.info("  - GET / (root endpoint)")
    logger.info("  - GET /docs (Swagger docs - check for 500 errors)")
    logger.info("  - GET /api/health (namespace route)")
    logger.info("  - GET /admin/status (admin namespace route)")

    app.run(host='0.0.0.0', port=5002, debug=False)
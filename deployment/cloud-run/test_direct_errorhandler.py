#!/usr/bin/env python3
"""
Test direct error handler registration
"""

import os
import logging


level_name = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(level=getattr(logging, level_name, logging.DEBUG))
logger = logging.getLogger(__name__)

def _main() -> None:
    logger.info("🔍 Testing direct error handler registration...")

    try:
        from flask import Flask
        from flask_restx import Api
        logger.info("✅ Imports successful")
    except Exception as e:
        logger.exception("❌ Import failed")
        raise RuntimeError(f"Import failed: {e}") from e

    try:
        app = Flask(__name__)
        api = Api(app, version='1.0.0', title='Test')
        logger.info("✅ API object created")
    except Exception as e:
        logger.exception("❌ API creation failed")
        raise RuntimeError(f"API creation failed: {e}") from e

    # Let's try to register error handlers with decorators
    try:
        logger.info("1. Testing error handler registration with decorators...")

        from werkzeug.exceptions import TooManyRequests

        @api.errorhandler(TooManyRequests)
        def rate_limit_handler(error) -> tuple:
            """Return JSON for 429 errors."""
            return {"error": "Rate limit exceeded"}, 429

        @api.errorhandler(Exception)
        def internal_error_handler(error) -> tuple:
            """Return JSON with appropriate status for unhandled errors."""
            status = getattr(error, "code", 500)
            return {"error": "Internal server error"}, status

        logger.info("✅ Decorator registration successful")
        logger.info(
            "Error handlers registered for: %s",
            [getattr(k, "__name__", str(k)) for k in api.error_handlers.keys()],
        )

    except Exception as e:
        logger.exception("❌ Decorator registration failed: %s", e)

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
        logger.exception("❌ Flask app error handler failed: %s", e)

    logger.info("Test complete.")

if __name__ == "__main__":
    _main() 
#!/usr/bin/env python3
"""Test direct error handler registration"""

import os
import sys


def main():
    """Main test function with proper environment handling."""
    # Save original ADMIN_API_KEY value
    original_admin_key = os.environ.get("ADMIN_API_KEY")

    try:
        # Set test environment
        os.environ["ADMIN_API_KEY"] = "test123"

        print("🔍 Testing direct error handler registration...")

        try:
            from flask import Flask
            from flask_restx import Api

            print("✅ Imports successful")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            raise SystemExit(1)

        try:
            app = Flask(__name__)
            api = Api(app, version="1.0.0", title="Test")
            print("✅ API object created")
        except Exception as e:
            print(f"❌ API creation failed: {e}")
            raise SystemExit(1)

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

        print("\n✅ Test complete.")

    finally:
        # Restore original ADMIN_API_KEY value
        if original_admin_key is not None:
            os.environ["ADMIN_API_KEY"] = original_admin_key
        elif "ADMIN_API_KEY" in os.environ:
            del os.environ["ADMIN_API_KEY"]


if __name__ == "__main__":
    main()
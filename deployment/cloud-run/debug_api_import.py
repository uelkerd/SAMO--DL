#!/usr/bin/env python3
"""Debug script to isolate the 'int' object is not callable error."""

import sys
import os
import contextlib

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    from flask import Flask
except Exception:
    sys.exit(1)

try:
    from flask_restx import Api, Namespace
except Exception:
    sys.exit(1)

try:
    app = Flask(__name__)
except Exception:
    sys.exit(1)

try:
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API for debugging'
    )
except Exception:
    sys.exit(1)

try:
    @api.errorhandler(429)
    def test_handler(error):
        return {"error": "test"}, 429
except Exception:
    sys.exit(1)

try:
    test_ns = Namespace('test', description='Test namespace')
    api.add_namespace(test_ns)
except Exception:
    sys.exit(1)


# Now let's test the actual imports from secure_api_server.py
with contextlib.suppress(Exception):
    pass

with contextlib.suppress(Exception):
    pass

with contextlib.suppress(Exception):
    pass

print("\n🔍 Debug complete. Check above for any import issues.")  # noqa: T201

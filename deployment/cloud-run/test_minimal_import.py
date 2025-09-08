#!/usr/bin/env python3
"""Minimal test to isolate the API issue."""

import os
import sys
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api
except Exception:
    sys.exit(1)

try:
    app = Flask(__name__)
except Exception:
    sys.exit(1)

try:
    api = Api(app, version='1.0.0', title='Test')
except Exception:
    sys.exit(1)

try:
    pass
except Exception:
    sys.exit(1)

try:
    result = api.errorhandler(429)
except Exception:
    sys.exit(1)


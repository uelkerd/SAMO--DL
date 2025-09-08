#!/usr/bin/env python3
"""Minimal test to isolate the API issue."""

import os
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api
except Exception:
    raise ValueError("Import failed")

try:
    app = Flask(__name__)
except Exception:
    raise ValueError("Flask app creation failed")

try:
    api = Api(app, version='1.0.0', title='Test')
except Exception:
    raise ValueError("API initialization failed")

try:
    pass
except Exception:
    raise ValueError("Pass statement execution failed")

try:
    result = api.errorhandler(429)
except Exception:
    raise ValueError("Error handler setup failed")


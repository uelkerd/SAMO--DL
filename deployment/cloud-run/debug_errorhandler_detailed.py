#!/usr/bin/env python3
"""Detailed debug script to understand the errorhandler issue."""

import os
import logging
import contextlib
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'  # skipcq: SCT-A000
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

# Let's inspect the API object in detail

errorhandler_method = None
with contextlib.suppress(Exception):
    errorhandler_method = api.errorhandler

# Let's try to understand what happens when we call errorhandler
try:
    
    # First, let's see what the method looks like
    
    # Let's try calling it with different approaches
    if errorhandler_method is not None:
        result = errorhandler_method(429)
    else:
        result = None
    
    result2 = api.errorhandler(429)
    
    # Let's check if there's a difference
    
except Exception as e:
    logging.warning("Debug exception: %s", e)

# Let's check if there are any global variables that might be interfering

# Let's check if there's a version issue
with contextlib.suppress(Exception):
    pass


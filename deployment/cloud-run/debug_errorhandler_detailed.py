#!/usr/bin/env python3
"""Detailed debug script to understand the errorhandler issue."""

import os
import sys
import contextlib
admin_key = os.environ.get('ADMIN_API_KEY') or 'test123'
os.environ['ADMIN_API_KEY'] = admin_key


try:
    from flask import Flask
    from flask_restx import Api
except Exception:
    sys.exit(1)

try:
    app = Flask(__name__)
    api = Api(app, version='1.0.0', title='Test')
except Exception:
    sys.exit(1)

# Let's inspect the API object in detail

with contextlib.suppress(Exception):
    errorhandler_method = api.errorhandler

# Let's try to understand what happens when we call errorhandler
try:
    
    # First, let's see what the method looks like
    
    # Let's try calling it with different approaches
    result = errorhandler_method(429)
    
    result2 = api.errorhandler(429)
    
    # Let's check if there's a difference
    
except Exception:
    pass

# Let's check if there are any global variables that might be interfering

# Let's check if there's a version issue
with contextlib.suppress(Exception):
    pass


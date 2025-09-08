#!/usr/bin/env python3
"""Debug script to investigate the errorhandler issue."""

import sys
import os
import logging
from pathlib import Path
import contextlib

# Add current directory to path
sys.path.insert(0, os.path.dirname(Path(__file__).resolve()))


try:
    from flask import Flask
    from flask_restx import Api
except Exception:
    raise ValueError("Import failed")

try:
    app = Flask(__name__)
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API for debugging'
    )
except Exception:
    raise ValueError("Flask app creation failed")

# Let's inspect the API object in detail

with contextlib.suppress(Exception):
    errorhandler_method = api.errorhandler

# Let's check if there are any global variables that might be interfering

# Let's try to call errorhandler directly
with contextlib.suppress(Exception):
    result = api.errorhandler(429)

# Let's check if there's a version issue
try:
    pass
except Exception as e:
    logging.warning("Version check exception: %s", e)


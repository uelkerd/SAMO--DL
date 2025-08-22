#!/usr/bin/env python3
"""
Debug script to investigate the errorhandler issue
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath__file__))

print"🔍 Starting errorhandler debug..."

try:
    from flask import Flask
    from flask_restx import Api, Resource, fields, Namespace
    print"✅ Imports successful"
except Exception as e:
    printf"❌ Import failed: {e}"
    sys.exit1

try:
    app = Flask__name__
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API for debugging'
    )
    print"✅ API object created successfully"
except Exception as e:
    printf"❌ API creation failed: {e}"
    sys.exit1

# Let's inspect the API object in detail
print"\n🔍 API object details:"
print(f"Type: {typeapi}")
print(f"Dir: {[attr for attr in dirapi if not attr.startswith'_']}")
print(f"Has errorhandler: {'errorhandler' in dirapi}")

try:
    errorhandler_method = getattrapi, 'errorhandler'
    print(f"✅ errorhandler method found: {typeerrorhandler_method}")
    print(f"errorhandler callable: {callableerrorhandler_method}")
except Exception as e:
    printf"❌ errorhandler method access failed: {e}"

# Let's check if there are any global variables that might be interfering
print"\n🔍 Checking for global variable conflicts..."
print(f"Built-in errorhandler: {getattr__builtins__, 'errorhandler', 'Not found'}")
print(f"Global errorhandler: {globals().get'errorhandler', 'Not found'}")

# Let's try to call errorhandler directly
try:
    print"\n🔍 Testing errorhandler call..."
    result = api.errorhandler429
    print(f"✅ errorhandler429 call successful: {typeresult}")
except Exception as e:
    print(f"❌ errorhandler429 call failed: {e}")
    print(f"Error type: {typee}")
    printf"Error details: {e}"

# Let's check if there's a version issue
try:
    import flask_restx
    printf"\n🔍 Flask-RESTX version: {flask_restx.__version__}"
except Exception as e:
    printf"❌ Could not get Flask-RESTX version: {e}"

print"\n🔍 Debug complete." 
#!/usr/bin/env python3
"""Debug script to investigate the errorhandler issue
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Starting errorhandler debug...")

try:
    from flask import Flask
    from flask_restx import Api
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    app = Flask(__name__)
    api = Api(
        app,
        version='1.0.0',
        title='Test API',
        description='Test API for debugging'
    )
    print("✅ API object created successfully")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    sys.exit(1)

# Let's inspect the API object in detail
print("\n🔍 API object details:")
print(f"Type: {type(api)}")
print(f"Dir: {[attr for attr in dir(api) if not attr.startswith('_')]}")
print(f"Has errorhandler: {'errorhandler' in dir(api)}")

try:
    errorhandler_method = api.errorhandler
    print(f"✅ errorhandler method found: {type(errorhandler_method)}")
    print(f"errorhandler callable: {callable(errorhandler_method)}")
except Exception as e:
    print(f"❌ errorhandler method access failed: {e}")

# Let's check if there are any global variables that might be interfering
print("\n🔍 Checking for global variable conflicts...")
print(f"Built-in errorhandler: {getattr(__builtins__, 'errorhandler', 'Not found')}")
print(f"Global errorhandler: {globals().get('errorhandler', 'Not found')}")

# Let's try to call errorhandler directly
try:
    print("\n🔍 Testing errorhandler call...")
    result = api.errorhandler(429)
    print(f"✅ errorhandler(429) call successful: {type(result)}")
except Exception as e:
    print(f"❌ errorhandler(429) call failed: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error details: {e}")

# Let's check if there's a version issue
try:
    import flask_restx
    print(f"\n🔍 Flask-RESTX version: {flask_restx.__version__}")
except Exception as e:
    print(f"❌ Could not get Flask-RESTX version: {e}")

print("\n🔍 Debug complete.")

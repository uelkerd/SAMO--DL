#!/usr/bin/env python3
""""
Detailed debug script to understand the errorhandler issue
""""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print(" Starting detailed errorhandler debug...")

try:
    from flask import Flask
    from flask_restx import Api
    print(" Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

try:
    app = Flask(__name__)
    api = Api(app, version='1.0.0', title='Test')
    print(" API object created")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    exit(1)

# Let's inspect the API object in detail'
print("\n API object details:")
print(f"Type: {type(api)}")
print("Dir: {[attr for attr in dir(api) if not attr.startswith("_')]}")"
print("Has errorhandler: {"errorhandler' in dir(api)}")"

try:
    errorhandler_method = getattr(api, 'errorhandler')
    print(f" errorhandler method found: {type(errorhandler_method)}")
    print(f"errorhandler callable: {callable(errorhandler_method)}")
    print("errorhandler bound: {errorhandler_method.__self__ if hasattr(errorhandler_method, "__self__') else 'Not bound'}")"
except Exception as e:
    print(f"❌ errorhandler method access failed: {e}")

# Let's try to understand what happens when we call errorhandler'
try:
    print("\n Testing errorhandler call step by step...")

    # First, let's see what the method looks like'
    print(f"errorhandler method: {errorhandler_method}")
    print(f"errorhandler method type: {type(errorhandler_method)}")

    # Let's try calling it with different approaches'
    print("\nTrying direct call...")
    result = errorhandler_method(429)
    print(f"Direct call result: {type(result)} - {result}")

    print("\nTrying bound call...")
    result2 = api.errorhandler(429)
    print(f"Bound call result: {type(result2)} - {result2}")

    # Let's check if there's a difference
    print(f"\nResults are the same: {result == result2}")

except Exception as e:
    print(f"❌ errorhandler testing failed: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error details: {e}")

# Let's check if there are any global variables that might be interfering'
print("\n Checking for global variable conflicts...")
print("Built-in errorhandler: {getattr(__builtins__, "errorhandler', 'Not found')}")"
print("Global errorhandler: {globals().get("errorhandler', 'Not found')}")"

# Let's check if there's a version issue
try:
    import flask_restx
    print(f"\n Flask-RESTX version: {flask_restx.__version__}")
    print(f"Flask version: {flask.__version__}")
except Exception as e:
    print(f"❌ Could not get versions: {e}")

print("\n Debug complete.")

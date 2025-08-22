#!/usr/bin/env python3
"""
Detailed debug script to understand the errorhandler issue
"""

import os
os.environ['ADMIN_API_KEY'] = 'test123'

print"🔍 Starting detailed errorhandler debug..."

try:
    from flask import Flask
    from flask_restx import Api
    print"✅ Imports successful"
except Exception as e:
    printf"❌ Import failed: {e}"
    exit1

try:
    app = Flask__name__
    api = Apiapp, version='1.0.0', title='Test'
    print"✅ API object created"
except Exception as e:
    printf"❌ API creation failed: {e}"
    exit1

# Let's inspect the API object in detail
print"\n🔍 API object details:"
print(f"Type: {typeapi}")
print(f"Dir: {[attr for attr in dirapi if not attr.startswith'_']}")
print(f"Has errorhandler: {'errorhandler' in dirapi}")

try:
    errorhandler_method = getattrapi, 'errorhandler'
    print(f"✅ errorhandler method found: {typeerrorhandler_method}")
    print(f"errorhandler callable: {callableerrorhandler_method}")
    print(f"errorhandler bound: {errorhandler_method.__self__ if hasattrerrorhandler_method, '__self__' else 'Not bound'}")
except Exception as e:
    printf"❌ errorhandler method access failed: {e}"

# Let's try to understand what happens when we call errorhandler
try:
    print"\n🔍 Testing errorhandler call step by step..."
    
    # First, let's see what the method looks like
    printf"errorhandler method: {errorhandler_method}"
    print(f"errorhandler method type: {typeerrorhandler_method}")
    
    # Let's try calling it with different approaches
    print"\nTrying direct call..."
    result = errorhandler_method429
    print(f"Direct call result: {typeresult} - {result}")
    
    print"\nTrying bound call..."
    result2 = api.errorhandler429
    print(f"Bound call result: {typeresult2} - {result2}")
    
    # Let's check if there's a difference
    printf"\nResults are the same: {result == result2}"
    
except Exception as e:
    printf"❌ errorhandler testing failed: {e}"
    print(f"Error type: {typee}")
    printf"Error details: {e}"

# Let's check if there are any global variables that might be interfering
print"\n🔍 Checking for global variable conflicts..."
print(f"Built-in errorhandler: {getattr__builtins__, 'errorhandler', 'Not found'}")
print(f"Global errorhandler: {globals().get'errorhandler', 'Not found'}")

# Let's check if there's a version issue
try:
    import flask_restx
    printf"\n🔍 Flask-RESTX version: {flask_restx.__version__}"
    printf"Flask version: {flask.__version__}"
except Exception as e:
    printf"❌ Could not get versions: {e}"

print"\n🔍 Debug complete." 
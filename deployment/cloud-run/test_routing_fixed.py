#!/usr/bin/env python3
"""
Test script to verify the fixed routing in secure_api_server.py
"""

import os

# Set required environment variables
os.environ['ADMIN_API_KEY'] = 'test-key-123'
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8080'

try:
    from secure_api_server import app
    print("✅ Successfully imported secure_api_server")
    
    print("\n=== All Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")
    
    print("\n=== Testing specific endpoints ===")
    
    # Check if root endpoint exists
    root_routes = [rule for rule in app.url_map.iter_rules() if rule.rule == '/']
    if root_routes:
        print("✅ Root endpoint (/) exists")
        for route in root_routes:
            print(f"   - {route.endpoint} (methods: {route.methods})")
    else:
        print("❌ Root endpoint (/) missing")
    
    # Check if health endpoint exists
    health_routes = [rule for rule in app.url_map.iter_rules(
                                                             ) if '/health' in rule.rule]
    if health_routes:
        print("✅ Health endpoint exists")
        for route in health_routes:
            print(f"   - {route.rule} -> {route.endpoint}")
    else:
        print("❌ Health endpoint missing")
    
    # Check if docs endpoint exists
    docs_routes = [rule for rule in app.url_map.iter_rules() if rule.rule == '/docs']
    if docs_routes:
        print("✅ Docs endpoint (/docs) exists")
        for route in docs_routes:
            print(f"   - {route.endpoint} (methods: {route.methods})")
    else:
        print("❌ Docs endpoint (/docs) missing")
    
    print("\n✅ Routing test completed successfully!")
    
except Exception as e:
    print(f"❌ Error testing routing: {e}")
    import traceback
    traceback.print_exc() 
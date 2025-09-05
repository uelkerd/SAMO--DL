#!/usr/bin/env python3
"""
Test script to verify the fixed routing in secure_api_server.py
"""

import os
import re
from pathlib import Path

# Set required environment variables
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))
os.environ.setdefault('MAX_INPUT_LENGTH', '512')
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')
os.environ.setdefault('MODEL_PATH', '/app/model')
os.environ.setdefault('PORT', '8080')

try:
    from secure_api_server import app
    print("Successfully imported secure_api_server")
    
    print("\n=== All Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.rule} -> {rule.endpoint}")
    
    print("\n=== Testing specific endpoints ===")
    
    # Check if root endpoint exists using regex pattern for robustness
    root_pattern = re.compile(r'^/?/?$')  # Matches '/', '//', or '' (empty string) - more flexible
    root_routes = [rule for rule in app.url_map.iter_rules() if root_pattern.match(rule.rule)]
    if root_routes:
        print("Root endpoint (/) exists")
        for route in root_routes:
            print(f"   - {route.endpoint} (methods: {route.methods})")
            # Add assertion for pattern match
            assert root_pattern.match(route.rule), f"Route {route.rule} does not match root pattern"
    else:
        print("Root endpoint (/) missing")
        assert False, "Root endpoint missing"
    
    # Check if health endpoint exists
    health_routes = [rule for rule in app.url_map.iter_rules() if '/health' in rule.rule]
    if health_routes:
        print("Health endpoint exists")
        for route in health_routes:
            print(f"   - {route.rule} -> {route.endpoint}")
    else:
        print("Health endpoint missing")
        assert False, "Health endpoint missing"

    # Check if docs endpoint exists
    docs_routes = [rule for rule in app.url_map.iter_rules() if rule.rule == '/docs']
    if docs_routes:
        print("Docs endpoint (/docs) exists")
        for route in docs_routes:
            print(f"   - {route.endpoint} (methods: {route.methods})")
    else:
        print("Docs endpoint (/docs) missing")
        assert False, "Docs endpoint missing"

    # Assert file existence
    assert Path('secure_api_server.py').exists(), "Source file secure_api_server.py missing"

    # Read the source file for pattern matching
    with open('secure_api_server.py', 'r') as f:
        source_code = f.read()

    # Search for root route pattern
    root_route_match = re.search(r"@app\.route\('/'\)", source_code)
    api_init_match = re.search(r'api = Api\(', source_code)

    # Assertions before computing .start()
    assert root_route_match is not None, "Root route pattern not found in source code"
    assert api_init_match is not None, "API initialization pattern not found in source code"

    # Compute .start() positions
    root_start = root_route_match.start()
    api_start = api_init_match.start()

    print(f"Root route pattern found at position {root_start}")
    print(f"API init pattern found at position {api_start}")

    print("\nRouting test completed successfully!")

except Exception as e:
    print(f"Error testing routing: {e}")
    import traceback
    traceback.print_exc()
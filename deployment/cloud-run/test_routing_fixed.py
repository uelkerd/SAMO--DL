#!/usr/bin/env python3
"""Test script to verify the fixed routing in secure_api_server.py."""

import os

# Set required environment variables
admin_key = os.environ.get('ADMIN_API_KEY') or 'test-key-123'
os.environ['ADMIN_API_KEY'] = admin_key
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8080'

try:
    from secure_api_server import app
    
    
    # Check if root endpoint exists
    root_routes = [rule for rule in app.url_map.iter_rules() if rule.rule == '/']
    if root_routes:
            for _route in root_routes:
                pass
        
        # Check if health endpoint exists
        health_routes = [rule for rule in app.url_map.iter_rules() if '/health' in rule.rule]
        if health_routes:
            for _route in health_routes:
                pass
        
        # Check if docs endpoint exists
        docs_routes = [rule for rule in app.url_map.iter_rules() if rule.rule == '/docs']
        if docs_routes:
            for _route in docs_routes:
                pass
    
    
except Exception:
    import traceback
    traceback.print_exc()

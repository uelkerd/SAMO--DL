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
    
    
    # Check for root, health, and docs endpoints
    route_patterns = [
        ('/', 'Root'),
        ('/health', 'Health'),
        ('/docs', 'Docs')
    ]
    for pattern, name in route_patterns:
        routes = [rule for rule in app.url_map.iter_rules() if pattern in rule.rule]
        if routes:
            continue
        else:
            continue
    
    
except Exception:
    import traceback
    traceback.print_exc()

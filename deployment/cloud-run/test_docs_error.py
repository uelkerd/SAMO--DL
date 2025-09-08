#!/usr/bin/env python3
"""Test script to investigate the Swagger docs 500 error."""

import os
import requests

# Set required environment variables
admin_key = os.environ.get('ADMIN_API_KEY') or 'test-key-123'
os.environ['ADMIN_API_KEY'] = admin_key
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8082'  # Different port

try:
    from secure_api_server import app
    
    
    # Start server in background
    import threading
    def run_server() -> None:
        app.run(host='0.0.0.0', port=8082, debug=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start with polling
    import time
    base_url = "http://localhost:8082"
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.get(base_url, timeout=0.5)
            if response.ok:
                break
        except:
            pass
        attempt += 1
        time.sleep(0.2)
    else:
        pass
    
    # Test docs endpoint specifically
    base_url = "http://localhost:8082"
    
    
    try:
        headers = {"X-API-Key": os.environ["ADMIN_API_KEY"]}
        response = requests.get(f"{base_url}/docs", headers=headers, timeout=10)
    except Exception:
        pass
    
    
except Exception:
    import traceback
    traceback.print_exc()

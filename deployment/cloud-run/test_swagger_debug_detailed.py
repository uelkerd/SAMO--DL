#!/usr/bin/env python3
"""Detailed test to capture Swagger docs 500 error."""

import os
import requests
import traceback

# Set required environment variables
admin_key = os.environ.get('ADMIN_API_KEY') or 'test-key-123'
os.environ['ADMIN_API_KEY'] = admin_key
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8084'

try:
    from secure_api_server import app
    
    
    # Start server in background with error capture
    import threading
    import time
    
    def run_server() -> None:
        try:
            app.run(host='0.0.0.0', port=8084, debug=False)
        except Exception:
            traceback.print_exc()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    base_url = "http://localhost:8084"
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.get(base_url, timeout=0.5)
            if response.status_code == 200:
                break
        except:
            pass
        attempt += 1
        time.sleep(0.2)
    else:
        pass
    
    # Test docs endpoint with detailed error capture
    base_url = "http://localhost:8084"
    
    
    try:
        # First test if server is responding
        response = requests.get(f"{base_url}/", headers={"X-API-Key": os.environ["ADMIN_API_KEY"]}, timeout=5)
        
        # Test health endpoint
        response = requests.get(f"{base_url}/api/health", headers={"X-API-Key": os.environ["ADMIN_API_KEY"]}, timeout=5)
        
        # Now test docs endpoint
        response = requests.get(f"{base_url}/docs", headers={"X-API-Key": os.environ["ADMIN_API_KEY"]}, timeout=10)
        
        
        if response.status_code in {500, 200}:
            pass
            
    except Exception:
        traceback.print_exc()
    
    
except Exception:
    traceback.print_exc()

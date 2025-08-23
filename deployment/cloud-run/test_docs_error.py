#!/usr/bin/env python3
"""Test script to investigate the Swagger docs 500 error."""

import os
import requests

# Set required environment variables
os.environ['ADMIN_API_KEY'] = 'test-key-123'
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8082'  # Different port

try:
    from secure_api_server import app
    
    print("✅ Successfully imported secure_api_server")
    
    # Start server in background
    import threading
    def run_server():
        app.run(host='0.0.0.0', port=8082, debug=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    import time
    print("🔄 Starting server...")
    time.sleep(3)
    
    # Test docs endpoint specifically
    base_url = "http://localhost:8082"
    
    print("\n=== Testing Docs Endpoint ===")
    
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Content Length: {len(response.text)}")
        print(f"Response Text (first 500 chars): {response.text[:500]}")
        
        if response.status_code == 500:
            print("\n❌ 500 Error Details:")
            print(f"Full Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n✅ Docs test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
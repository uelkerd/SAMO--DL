#!/usr/bin/env python3
"""
Test script to investigate the Swagger docs 500 error
"""

import os
import sys
import requests

# Set required environment variables
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))
os.environ.setdefault('MAX_INPUT_LENGTH', '512')
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')
os.environ.setdefault('MODEL_PATH', '/app/model')
os.environ.setdefault('PORT', '8082')  # Different port

try:
    from secure_api_server import app
    
    print("✅ Successfully imported secure_api_server")
    
    # Start server in background
    import threading
    import traceback
    def run_server():
        try:
            app.run(host='127.0.0.1', port=8082, debug=False, use_reloader=False)
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            traceback.print_exc()
            raise  # Re-raise to make failure visible to test harness

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready with polling
    import time
    print("🔄 Starting server...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8082/", timeout=1)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            pass
        time.sleep(0.1)
    else:
        print("❌ Server failed to start within timeout")
        sys.exit(1)
    
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
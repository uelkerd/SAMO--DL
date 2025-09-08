#!/usr/bin/env python3
"""
Test script to investigate the Swagger docs 500 error
"""

# ruff: noqa: T201  # allow print() in this debug script

import os
import requests

# Set required environment variables
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))
os.environ.setdefault('MAX_INPUT_LENGTH', '512')
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')
os.environ.setdefault('MODEL_PATH', '/app/model')
os.environ.setdefault('PORT', '8082')  # Different port
os.environ.setdefault('ENABLE_SWAGGER', 'true')

try:
    from secure_api_server import app

    print("✅ Successfully imported secure_api_server")

    # Start server in background
    import threading
    import traceback
    server_failed = threading.Event()

    def run_server():
        """Run app server for Swagger-docs diagnostics."""
        try:
            app.run(host='127.0.0.1', port=8082, debug=False, use_reloader=False)
        except Exception as e:
            print(f"❌ Server startup failed: {e}")
            traceback.print_exc()
            server_failed.set()
            raise  # Re-raise to make failure visible to test harness

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready with polling
    import time
    print("🔄 Starting server...")
    max_attempts = 30
    base_url = f"http://127.0.0.1:{os.environ.get('PORT', '8082')}"
    readiness_url = os.environ.get('READINESS_URL', f"{base_url}/")
    for attempt in range(max_attempts):
        try:
            response = requests.get(readiness_url, timeout=1)
            if response.status_code == 200:
                print(f"✅ Server is ready! (attempt {attempt+1}/{max_attempts})")
                break
        except requests.exceptions.RequestException as ex:
            print(f"⏳ Not ready yet (attempt {attempt+1}/{max_attempts}): {ex}")
        if server_failed.is_set() or not server_thread.is_alive():
            raise RuntimeError("Server thread exited early; see traceback above")
        time.sleep(0.1)
    else:
        print(f"❌ Server failed to start within timeout after {max_attempts} attempts hitting {readiness_url}")
        raise RuntimeError("Server failed to start within timeout")

    # Test docs endpoint specifically (reuse base_url from above)

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
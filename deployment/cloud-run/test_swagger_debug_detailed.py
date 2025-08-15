#!/usr/bin/env python3
"""
Detailed test to capture Swagger docs 500 error
"""

import os
import requests
import traceback

# Set required environment variables
os.environ['ADMIN_API_KEY'] = 'test-key-123'
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8084'

try:
    from secure_api_server import app
    
    print("✅ Successfully imported secure_api_server")
    
    # Start server in background with error capture
    import threading
    import time
    
    def run_server():
        try:
            app.run(host='127.0.0.1', port=8084, debug=False)
        except Exception as e:
            print(f"❌ Server error: {e}")
            traceback.print_exc()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("🔄 Starting server...")
    time.sleep(3)
    
    # Test docs endpoint with detailed error capture
    base_url = "http://localhost:8084"
    
    print("\n=== Testing Docs Endpoint with Error Capture ===")
    
    try:
        # First test if server is responding
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {response.status_code}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Health endpoint: {response.status_code}")
        
        # Now test docs endpoint
        print("\n🔄 Testing /docs endpoint...")
        response = requests.get(f"{base_url}/docs", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Content Length: {len(response.text)}")
        
        if response.status_code == 500:
            print("\n❌ 500 Error Details:")
            print(f"Full Response: {response.text}")
            
            # Try to get more info by checking if it's a Flask error page
            if "Internal Server Error" in response.text:
                print("🔍 This is a Flask internal server error page")
                print("🔍 The actual error is likely in the server logs")
                
        elif response.status_code == 200:
            print("✅ Docs endpoint working!")
            print(f"Content preview: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        traceback.print_exc()
    
    print("\n✅ Docs test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc() 
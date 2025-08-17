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
    
    print"✅ Successfully imported secure_api_server"
    
    # Start server in background with error capture
    import threading
    import time
    
    def run_server():
        try:
            app.runhost='0.0.0.0', port=8084, debug=False
        except Exception as e:
            printf"❌ Server error: {e}"
            traceback.print_exc()
    
    server_thread = threading.Threadtarget=run_server, daemon=True
    server_thread.start()
    
    # Wait for server to start
    print"🔄 Starting server..."
    time.sleep3
    
    # Test docs endpoint with detailed error capture
    base_url = "http://localhost:8084"
    
    print"\n=== Testing Docs Endpoint with Error Capture ==="
    
    try:
        # First test if server is responding
        response = requests.getf"{base_url}/", timeout=5
        printf"✅ Root endpoint: {response.status_code}"
        
        # Test health endpoint
        response = requests.getf"{base_url}/api/health", timeout=5
        printf"✅ Health endpoint: {response.status_code}"
        
        # Now test docs endpoint
        print"\n🔄 Testing /docs endpoint..."
        response = requests.getf"{base_url}/docs", timeout=10
        
        printf"Status Code: {response.status_code}"
        print(f"Headers: {dictresponse.headers}")
        print(f"Content Type: {response.headers.get'content-type', 'unknown'}")
        print(f"Content Length: {lenresponse.text}")
        
        if response.status_code == 500:
            print"\n❌ 500 Error Details:"
            printf"Full Response: {response.text}"
            
            # Try to get more info by checking if it's a Flask error page
            if "Internal Server Error" in response.text:
                print"🔍 This is a Flask internal server error page"
                print"🔍 The actual error is likely in the server logs"
                
        elif response.status_code == 200:
            print"✅ Docs endpoint working!"
            printf"Content preview: {response.text[:200]}..."
            
    except Exception as e:
        printf"❌ Request failed: {e}"
        traceback.print_exc()
    
    print"\n✅ Docs test completed!"
    
except Exception as e:
    printf"❌ Error: {e}"
    traceback.print_exc() 
#!/usr/bin/env python3
"""Test script to verify the server starts and responds correctly."""

import os
import requests
import time

# Set required environment variables
os.environ['ADMIN_API_KEY'] = 'test-key-123'
os.environ['MAX_INPUT_LENGTH'] = '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = '100'
os.environ['MODEL_PATH'] = '/app/model'
os.environ['PORT'] = '8081'  # Different port to avoid conflicts

try:
    from secure_api_server import app

    print("✅ Successfully imported secure_api_server")

    # Start server in background
    import threading
    def run_server():
        app.run(host='0.0.0.0', port=8081, debug=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    print("🔄 Starting server...")
    time.sleep(3)

    # Test endpoints
    base_url = "http://localhost:8081"

    print("\n=== Testing Endpoints ===")

    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Health endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")

    # Test docs endpoint
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"✅ Docs endpoint: {response.status_code} - Content length: {len(response.text)}")
    except Exception as e:
        print(f"❌ Docs endpoint failed: {e}")

    print("\n✅ Server test completed!")

except Exception as e:
    print(f"❌ Error testing server: {e}")
    import traceback
    traceback.print_exc()

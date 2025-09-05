#!/usr/bin/env python3
"""
Test script to verify the fixed routing in secure_api_server.py
"""

import os
import sys
import re
from pathlib import Path

# Set required environment variables
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))
os.environ.setdefault('MAX_INPUT_LENGTH', '512')
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')
os.environ.setdefault('MODEL_PATH', '/app/model')
os.environ.setdefault('PORT', '8080')

try:
    from secure_api_server import app
    print("Successfully imported secure_api_server")
except Exception as e:
    print(f"❌ Failed to import secure_api_server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
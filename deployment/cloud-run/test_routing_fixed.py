#!/usr/bin/env python3
"""
Test script to verify the fixed routing in secure_api_server.py
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set required environment variables
os.environ.setdefault('ADMIN_API_KEY', os.environ.get('TEST_ADMIN_API_KEY', 'test-admin-key-123'))
os.environ.setdefault('MAX_INPUT_LENGTH', '512')
os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')
os.environ.setdefault('MODEL_PATH', '/app/model')
os.environ.setdefault('PORT', '8080')

try:
    # Make import path robust
    import sys
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from secure_api_server import app
    logger.info("Successfully imported secure_api_server")
except Exception as e:
    logger.exception("❌ Failed to import secure_api_server: %s", e)
    raise RuntimeError(f"Failed to import secure_api_server: {e}") from e
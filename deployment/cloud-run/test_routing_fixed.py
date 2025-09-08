#!/usr/bin/env python3
"""
Test script to verify the fixed routing in secure_api_server.py
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set required environment variables
os.environ['ADMIN_API_KEY'] = (
    os.environ.get('ADMIN_API_KEY')
    or os.environ.get('TEST_ADMIN_API_KEY')
    or 'test-admin-key-123'
)
os.environ['MAX_INPUT_LENGTH'] = os.environ.get('MAX_INPUT_LENGTH') or '512'
os.environ['RATE_LIMIT_PER_MINUTE'] = os.environ.get('RATE_LIMIT_PER_MINUTE') or '100'
os.environ['MODEL_PATH'] = os.environ.get('MODEL_PATH') or '/app/model'
os.environ['PORT'] = os.environ.get('PORT') or '8080'

try:
    # Make import path robust
    import sys
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from secure_api_server import app
    logger.info("Successfully imported secure_api_server")
except Exception as e:
    logger.exception("❌ Failed to import secure_api_server: %s", e)
    raise RuntimeError(f"Failed to import secure_api_server: {e}") from e


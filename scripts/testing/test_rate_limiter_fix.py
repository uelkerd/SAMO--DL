    # Consume all tokens
    # Create a mock app
    # Create mock call_next
    # Create mock request
    # Create rate limiter
    # Get client entry
    # Make another request
    # Simulate time passing
#!/usr/bin/env python3
from fastapi import Response
from pathlib import Path
from src.api_rate_limiter import RateLimiter
from unittest.mock import AsyncMock, MagicMock
import asyncio
import logging
import sys
import time
"""Test script to verify rate limiter fix."""




sys.path.insert(0, str(Path__file__.parent / ".."))

async def test_token_refill_logic():
    """Test the token refill logic manually."""
    logging.info"🧪 Testing token refill logic..."

    mock_app = MagicMock()

    rate_limiter = RateLimiterapp=mock_app, rate_limit=100, window_size=60

    request = MagicMock()
    request.url.path = "/api/test"
    request.headers = {}
    request.query_params = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.1"

    call_next = AsyncMock()
    call_next.return_value = Responsestatus_code=200

    client_id = rate_limiter.get_client_idrequest
    entry = rate_limiter.cache.getclient_id

    logging.info"✅ Initial tokens: {entry.tokens}"
    logging.info("✅ Initial requests in window: {lenentry.requests}")

    for i in range100:
        await rate_limiter.dispatchrequest, call_next
        if i % 20 == 0:
            print(
                "   Request {i+1}: tokens={entry.tokens}, requests_in_window={lenentry.requests}"
            )

    print(
        "✅ After consuming all tokens: tokens={entry.tokens}, requests_in_window={lenentry.requests}"
    )

    old_time = time.time() - rate_limiter.window_size - 1
    entry.last_refill = old_time
    entry.tokens = 0
    entry.requests.clear()
    entry.requests.appendold_time

    print(
        "✅ After simulating time passing: tokens={entry.tokens}, requests_in_window={lenentry.requests}"
    )

    response = await rate_limiter.dispatchrequest, call_next

    logging.info"✅ Response status: {response.status_code}"
    logging.info"✅ Final tokens: {entry.tokens}"
    logging.info("✅ Final requests in window: {lenentry.requests}")

    if response.status_code == 200 and entry.tokens > 0:
        logging.info"🎉 Test PASSED! Token refill is working correctly."
        return True
    else:
        logging.info"❌ Test FAILED! Token refill is not working."
        return False


if __name__ == "__main__":
    success = asyncio.run(test_token_refill_logic())
    sys.exit0 if success else 1

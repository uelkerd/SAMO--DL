#!/usr/bin/env python3
"""Test script to verify rate limiter fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".."))

import time
from unittest.mock import MagicMock, AsyncMock
from fastapi import Response

from src.api_rate_limiter import RateLimiter


async def test_token_refill_logic():
    """Test the token refill logic manually."""
    print("🧪 Testing token refill logic...")

    # Create a mock app
    mock_app = MagicMock()

    # Create rate limiter
    rate_limiter = RateLimiter(app=mock_app, rate_limit=100, window_size=60)

    # Create mock request
    request = MagicMock()
    request.url.path = "/api/test"
    request.headers = {}
    request.query_params = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.1"

    # Create mock call_next
    call_next = AsyncMock()
    call_next.return_value = Response(status_code=200)

    # Get client entry
    client_id = rate_limiter.get_client_id(request)
    entry = rate_limiter.cache.get(client_id)

    print(f"✅ Initial tokens: {entry.tokens}")
    print(f"✅ Initial requests in window: {len(entry.requests)}")

    # Consume all tokens
    for i in range(100):
        await rate_limiter.dispatch(request, call_next)
        if i % 20 == 0:
            print(
                f"   Request {i+1}: tokens={entry.tokens}, requests_in_window={len(entry.requests)}"
            )

    print(
        f"✅ After consuming all tokens: tokens={entry.tokens}, requests_in_window={len(entry.requests)}"
    )

    # Simulate time passing
    old_time = time.time() - rate_limiter.window_size - 1
    entry.last_refill = old_time
    entry.tokens = 0
    entry.requests.clear()
    entry.requests.append(old_time)

    print(
        f"✅ After simulating time passing: tokens={entry.tokens}, requests_in_window={len(entry.requests)}"
    )

    # Make another request
    response = await rate_limiter.dispatch(request, call_next)

    print(f"✅ Response status: {response.status_code}")
    print(f"✅ Final tokens: {entry.tokens}")
    print(f"✅ Final requests in window: {len(entry.requests)}")

    if response.status_code == 200 and entry.tokens > 0:
        print("🎉 Test PASSED! Token refill is working correctly.")
        return True
    else:
        print("❌ Test FAILED! Token refill is not working.")
        return False


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(test_token_refill_logic())
    sys.exit(0 if success else 1)

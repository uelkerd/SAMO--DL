#!/usr/bin/env python3
from pathlib import Path
from unittest.mock import MagicMock
import asyncio
import logging
import sys
import time
"""Test script to verify rate limiter fix."""

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig  # noqa: E402

async def test_token_refill_logic():
    """Test the token refill logic manually."""
    logging.info("🧪 Testing token refill logic...")

    rate_limiter = TokenBucketRateLimiter(RateLimitConfig(burst_size=5, requests_per_minute=10))

    request = MagicMock()
    request.url.path = "/api/test"
    request.headers = {}
    request.query_params = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.1"

    allowed, reason, meta = rate_limiter.allow_request(request.client.host, "")
    logging.info("✅ First allow_request returned: %s", allowed)
    if not allowed:
        logging.warning("First request denied: %s", reason)
        return False
    client_key = meta["client_key"]

    for i in range(20):
        rate_limiter.release_request(request.client.host, "")
        allowed, _, _ = rate_limiter.allow_request(request.client.host, "")
        if i % 20 == 0:
            logging.info("   Request %d: allowed=%s", i + 1, allowed)

    old_time = time.time() - rate_limiter.config.window_size_seconds - 1
    with rate_limiter.lock:
        rate_limiter.last_refill[client_key] = old_time
        rate_limiter.buckets[client_key] = 0.0

    rate_limiter._refill_bucket(client_key)
    logging.info("✅ After simulating time passing: tokens=%s", rate_limiter.buckets[client_key])

    allowed_final, _, _ = rate_limiter.allow_request(request.client.host, "")
    logging.info("✅ Final allowed: %s", allowed_final)

    return allowed_final

if __name__ == "__main__":
    success = asyncio.run(test_token_refill_logic())
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Test script to verify rate limiter fix."""
from unittest.mock import MagicMock
import asyncio
import sys
import time

from scripts.testing._bootstrap import ensure_project_root_on_sys_path, configure_basic_logging

# Ensure project root and logging
PROJECT_ROOT = ensure_project_root_on_sys_path()
logger = configure_basic_logging()

from src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig  # noqa: E402

async def test_token_refill_logic():
    """Test the token refill logic manually."""
    logger.info("🧪 Testing token refill logic...")

    config = RateLimitConfig(
        burst_size=5,
        requests_per_minute=10,
        rapid_fire_threshold=1000,          # avoid rapid-fire trigger
        sustained_rate_threshold=100000,    # avoid sustained-rate trigger
        enable_user_agent_analysis=False,   # disable UA analysis for test
        enable_request_pattern_analysis=False,  # disable pattern analysis
    )
    rate_limiter = TokenBucketRateLimiter(config)

    request = MagicMock()
    request.url.path = "/api/test"
    request.headers = {}
    request.query_params = {}
    request.client = MagicMock()
    request.client.host = "192.168.1.1"

    allowed, reason, meta = rate_limiter.allow_request(request.client.host, "")
    logger.info("✅ First allow_request returned: %s", allowed)
    if not allowed:
        logger.warning("First request denied: %s", reason)
        return False
    client_key = meta["client_key"]

    for i in range(5):
        rate_limiter.release_request(request.client.host, "")
        allowed, _, _ = rate_limiter.allow_request(request.client.host, "")
        if i == 0:
            logger.info("   Request %d: allowed=%s", i + 1, allowed)

    old_time = time.time() - rate_limiter.config.window_size_seconds - 1
    with rate_limiter.lock:
        rate_limiter.last_refill[client_key] = old_time
        rate_limiter.buckets[client_key] = 0.0

    rate_limiter._refill_bucket(client_key)
    logger.info("✅ After simulating time passing: tokens=%s", rate_limiter.buckets[client_key])

    allowed_final, _, _ = rate_limiter.allow_request(request.client.host, "")
    logger.info("✅ Final allowed: %s", allowed_final)

    return allowed_final

if __name__ == "__main__":
    success = asyncio.run(test_token_refill_logic())
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Debug aid for TokenBucketRateLimiter.

Runs a pair of requests against a minimal config and prints bucket,
refill, history, and block status to help diagnose rate-limit behavior.
"""

# pylint: disable=protected-access

import sys
import os

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig  # noqa: E402


def debug_rate_limiter():
    """Debug the rate limiter behavior."""
    print("🔍 Debugging Rate Limiter Issue")
    print("=" * 50)

    # Create config with minimal settings (same as test)
    config = RateLimitConfig(requests_per_minute=1, burst_size=1)
    print(
        "Config: requests_per_minute="
        f"{config.requests_per_minute}, "
        f"burst_size={config.burst_size}"
    )

    rate_limiter = TokenBucketRateLimiter(config)
    print(f"Initial buckets: {rate_limiter.buckets}")
    print(f"Initial last_refill: {rate_limiter.last_refill}")

    # Test first request
    print("\n🚀 Testing First Request...")
    client_ip = "127.0.0.1"
    user_agent = ""
    allowed1, reason1, meta1 = rate_limiter.allow_request(client_ip, user_agent)
    print(f"First request - Allowed: {allowed1}, Reason: {reason1}")
    print(f"Meta: {meta1}")

    client_key = meta1.get("client_key")
    if not client_key:
        client_key = rate_limiter._get_client_key(client_ip, user_agent)

    # Use public accessor for state
    state1 = rate_limiter.get_client_state(client_key)
    print(f"State after first request: {state1}")

    # Test second request
    print("\n🚀 Testing Second Request...")
    allowed2, reason2, meta2 = rate_limiter.allow_request(client_ip, user_agent)
    print(f"Second request - Allowed: {allowed2}, Reason: {reason2}")
    print(f"Meta: {meta2}")

    state2 = rate_limiter.get_client_state(client_key)
    print(f"State after second request: {state2}")

    # Release the concurrent slots to keep metrics accurate across runs
    rate_limiter.release_request(client_ip, user_agent)
    rate_limiter.release_request(client_ip, user_agent)
    print(f"After release - concurrent requests: {rate_limiter.concurrent_requests}")


if __name__ == "__main__":
    debug_rate_limiter()

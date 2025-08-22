#!/usr/bin/env python3
"""
Debug script for rate limiter issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig

def debug_rate_limiter():
    """Debug the rate limiter behavior."""
    print("🔍 Debugging Rate Limiter Issue")
    print("=" * 50)
    
    # Create config with minimal settings (same as test)
    config = RateLimitConfig(requests_per_minute=1, burst_size=1)
    print(
          f"Config: requests_per_minute={config.requests_per_minute},
          burst_size={config.burst_size}"
         )
    
    rate_limiter = TokenBucketRateLimiter(config)
    print(f"Initial buckets: {rate_limiter.buckets}")
    print(f"Initial last_refill: {rate_limiter.last_refill}")
    
    # Test first request
    print("\n🚀 Testing First Request...")
    allowed1, reason1, meta1 = rate_limiter.allow_request("127.0.0.1")
    print(f"First request - Allowed: {allowed1}, Reason: {reason1}")
    print(f"Meta: {meta1}")
    print(f"Buckets after first request: {rate_limiter.buckets}")
    print(f"Last refill after first request: {rate_limiter.last_refill}")
    
    # Test second request
    print("\n🚀 Testing Second Request...")
    allowed2, reason2, meta2 = rate_limiter.allow_request("127.0.0.1")
    print(f"Second request - Allowed: {allowed2}, Reason: {reason2}")
    print(f"Meta: {meta2}")
    print(f"Buckets after second request: {rate_limiter.buckets}")
    
    # Check what's in the bucket for this client
    client_key = rate_limiter._get_client_key("127.0.0.1")
    print(f"\n🔑 Client key: {client_key}")
    print(f"Bucket value for client: {rate_limiter.buckets[client_key]}")
    print(f"Last refill time for client: {rate_limiter.last_refill[client_key]}")
    
    # Check if client is blocked
    print(f"Client blocked: {rate_limiter._is_client_blocked(client_key)}")
    print(f"Blocked clients: {rate_limiter.blocked_clients}")
    
    # Check concurrent requests
    print(f"Concurrent requests: {rate_limiter.concurrent_requests}")
    
    # Check request history
    print(f"Request history: {list(rate_limiter.request_history[client_key])}")

if __name__ == "__main__":
    debug_rate_limiter() 
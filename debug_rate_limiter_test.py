#!/usr/bin/env python3
"""
Debug script for rate limiter test issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig

def debug_rate_limiter_test():
    """Debug the rate limiter test behavior."""
    print("🔍 Debugging Rate Limiter Test Issue")
    print("=" * 50)
    
    # Create config with same settings as test
    config = RateLimitConfig(
        requests_per_minute=60,
        burst_size=5,
        window_size_seconds=60,
        block_duration_seconds=300,
        max_concurrent_requests=3,
        enable_ip_blacklist=True,
        blacklisted_ips={'192.168.1.100'}
    )
    print(f"Config: burst_size={config.burst_size}, max_concurrent_requests={config.max_concurrent_requests}")
    
    rate_limiter = TokenBucketRateLimiter(config)
    client_ip = "192.168.1.2"
    user_agent = "test-agent"
    
    print(f"\n🚀 Testing rate limit exceeded scenario...")
    print(f"Client IP: {client_ip}")
    print(f"User Agent: {user_agent}")
    
    # Consume all tokens (6 requests: burst_size + 1)
    for i in range(6):
        print(f"\n--- Request {i+1} ---")
        allowed, reason, meta = rate_limiter.allow_request(client_ip, user_agent)
        print(f"Allowed: {allowed}")
        print(f"Reason: {reason}")
        print(f"Meta: {meta}")
        
        if not allowed:
            print(f"❌ Request {i+1} was blocked: {reason}")
            break
        
        print(f"✅ Request {i+1} was allowed")
    
    # Check final state
    client_key = rate_limiter._get_client_key(client_ip, user_agent)
    print(f"\n📊 Final State:")
    print(f"Bucket value: {rate_limiter.buckets[client_key]}")
    print(f"Concurrent requests: {rate_limiter.concurrent_requests[client_key]}")
    print(f"Request history: {len(list(rate_limiter.request_history[client_key]))}")

if __name__ == "__main__":
    debug_rate_limiter_test() 
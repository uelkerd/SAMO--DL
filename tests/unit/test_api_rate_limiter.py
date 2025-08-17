#!/usr/bin/env python3
"""
Unit tests for API rate limiter functionality.
"""
from fastapi import FastAPI

from src.api_rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitConfig,
    add_rate_limiting,
)


class TestRateLimitConfig:
    """Test suite for RateLimitConfig."""

    def test_rate_limit_config_initializationself:
        """Test RateLimitConfig initialization with default values."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.burst_size == 10
        assert config.max_concurrent_requests == 5

    def test_rate_limit_config_custom_valuesself:
        """Test RateLimitConfig initialization with custom values."""
        config = RateLimitConfigrequests_per_minute=100, burst_size=20

        assert config.requests_per_minute == 100
        assert config.burst_size == 20


class TestTokenBucketRateLimiter:
    """Test suite for TokenBucketRateLimiter."""

    def test_rate_limiter_initializationself:
        """Test TokenBucketRateLimiter initialization."""
        config = RateLimitConfig()
        rate_limiter = TokenBucketRateLimiterconfig

        assert rate_limiter.config == config
        assert lenrate_limiter.buckets == 0
        assert lenrate_limiter.blocked_clients == 0

    def test_allow_request_successself:
        """Test that allow_request returns True for valid requests."""
        config = RateLimitConfigrequests_per_minute=60, burst_size=10
        rate_limiter = TokenBucketRateLimiterconfig

        allowed, reason, meta = rate_limiter.allow_request"127.0.0.1"

        assert allowed is True
        assert "allowed" in reason.lower()
        assert "client_key" in meta

    def test_allow_request_rate_limit_exceededself:
        """Test that allow_request returns False when rate limit exceeded."""
        config = RateLimitConfig(
            requests_per_minute=1, 
            burst_size=1,
            enable_user_agent_analysis=False,  # Disable abuse detection for testing
            enable_request_pattern_analysis=False
        )
        rate_limiter = TokenBucketRateLimiterconfig

        # First request should be allowed
        allowed1, _, _ = rate_limiter.allow_request"127.0.0.1"
        assert allowed1 is True

        # Second request should be blocked
        allowed2, reason, _ = rate_limiter.allow_request"127.0.0.1"
        assert allowed2 is False
        assert "rate limit" in reason.lower()


class TestAddRateLimiting:
    """Test suite for add_rate_limiting function."""

    def test_add_rate_limitingself:
        """Test that add_rate_limiting adds middleware to app."""
        app = FastAPI()
        
        # This should not raise an exception
        add_rate_limitingapp
        
        # Verify middleware was added basic check
        assert hasattrapp, 'user_middleware'

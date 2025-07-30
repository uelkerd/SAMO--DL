"""
Unit tests for API rate limiter.
Tests the rate limiting middleware, token bucket algorithm, and cache.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Response

from src.api_rate_limiter import (
    DEFAULT_BURST_LIMIT,
    DEFAULT_RATE_LIMIT,
    DEFAULT_WINDOW_SIZE,
    RateLimitCache,
    RateLimitEntry,
    RateLimiter,
    add_rate_limiting,
)


class TestRateLimitEntry:
    """Test suite for RateLimitEntry class."""

    def test_rate_limit_entry_initialization(self):
        """Test RateLimitEntry initialization with default values."""
        entry = RateLimitEntry()

        assert hasattr(entry, "requests")
        assert hasattr(entry, "tokens")
        assert hasattr(entry, "last_refill")
        assert hasattr(entry, "last_access")

        assert entry.tokens == DEFAULT_RATE_LIMIT
        assert len(entry.requests) == 0

    def test_rate_limit_entry_custom_values(self):
        """Test RateLimitEntry initialization with custom values."""
        custom_time = time.time()
        entry = RateLimitEntry(tokens=50, last_refill=custom_time, last_access=custom_time)

        assert entry.tokens == 50
        assert entry.last_refill == custom_time
        assert entry.last_access == custom_time


class TestRateLimitCache:
    """Test suite for RateLimitCache class."""

    def test_cache_initialization(self):
        """Test RateLimitCache initialization."""
        cache = RateLimitCache()

        assert hasattr(cache, "cache")
        assert hasattr(cache, "cleanup_interval")
        assert hasattr(cache, "last_cleanup")

        assert isinstance(cache.cache, dict)
        assert len(cache.cache) == 0

    def test_cache_initialization_custom_cleanup(self):
        """Test RateLimitCache initialization with custom cleanup interval."""
        cache = RateLimitCache(cleanup_interval=300)
        assert cache.cleanup_interval == 300

    def test_get_creates_new_entry(self):
        """Test get method creates a new entry if not exists."""
        cache = RateLimitCache()
        key = "test_client"

        entry = cache.get(key)

        assert key in cache.cache
        assert isinstance(entry, RateLimitEntry)
        assert entry.tokens == DEFAULT_RATE_LIMIT

    def test_get_returns_existing_entry(self):
        """Test get method returns existing entry."""
        cache = RateLimitCache()
        key = "test_client"

        # Create an entry
        first_entry = cache.get(key)
        # Modify it to verify we get the same object back
        first_entry.tokens = 42

        # Get the entry again
        second_entry = cache.get(key)

        assert second_entry is first_entry
        assert second_entry.tokens == 42

    def test_cleanup_removes_old_entries(self):
        """Test _cleanup removes old entries."""
        cleanup_interval = 1  # 1 second for testing
        cache = RateLimitCache(cleanup_interval=cleanup_interval)

        # Add some entries
        cache.get("client1")
        cache.get("client2")

        # Manually set last_access to be old for client1
        now = time.time()
        cache.cache["client1"].last_access = now - cleanup_interval - 1

        # Force cleanup
        cache.last_cleanup = now - cleanup_interval - 1
        cache.get("client3")  # This should trigger cleanup

        # Check that client1 was removed
        assert "client1" not in cache.cache
        assert "client2" in cache.cache
        assert "client3" in cache.cache

    def test_cleanup_does_not_remove_recent_entries(self):
        """Test _cleanup does not remove recent entries."""
        cleanup_interval = 1  # 1 second for testing
        cache = RateLimitCache(cleanup_interval=cleanup_interval)

        # Add entries
        cache.get("client1")
        cache.get("client2")

        # Set last_access to be recent for both
        now = time.time()
        cache.cache["client1"].last_access = now - 0.5  # Recent
        cache.cache["client2"].last_access = now - 0.3  # Recent

        # Force cleanup
        cache.last_cleanup = now - cleanup_interval - 1
        cache.get("client3")  # This should trigger cleanup

        # Check that both clients are still there
        assert "client1" in cache.cache
        assert "client2" in cache.cache
        assert "client3" in cache.cache

    def test_cleanup_not_triggered_when_not_needed(self):
        """Test _cleanup is not triggered when cleanup interval not reached."""
        cleanup_interval = 10  # 10 seconds for testing
        cache = RateLimitCache(cleanup_interval=cleanup_interval)

        # Add entries
        cache.get("client1")
        cache.get("client2")

        # Set last_cleanup to be recent
        now = time.time()
        cache.last_cleanup = now - 5  # Not old enough to trigger cleanup

        # Get another entry - should not trigger cleanup
        cache.get("client3")

        # All entries should still be there
        assert "client1" in cache.cache
        assert "client2" in cache.cache
        assert "client3" in cache.cache


class TestRateLimiter:
    """Test suite for RateLimiter middleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        app = FastAPI()
        return app

    @pytest.fixture
    def rate_limiter(self, mock_app):
        """Create a RateLimiter instance."""
        return RateLimiter(mock_app)

    def test_initialization(self, mock_app):
        """Test RateLimiter initialization."""
        rate_limiter = RateLimiter(mock_app)

        assert rate_limiter.rate_limit == DEFAULT_RATE_LIMIT
        assert rate_limiter.window_size == DEFAULT_WINDOW_SIZE
        assert rate_limiter.burst_limit == DEFAULT_BURST_LIMIT
        assert rate_limiter.excluded_paths == ["/health", "/docs", "/redoc", "/openapi.json"]
        assert rate_limiter.get_client_id == RateLimiter._default_client_id

    def test_initialization_custom_values(self, mock_app):
        """Test RateLimiter initialization with custom values."""

        def custom_get_client_id(req):
            return "custom"

        rate_limiter = RateLimiter(
            mock_app,
            rate_limit=100,
            window_size=60,
            burst_limit=50,
            excluded_paths=["/custom"],
            get_client_id=custom_get_client_id,
        )

        assert rate_limiter.rate_limit == 100
        assert rate_limiter.window_size == 60
        assert rate_limiter.burst_limit == 50
        assert rate_limiter.excluded_paths == ["/custom"]
        assert rate_limiter.get_client_id == custom_get_client_id

    def test_default_client_id_from_header(self):
        """Test default client ID extraction from API key header."""
        request = MagicMock()
        request.headers = {"X-API-Key": "test-api-key"}
        request.query_params = {}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "test-api-key"

    def test_default_client_id_from_authorization_header(self):
        """Test default client ID extraction from Authorization header."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer test-token"}
        request.query_params = {}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "Bearer test-token"

    def test_default_client_id_from_query_param(self):
        """Test default client ID extraction from query parameter."""
        request = MagicMock()
        request.headers = {}
        request.query_params = {"api_key": "test-api-key"}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "test-api-key"

    def test_default_client_id_from_ip(self):
        """Test default client ID extraction from IP address."""
        request = MagicMock()
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "192.168.1.1"

    def test_default_client_id_from_forwarded_header(self):
        """Test default client ID extraction from X-Forwarded-For header."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.query_params = {}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "192.168.1.1"

    def test_default_client_id_fallback_to_unknown(self):
        """Test default client ID falls back to 'unknown' when no identifiers found."""
        request = MagicMock()
        request.headers = {}
        request.query_params = {}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "unknown"

    def test_default_client_id_forwarded_header_multiple_ips(self):
        """Test default client ID handles multiple IPs in X-Forwarded-For."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "  10.0.0.1  , 192.168.1.1  "}
        request.query_params = {}
        request.client = None

        client_id = RateLimiter._default_client_id(request)
        assert client_id == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_excluded_path_skips_rate_limiting(self, rate_limiter):
        """Test that excluded paths skip rate limiting."""
        request = MagicMock()
        request.url.path = "/health"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_excluded_path_docs_skips_rate_limiting(self, rate_limiter):
        """Test that /docs path skips rate limiting."""
        request = MagicMock()
        request.url.path = "/docs"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, rate_limiter):
        """Test that rate limit exceeded returns 429 status."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()

        # Make many requests to exceed rate limit
        for _ in range(rate_limiter.rate_limit + 1):
            response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 429
        assert "rate_limit_exceeded" in response.body.decode()

    @pytest.mark.asyncio
    async def test_rate_limit_headers_added_to_response(self, rate_limiter):
        """Test that rate limit headers are added to successful responses."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self, rate_limiter):
        """Test that tokens refill over time."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        # Get initial entry
        client_id = rate_limiter.get_client_id(request)
        entry = rate_limiter.cache.get(client_id)

        # Consume all tokens
        for _ in range(rate_limiter.rate_limit):
            await rate_limiter.dispatch(request, call_next)

        # Verify no tokens left
        assert entry.tokens == 0

        # Simulate time passing (refill tokens AND clear sliding window)
        old_time = time.time() - rate_limiter.window_size - 1  # More than window size ago
        entry.last_refill = old_time
        entry.tokens = 0

        # Clear the sliding window by setting all request timestamps to old time
        entry.requests.clear()
        # Add one old request to simulate a request that will be cleaned up
        entry.requests.append(old_time)

        # Make another request - should refill tokens and clean old requests
        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        assert entry.tokens > 0

    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, rate_limiter):
        """Test token bucket algorithm with partial refill."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        # Get initial entry
        client_id = rate_limiter.get_client_id(request)
        entry = rate_limiter.cache.get(client_id)

        # Consume some tokens (but not all to avoid sliding window limit)
        for _ in range(50):
            await rate_limiter.dispatch(request, call_next)

        # Simulate partial time passing
        entry.last_refill = time.time() - (rate_limiter.window_size / 2)
        entry.tokens = 0

        # Make another request - should refill some tokens
        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        assert entry.tokens > 0
        assert entry.tokens < rate_limiter.rate_limit

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_by_tokens(self, rate_limiter):
        """Test rate limit exceeded when tokens are 0."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        # Get initial entry and set tokens to 0
        client_id = rate_limiter.get_client_id(request)
        entry = rate_limiter.cache.get(client_id)
        entry.tokens = 0

        # Make request - should be rate limited
        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 429
        assert "rate_limit_exceeded" in response.body.decode()

    @pytest.mark.asyncio
    async def test_rate_limit_headers_in_429_response(self, rate_limiter):
        """Test that rate limit headers are added to 429 responses."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()

        # Get initial entry and set tokens to 0
        client_id = rate_limiter.get_client_id(request)
        entry = rate_limiter.cache.get(client_id)
        entry.tokens = 0

        # Make request - should be rate limited
        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_request_window_cleanup(self, rate_limiter):
        """Test that old requests are cleaned from the window."""
        request = MagicMock()
        request.url.path = "/api/test"
        request.headers = {}
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        call_next = AsyncMock()
        call_next.return_value = Response(status_code=200)

        # Get initial entry
        client_id = rate_limiter.get_client_id(request)
        entry = rate_limiter.cache.get(client_id)

        # Add old request to window
        old_time = time.time() - rate_limiter.window_size - 1
        entry.requests.append(old_time)

        # Make a new request - should clean old request
        response = await rate_limiter.dispatch(request, call_next)

        assert response.status_code == 200
        assert old_time not in entry.requests


class TestAddRateLimiting:
    """Test suite for add_rate_limiting function."""

    def test_add_rate_limiting(self):
        """Test add_rate_limiting function."""
        app = FastAPI()

        # Should not raise any exceptions
        add_rate_limiting(app)

        # Verify middleware was added
        assert len(app.user_middleware) > 0

    def test_add_rate_limiting_custom_values(self):
        """Test add_rate_limiting with custom values."""
        app = FastAPI()

        def custom_get_client_id(req):
            return "custom"

        add_rate_limiting(
            app,
            rate_limit=100,
            window_size=60,
            burst_limit=50,
            excluded_paths=["/custom"],
            get_client_id=custom_get_client_id,
        )

        # Verify middleware was added
        assert len(app.user_middleware) > 0

    def test_add_rate_limiting_with_none_excluded_paths(self):
        """Test add_rate_limiting with None excluded_paths."""
        app = FastAPI()

        add_rate_limiting(app, excluded_paths=None)

        # Verify middleware was added
        assert len(app.user_middleware) > 0

    def test_add_rate_limiting_with_none_get_client_id(self):
        """Test add_rate_limiting with None get_client_id."""
        app = FastAPI()

        add_rate_limiting(app, get_client_id=None)

        # Verify middleware was added
        assert len(app.user_middleware) > 0

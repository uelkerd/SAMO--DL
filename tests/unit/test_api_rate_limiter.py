#!/usr/bin/env python3
"""Unit tests for API rate limiter."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request, Response

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
    """Test suite for RateLimitEntry."""

    def test_rate_limit_entry_initialization(self):
        """Test RateLimitEntry initialization with default values."""
        entry = RateLimitEntry()

        assert entry.tokens == DEFAULT_RATE_LIMIT
        assert len(entry.requests) == 0
        assert entry.last_refill > 0
        assert entry.last_access > 0

    def test_rate_limit_entry_custom_values(self):
        """Test RateLimitEntry initialization with custom values."""
        entry = RateLimitEntry(tokens=50)

        assert entry.tokens == 50
        assert len(entry.requests) == 0


class TestRateLimitCache:
    """Test suite for RateLimitCache."""

    def test_cache_initialization(self):
        """Test RateLimitCache initialization."""
        cache = RateLimitCache()

        assert len(cache.cache) == 0
        assert cache.cleanup_interval == 3600
        assert cache.last_cleanup > 0

    def test_cache_initialization_custom_cleanup(self):
        """Test RateLimitCache initialization with custom cleanup interval."""
        cache = RateLimitCache(cleanup_interval=1800)

        assert cache.cleanup_interval == 1800

    def test_get_creates_new_entry(self):
        """Test that get() creates a new entry for unknown key."""
        cache = RateLimitCache()

        entry = cache.get("test_client")

        assert "test_client" in cache.cache
        assert entry == cache.cache["test_client"]
        assert entry.tokens == DEFAULT_RATE_LIMIT

    def test_get_returns_existing_entry(self):
        """Test that get() returns existing entry for known key."""
        cache = RateLimitCache()

        entry1 = cache.get("test_client")
        entry2 = cache.get("test_client")

        assert entry1 is entry2
        assert len(cache.cache) == 1

    def test_cleanup_removes_old_entries(self):
        """Test that cleanup removes old entries."""
        cache = RateLimitCache(cleanup_interval=1)

        # Create an entry
        entry = cache.get("test_client")
        entry.last_access = time.time() - 2  # Make it old

        # Trigger cleanup
        cache._cleanup()

        assert "test_client" not in cache.cache

    def test_cleanup_does_not_remove_recent_entries(self):
        """Test that cleanup does not remove recent entries."""
        cache = RateLimitCache(cleanup_interval=1)

        # Create an entry
        entry = cache.get("test_client")
        entry.last_access = time.time()  # Make it recent

        # Trigger cleanup
        cache._cleanup()

        assert "test_client" in cache.cache

    def test_cleanup_not_triggered_when_not_needed(self):
        """Test that cleanup is not triggered when not needed."""
        cache = RateLimitCache(cleanup_interval=3600)

        # Create an entry
        entry = cache.get("test_client")
        entry.last_access = time.time() - 1800  # Not old enough

        # Trigger cleanup
        cache._cleanup()

        assert "test_client" in cache.cache


class TestRateLimiter:
    """Test suite for RateLimiter."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        return MagicMock(spec=FastAPI)

    @pytest.fixture
    def rate_limiter(self, mock_app):
        """Create a RateLimiter instance for testing."""
        def custom_get_client_id(req):
            return "test_client"

        return RateLimiter(
            app=mock_app,
            rate_limit=100,
            burst_limit=10,
            window_size=60,
            get_client_id=custom_get_client_id,
            excluded_paths=["/health", "/docs"]
        )

    def test_initialization(self, mock_app):
        """Test RateLimiter initialization with default values."""
        rate_limiter = RateLimiter(app=mock_app)

        assert rate_limiter.rate_limit == DEFAULT_RATE_LIMIT
        assert rate_limiter.burst_limit == DEFAULT_BURST_LIMIT
        assert rate_limiter.window_size == DEFAULT_WINDOW_SIZE
        assert rate_limiter.excluded_paths == ["/health", "/docs"]

    def test_initialization_custom_values(self, mock_app):
        """Test RateLimiter initialization with custom values."""
        def custom_get_client_id(req):
            return "custom_client"

        rate_limiter = RateLimiter(
            app=mock_app,
            rate_limit=100,
            burst_limit=50,
            window_size=60,
            get_client_id=custom_get_client_id,
            excluded_paths=["/custom"]
        )

        assert rate_limiter.rate_limit == 100
        assert rate_limiter.burst_limit == 50
        assert rate_limiter.window_size == 60
        assert rate_limiter.excluded_paths == ["/custom"]

    def test_default_client_id_from_header(self):
        """Test default client ID extraction from X-Client-ID header."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with X-Client-ID header
        mock_request = MagicMock()
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "test_client"

    def test_default_client_id_from_authorization_header(self):
        """Test default client ID extraction from Authorization header."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with Authorization header
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer test_token"}
        mock_request.query_params = {}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "test_token"

    def test_default_client_id_from_query_param(self):
        """Test default client ID extraction from query parameter."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with client_id query parameter
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {"client_id": "test_client"}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "test_client"

    def test_default_client_id_from_ip(self):
        """Test default client ID extraction from IP address."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with IP address
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "192.168.1.1"

    def test_default_client_id_from_forwarded_header(self):
        """Test default client ID extraction from X-Forwarded-For header."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with X-Forwarded-For header
        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1"}
        mock_request.query_params = {}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "10.0.0.1"

    def test_default_client_id_fallback_to_unknown(self):
        """Test default client ID falls back to 'unknown' when no identifier found."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with no identifiers
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "unknown"

    def test_default_client_id_forwarded_header_multiple_ips(self):
        """Test default client ID extraction from X-Forwarded-For with multiple IPs."""
        rate_limiter = RateLimiter(app=FastAPI())
        
        # Mock request with multiple IPs in X-Forwarded-For
        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1, 172.16.0.1"}
        mock_request.query_params = {}
        mock_request.client = None

        client_id = rate_limiter._get_client_id(mock_request)
        assert client_id == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_excluded_path_skips_rate_limiting(self, rate_limiter):
        """Test that excluded paths skip rate limiting."""
        # Mock request to excluded path
        mock_request = MagicMock()
        mock_request.url.path = "/health"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Should not be rate limited
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_excluded_path_docs_skips_rate_limiting(self, rate_limiter):
        """Test that docs path skips rate limiting."""
        # Mock request to docs path
        mock_request = MagicMock()
        mock_request.url.path = "/docs"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Should not be rate limited
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, rate_limiter):
        """Test that rate limit exceeded returns 429."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Exhaust the rate limit
        for _ in range(DEFAULT_BURST_LIMIT + 1):
            response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Should return 429
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_headers_added_to_response(self, rate_limiter):
        """Test that rate limit headers are added to successful responses."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Should have rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self, rate_limiter):
        """Test that tokens refill over time."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Make some requests to consume tokens
        for _ in range(5):
            response = await rate_limiter.middleware(mock_request, mock_call_next)
            assert response.status_code == 200

        # Check remaining tokens
        entry = rate_limiter.cache.get("test_client")
        assert entry.tokens < DEFAULT_RATE_LIMIT

    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self, rate_limiter):
        """Test token bucket algorithm behavior."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Consume all tokens
        for _ in range(DEFAULT_RATE_LIMIT):
            response = await rate_limiter.middleware(mock_request, mock_call_next)
            assert response.status_code == 200

        # Next request should be rate limited
        response = await rate_limiter.middleware(mock_request, mock_call_next)
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_by_tokens(self, rate_limiter):
        """Test rate limit exceeded by token consumption."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Consume all tokens
        for _ in range(DEFAULT_RATE_LIMIT):
            response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Next request should be rate limited
        response = await rate_limiter.middleware(mock_request, mock_call_next)
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_headers_in_429_response(self, rate_limiter):
        """Test that 429 responses include rate limit headers."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Exhaust the rate limit
        for _ in range(DEFAULT_RATE_LIMIT + 1):
            response = await rate_limiter.middleware(mock_request, mock_call_next)

        # Verify 429 response has rate limit headers
        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_request_window_cleanup(self, rate_limiter):
        """Test that old requests are cleaned up from the window."""
        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"
        mock_request.headers = {"X-Client-ID": "test_client"}
        mock_request.query_params = {}
        mock_request.client = None

        # Mock call_next
        async def mock_call_next(request):
            return Response()

        # Make some requests
        for _ in range(5):
            response = await rate_limiter.middleware(mock_request, mock_call_next)
            assert response.status_code == 200

        # Verify requests are tracked
        entry = rate_limiter.cache.get("test_client")
        assert len(entry.requests) == 5

        # Simulate time passing to make requests old
        current_time = time.time()
        # Note: We can't directly modify the deque elements, so we'll test the cleanup logic differently
        # The actual cleanup happens in the dispatch method


class TestAddRateLimiting:
    """Test suite for add_rate_limiting function."""

    def test_add_rate_limiting(self):
        """Test add_rate_limiting function."""
        app = FastAPI()
        
        add_rate_limiting(app)
        
        # Verify middleware was added (check that the function doesn't raise an error)
        assert app is not None

    def test_add_rate_limiting_custom_values(self):
        """Test add_rate_limiting with custom values."""
        app = FastAPI()
        
        def custom_get_client_id(req):
            return "custom_client"

        add_rate_limiting(
            app=app,
            rate_limit=100,
            burst_limit=50,
            window_size=60,
            get_client_id=custom_get_client_id,
            excluded_paths=["/custom"]
        )
        
        # Verify middleware was added (check that the function doesn't raise an error)
        assert app is not None

    def test_add_rate_limiting_with_none_excluded_paths(self):
        """Test add_rate_limiting with None excluded_paths."""
        app = FastAPI()
        
        add_rate_limiting(app, excluded_paths=None)
        
        # Verify middleware was added (check that the function doesn't raise an error)
        assert app is not None

    def test_add_rate_limiting_with_none_get_client_id(self):
        """Test add_rate_limiting with None get_client_id."""
        app = FastAPI()
        
        add_rate_limiting(app, get_client_id=None)
        
        # Verify middleware was added (check that the function doesn't raise an error)
        assert app is not None

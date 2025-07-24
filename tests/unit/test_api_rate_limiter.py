"""
Unit tests for API rate limiter.
Tests the rate limiting middleware, token bucket algorithm, and cache.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, Response
from starlette.types import Scope

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


class TestRateLimiter:
    """Test suite for RateLimiter middleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def rate_limiter(self, mock_app):
        """Create a RateLimiter instance with a mock app."""
        return RateLimiter(
            app=mock_app,
            rate_limit=5,  # Small value for testing
            window_size=60,
            burst_limit=2,
        )

    def test_initialization(self, mock_app):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(mock_app)
        
        assert limiter.app is mock_app
        assert limiter.rate_limit == DEFAULT_RATE_LIMIT
        assert limiter.window_size == DEFAULT_WINDOW_SIZE
        assert limiter.burst_limit == DEFAULT_BURST_LIMIT
        assert isinstance(limiter.cache, RateLimitCache)
        assert callable(limiter.get_client_id)

    def test_default_client_id_from_header(self):
        """Test default client ID extraction from header."""
        limiter = RateLimiter(MagicMock())
        
        # Create a mock request with an API key header
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "test-api-key"}
        mock_request.query_params = {}
        
        client_id = limiter._default_client_id(mock_request)
        
        assert client_id == "test-api-key"

    def test_default_client_id_from_query_param(self):
        """Test default client ID extraction from query parameter."""
        limiter = RateLimiter(MagicMock())
        
        # Create a mock request with an API key query parameter
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {"api_key": "test-api-key"}
        
        client_id = limiter._default_client_id(mock_request)
        
        assert client_id == "test-api-key"

    def test_default_client_id_from_ip(self):
        """Test default client ID extraction from IP address."""
        limiter = RateLimiter(MagicMock())
        
        # Create a mock request with client IP
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        
        client_id = limiter._default_client_id(mock_request)
        
        assert client_id == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_excluded_path_skips_rate_limiting(self, rate_limiter):
        """Test that excluded paths skip rate limiting."""
        # Create a mock request with an excluded path
        mock_request = MagicMock()
        mock_request.url.path = "/health"
        
        # Create a mock response
        mock_response = Response(content="OK")
        
        # Set up the call_next mock to return the mock response
        call_next = AsyncMock(return_value=mock_response)
        
        # Call dispatch
        response = await rate_limiter.dispatch(mock_request, call_next)
        
        # Verify that call_next was called with the request
        call_next.assert_called_once_with(mock_request)
        
        # Verify that the response is the mock response
        assert response is mock_response
        
        # Verify that no rate limit headers were added
        assert "X-RateLimit-Limit" not in response.headers


class TestAddRateLimiting:
    """Test suite for add_rate_limiting function."""

    def test_add_rate_limiting(self):
        """Test add_rate_limiting adds middleware to FastAPI app."""
        app = FastAPI()
        
        # Mock the add_middleware method
        app.add_middleware = MagicMock()
        
        # Call add_rate_limiting
        add_rate_limiting(app, rate_limit=10, window_size=30)
        
        # Verify add_middleware was called with RateLimiter
        app.add_middleware.assert_called_once_with(
            RateLimiter,
            rate_limit=10,
            window_size=30,
            burst_limit=DEFAULT_BURST_LIMIT,
            excluded_paths=None,
            get_client_id=None,
        ) 
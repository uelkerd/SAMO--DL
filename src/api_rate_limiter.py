"""API Rate Limiter for SAMO Deep Learning.

This module implements rate limiting for the SAMO API endpoints using FastAPI middleware.
It uses a token bucket algorithm to limit the rate of requests per user based on API keys
or IP addresses when no authentication is provided.

Key Features:
- Per-user rate limiting (100 requests/minute default)
- Token bucket algorithm for request throttling
- Configuration options for different endpoints
- Automatic rate limit header inclusion in responses
- Cache-based storage with automatic cleanup
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Default rate limit constants
DEFAULT_RATE_LIMIT = 100  # 100 requests per minute
DEFAULT_WINDOW_SIZE = 60  # 1 minute (in seconds)
DEFAULT_BURST_LIMIT = 10  # Allow 10 requests at once
DEFAULT_CACHE_CLEANUP_INTERVAL = 3600  # 1 hour (in seconds)


@dataclass
class RateLimitEntry:
    """Rate limit tracking for a single client."""

    # Sliding window of request timestamps
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Sliding window for token bucket
    tokens: int = DEFAULT_RATE_LIMIT

    # Last token refill time
    last_refill: float = field(default_factory=time.time)

    # Last time this entry was used (for cleanup)
    last_access: float = field(default_factory=time.time)


class RateLimitCache:
    """In-memory cache for rate limiting data."""

    def __init__(self, cleanup_interval: int = DEFAULT_CACHE_CLEANUP_INTERVAL):
        """Initialize rate limit cache.

        Args:
            cleanup_interval: Seconds between cache cleanup runs
        """
        self.cache: dict[str, RateLimitEntry] = {}
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()

    def get(self, key: str) -> RateLimitEntry:
        """Get rate limit entry for a client, creating if needed.

        Args:
            key: Client identifier (API key or IP address)

        Returns:
            Rate limit entry
        """
        if key not in self.cache:
            self.cache[key] = RateLimitEntry()

        # Update last access time
        self.cache[key].last_access = time.time()

        # Periodic cleanup of old entries
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup()

        return self.cache[key]

    def _cleanup(self):
        """Remove old entries from cache."""
        now = time.time()
        expired_keys = []

        # Find keys that haven't been accessed in a long time
        for key, entry in self.cache.items():
            if now - entry.last_access > self.cleanup_interval:
                expired_keys.append(key)

        # Remove expired keys
        for key in expired_keys:
            del self.cache[key]

        self.last_cleanup = now


class RateLimiter(BaseHTTPMiddleware):
    """Middleware for API rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        window_size: int = DEFAULT_WINDOW_SIZE,
        burst_limit: int = DEFAULT_BURST_LIMIT,
        excluded_paths: Optional[list[str]] = None,
        get_client_id: Optional[Callable[[Request], str]] = None,
    ):
        """Initialize rate limiter middleware.

        Args:
            app: FastAPI application
            rate_limit: Maximum number of requests allowed per window
            window_size: Time window in seconds
            burst_limit: Maximum requests to process at once
            excluded_paths: URL paths to exclude from rate limiting
            get_client_id: Function to extract client ID from request
        """
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window_size = window_size
        self.burst_limit = burst_limit
        self.excluded_paths = excluded_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
        self.cache = RateLimitCache()

        # Function to extract client ID (API key or IP)
        self.get_client_id = get_client_id or self._default_client_id

    @staticmethod
    def _default_client_id(request: Request) -> str:
        """Default method to extract client ID from request.

        Tries API key from header or query param first, falls back to IP address.

        Args:
            request: FastAPI request

        Returns:
            Client identifier string
        """
        # Try API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            return api_key

        # Try API key from query param
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key

        # Fallback to client IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Direct client IP
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Skip rate limiting for excluded paths
        for path in self.excluded_paths:
            if request.url.path.startswith(path):
                return await call_next(request)

        # Get client identifier
        client_id = self.get_client_id(request)

        # Get rate limit entry for this client
        entry = self.cache.get(client_id)

        # Refill tokens based on time elapsed
        now = time.time()
        time_passed = now - entry.last_refill

        # Token refill calculation (tokens accumulate over time)
        new_tokens = int(time_passed * (self.rate_limit / self.window_size))
        entry.tokens = min(self.rate_limit, entry.tokens + new_tokens)
        entry.last_refill = now

        # Add request timestamp to window
        entry.requests.append(now)

        # Clean old requests outside window
        while entry.requests and entry.requests[0] < now - self.window_size:
            entry.requests.popleft()

        # Check if rate limit exceeded
        requests_in_window = len(entry.requests)

        # Add rate limit headers
        response = None

        if requests_in_window > self.rate_limit or entry.tokens <= 0:
            # Rate limit exceeded
            headers = {
                "X-RateLimit-Limit": str(self.rate_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(entry.last_refill + self.window_size)),
                "Retry-After": str(self.window_size),
            }

            content = {
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. {self.rate_limit} requests allowed per minute.",
                "rate_limit": self.rate_limit,
                "window_size_seconds": self.window_size,
                "retry_after_seconds": self.window_size,
            }

            response = Response(
                status_code=429,
                content=str(content),
                headers=headers,
                media_type="application/json",
            )
        else:
            # Consume one token
            entry.tokens -= 1

            # Process request
            response = await call_next(request)

            # Add rate limit headers to normal responses
            response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(self.rate_limit - requests_in_window)
            response.headers["X-RateLimit-Reset"] = str(int(now + self.window_size))

        return response


def add_rate_limiting(
    app: FastAPI,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    window_size: int = DEFAULT_WINDOW_SIZE,
    burst_limit: int = DEFAULT_BURST_LIMIT,
    excluded_paths: Optional[list[str]] = None,
    get_client_id: Optional[Callable[[Request], str]] = None,
) -> None:
    """Add rate limiting middleware to a FastAPI application.

    Args:
        app: FastAPI application
        rate_limit: Maximum number of requests allowed per window
        window_size: Time window in seconds
        burst_limit: Maximum requests to process at once
        excluded_paths: URL paths to exclude from rate limiting
        get_client_id: Function to extract client ID from request
    """
    app.add_middleware(
        RateLimiter,
        rate_limit=rate_limit,
        window_size=window_size,
        burst_limit=burst_limit,
        excluded_paths=excluded_paths,
        get_client_id=get_client_id,
    )

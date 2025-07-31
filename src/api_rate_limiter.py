#!/usr/bin/env python3
"""
API Rate Limiter for SAMO Deep Learning.

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

    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    tokens: int = DEFAULT_RATE_LIMIT
    last_refill: float = field(default_factory=time.time)
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

        self.cache[key].last_access = time.time()

        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup()

        return self.cache[key]

    def _cleanup(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.last_access > self.cleanup_interval
        ]
        for key in expired_keys:
            del self.cache[key]
        self.last_cleanup = current_time


class RateLimiter(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        window_size: int = DEFAULT_WINDOW_SIZE,
        burst_limit: int = DEFAULT_BURST_LIMIT,
        excluded_paths: Optional[list[str]] = None,
        get_client_id: Optional[Callable[[Request], str]] = None,
    ):
        """Initialize rate limiter.

        Args:
            app: FastAPI application
            rate_limit: Maximum requests per window
            window_size: Time window in seconds
            burst_limit: Maximum burst requests
            excluded_paths: Paths to exclude from rate limiting
            get_client_id: Function to extract client ID from request
        """
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window_size = window_size
        self.burst_limit = burst_limit
        self.excluded_paths = excluded_paths or ["/health", "/docs"]
        self.get_client_id = get_client_id or self._default_client_id
        self.cache = RateLimitCache()

    @staticmethod
    def _default_client_id(request: Request) -> str:
        """Extract client ID from request using multiple strategies."""
        # Try API key from header
        if "X-Client-ID" in request.headers:
            return request.headers["X-Client-ID"]
        
        # Try API key from Authorization header
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                return auth_header[7:]  # Remove "Bearer " prefix
        
        # Try API key from query param
        if "client_id" in request.query_params:
            return request.query_params["client_id"]
        
        # Try X-Forwarded-For header
        if "X-Forwarded-For" in request.headers:
            forwarded_for = request.headers["X-Forwarded-For"]
            # Take the first IP if multiple are present
            return forwarded_for.split(",")[0].strip()
        
        # Fallback to client IP address
        if request.client:
            return request.client.host
        
        # Final fallback
        return "unknown"

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier."""
        return self.get_client_id(request)

    async def middleware(self, request: Request, call_next):
        """Middleware function for rate limiting."""
        return await self.dispatch(request, call_next)

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit entry for this client
        entry = self.cache.get(client_id)
        
        # Clean old requests outside window
        current_time = time.time()
        while entry.requests and current_time - entry.requests[0] > self.window_size:
            entry.requests.popleft()
        
        # Refill tokens based on time elapsed
        time_since_refill = current_time - entry.last_refill
        tokens_to_add = int(time_since_refill * (self.rate_limit / self.window_size))
        entry.tokens = min(self.rate_limit, entry.tokens + tokens_to_add)
        entry.last_refill = current_time
        
        # Check if we have tokens available
        if entry.tokens <= 0:
            # Rate limit exceeded - no tokens available
            response = Response(
                content="Rate limit exceeded",
                status_code=429,
                media_type="text/plain"
            )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))
            response.headers["Retry-After"] = str(self.window_size)
            
            return response
        
        # Consume one token
        entry.tokens -= 1
        
        # Add request timestamp to window
        entry.requests.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to normal responses
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(entry.tokens)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))
        
        return response


def add_rate_limiting(
    app: FastAPI,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    window_size: int = DEFAULT_WINDOW_SIZE,
    burst_limit: int = DEFAULT_BURST_LIMIT,
    excluded_paths: Optional[list[str]] = None,
    get_client_id: Optional[Callable[[Request], str]] = None,
) -> None:
    """Add rate limiting middleware to FastAPI app.

    Args:
        app: FastAPI application
        rate_limit: Maximum requests per window
        window_size: Time window in seconds
        burst_limit: Maximum burst requests
        excluded_paths: Paths to exclude from rate limiting
        get_client_id: Function to extract client ID from request
    """
    middleware = RateLimiter(
        app=app,
        rate_limit=rate_limit,
        window_size=window_size,
        burst_limit=burst_limit,
        excluded_paths=excluded_paths,
        get_client_id=get_client_id,
    )
    
    # Add middleware to app
    app.add_middleware(RateLimiter, 
                      rate_limit=rate_limit,
                      window_size=window_size,
                      burst_limit=burst_limit,
                      excluded_paths=excluded_paths,
                      get_client_id=get_client_id)

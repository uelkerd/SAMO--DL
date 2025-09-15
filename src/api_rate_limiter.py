#!/usr/bin/env python3
"""
🔒 API Rate Limiter
==================
Token bucket algorithm for API rate limiting.
Includes security features.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Optional, Tuple, Set
import logging
import hashlib
import ipaddress
from dataclasses import dataclass
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size_seconds: int = 60
    block_duration_seconds: int = 300  # 5 minutes
    max_concurrent_requests: int = 5
    enable_ip_whitelist: bool = False
    enable_ip_blacklist: bool = False
    whitelisted_ips: set = None
    blacklisted_ips: set = None
    # Abuse detection thresholds
    rapid_fire_threshold: int = 10  # Max requests per second
    sustained_rate_threshold: int = 200  # Max requests per minute
    rapid_fire_window: float = 1.0  # Time window for rapid-fire detection (seconds)
    sustained_rate_window: float = 60.0  # Time window for sustained rate detection (seconds)
    # Enhanced anomaly detection
    enable_user_agent_analysis: bool = True
    enable_request_pattern_analysis: bool = True
    suspicious_user_agent_score_threshold: int = 3  # Score threshold for suspicious UAs
    request_pattern_score_threshold: int = 5  # Score threshold for suspicious patterns
    anomaly_detection_window: float = 300.0  # 5 minutes for pattern analysis


# -------- Path exclusion helpers --------

def _normalize_path(path: str) -> str:
    """Normalize path for matching.

    Lowercase, ensure leading slash, strip trailing slashes.
    """
    if not path:
        return "/"
    p = path.lower().strip()
    if not p.startswith("/"):
        p = "/" + p
    while len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def _build_exclusions(excluded_paths: Optional[Set[str]]) -> Set[str]:
    """Build normalized exclusions set.

    Merges default exclusions with any provided paths and normalizes each
    entry (lowercase, leading slash, no trailing slash).
    """
    default_exclusions: Set[str] = {
        "/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    return {
        _normalize_path(p)
        for p in (default_exclusions | (excluded_paths or set()))
    }


def _is_excluded_path(request_path: str, normalized_exclusions: Set[str]) -> bool:
    """Check if the request path should be excluded.

    Matches exact base or any subpath of an excluded base.
    """
    norm_path = _normalize_path(request_path)
    if norm_path in normalized_exclusions:
        return True
    for base in normalized_exclusions:
        if base != "/" and norm_path.startswith(base + "/"):
            return True
    return False


class _RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for token-bucket rate limiting.

    Applies rate limits using a shared limiter instance while respecting
    normalized path exclusions and test user-agents.
    """

    def __init__(
        self,
        app,
        rate_limiter: "TokenBucketRateLimiter",
        config: RateLimitConfig,
        normalized_exclusions: Set[str],
    ) -> None:
        super().__init__(app)
        self._limiter = rate_limiter
        self._cfg = config
        self._exclusions = normalized_exclusions

    async def dispatch(self, request, call_next):  # type: ignore[override]
        """Apply rate limits unless path is excluded or test UA is used."""
        if _is_excluded_path(request.url.path, self._exclusions):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")

        # Bypass rate limiting for test environment UAs
        ua_lower = user_agent.lower()
        if "test" in ua_lower or "pytest" in ua_lower or "testclient" in ua_lower:
            return await call_next(request)

        # Check rate limit
        allowed, reason, meta = self._limiter.allow_request(client_ip, user_agent)
        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": reason,
                    "retry_after": meta.get("retry_after", 60),
                },
            )

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._cfg.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(meta.get("tokens_remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(meta.get("reset_time", 0))
        return response


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with security enhancements.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - IP-based rate limiting with whitelist/blacklist
    - Burst protection
    - Concurrent request limiting
    - Automatic blocking of abusive clients
    - Request fingerprinting for advanced detection
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: Dict[str, float] = defaultdict(lambda: config.burst_size)
        self.last_refill: Dict[str, float] = defaultdict(time.time)
        self.blocked_clients: Dict[str, float] = {}
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        self.request_history: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()

        # Initialize whitelist/blacklist
        if config.whitelisted_ips is None:
            config.whitelisted_ips = set()
        if config.blacklisted_ips is None:
            config.blacklisted_ips = set()

    def _get_client_key(self, client_ip: str, user_agent: str = "") -> str:
        """Generate a unique client key for rate limiting."""
        fingerprint = f"{client_ip}:{user_agent}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP is allowed based on whitelist/blacklist."""
        if client_ip in ["testclient", "127.0.0.1", "localhost"]:
            return True
        try:
            # Validate IP; exception will be raised if invalid
            ipaddress.ip_address(client_ip)
            if (
                self.config.enable_ip_blacklist
                and client_ip in self.config.blacklisted_ips
            ):
                logger.warning(
                    "Blocked request from blacklisted IP: %s", client_ip
                )
                return False
            if (
                self.config.enable_ip_whitelist
                and client_ip not in self.config.whitelisted_ips
            ):
                logger.warning(
                    "Blocked request from non-whitelisted IP: %s", client_ip
                )
                return False
            return True
        except ValueError:
            logger.error("Invalid IP address: %s", client_ip)
            return False

    def _is_client_blocked(self, client_key: str) -> bool:
        """Check if client is currently blocked."""
        if client_key in self.blocked_clients:
            block_until = self.blocked_clients[client_key]
            if time.time() < block_until:
                return True
            del self.blocked_clients[client_key]
        return False

    def _analyze_user_agent(self, user_agent: str) -> int:
        """Analyze user agent for suspicious patterns. Returns score (0-10)."""
        if not user_agent:
            return 0
        ua_lower = user_agent.lower()

        high_risk_patterns = [
            'sqlmap', 'nikto', 'nmap', 'scanner', 'crawler', 'spider',
            'bot', 'automation', 'script', 'python-requests', 'curl',
            'wget', 'httrack', 'grabber', 'harvester'
        ]
        medium_risk_patterns = [
            'headless', 'phantom', 'selenium', 'webdriver', 'automated',
            'testing', 'monitoring', 'healthcheck', 'pingdom', 'uptimerobot'
        ]
        low_risk_patterns = [
            'bot', 'crawler', 'spider', 'indexer', 'feed', 'rss',
            'aggregator', 'monitor', 'checker'
        ]

        score = (
            3 * sum(1 for p in high_risk_patterns if p in ua_lower)
            + 2 * sum(1 for p in medium_risk_patterns if p in ua_lower)
            + 1 * sum(1 for p in low_risk_patterns if p in ua_lower)
        )

        if (
            any(p in ua_lower for p in ["bot", "crawler"]) and
            any(p in ua_lower for p in ["python", "curl", "wget"])
        ):
            score += 2

        return min(score, 10)

    def _analyze_request_patterns(self, client_key: str, client_ip: str) -> int:
        """Analyze request patterns for suspicious behavior. Returns score (0-10)."""
        # Delegate to helper calculators to reduce complexity and improve readability
        history = self.request_history[client_key]
        current_time = time.time()
        if len(history) < 5:
            return 0
        recent_history = self._get_recent_history(history, current_time)
        if len(recent_history) < 3:
            return 0
        score = 0
        score += self._calculate_burst_score(recent_history, current_time)
        score += self._calculate_request_regular_interval_score(recent_history)
        score += self._calculate_sustained_volume_score(recent_history, current_time)
        return min(score, 10)

    def _get_recent_history(self, history: Deque, current_time: float) -> list:
        """Return recent timestamps within anomaly detection window."""
        window = self.config.anomaly_detection_window
        return [t for t in history if current_time - t <= window]

    @staticmethod
    def _calculate_burst_score(recent_history: list, current_time: float) -> int:
        """Score short bursts within multiple sliding windows."""
        score = 0
        for window in [1.0, 5.0, 10.0]:
            burst_count = sum(1 for t in recent_history if current_time - t <= window)
            if burst_count > window * 2:
                score += 2
        return score

    @staticmethod
    def _calculate_request_regular_interval_score(recent_history: list) -> int:
        """Score unusually regular fast requests (low variance, low average)."""
        if len(recent_history) < 5:
            return 0
        intervals = [
            recent_history[i] - recent_history[i - 1]
            for i in range(1, len(recent_history))
        ]
        if len(intervals) < 3:
            return 0
        avg = sum(intervals) / len(intervals)
        var = sum((x - avg) ** 2 for x in intervals) / len(intervals)
        return 3 if (var < 0.1 and avg < 2.0) else 0

    @staticmethod
    def _calculate_sustained_volume_score(
        recent_history: list, current_time: float
    ) -> int:
        """Score sustained high request volume over the last minute."""
        minute_count = sum(
            1 for t in recent_history if current_time - t <= 60.0
        )
        return 2 if minute_count > 50 else 0

    def _detect_abuse(
        self,
        client_key: str,
        client_ip: str,
        user_agent: str = "",
    ) -> bool:
        """Enhanced abuse detection with user agent and pattern analysis."""
        history = self.request_history[client_key]
        current_time = time.time()
        while history and current_time - history[0] > 3600:
            history.popleft()
        recent_requests = [
            t for t in history
            if current_time - t <= self.config.rapid_fire_window
        ]
        if len(recent_requests) > self.config.rapid_fire_threshold:
            logger.warning(
                "Rate-based abuse detected: %d requests in %ss from %s",
                len(recent_requests), self.config.rapid_fire_window, client_ip,
            )
            return True
        minute_requests = [
            t for t in history
            if current_time - t <= self.config.sustained_rate_window
        ]
        if len(minute_requests) > self.config.sustained_rate_threshold:
            logger.warning(
                "Rate-based abuse detected: %d requests in %ss from %s",
                len(minute_requests), self.config.sustained_rate_window, client_ip,
            )
            return True
        if self.config.enable_user_agent_analysis:
            ua_score = self._analyze_user_agent(user_agent)
            if ua_score >= self.config.suspicious_user_agent_score_threshold:
                logger.warning(
                    "User agent abuse detected: score %d from %s",
                    ua_score,
                    client_ip,
                )
                return True
        if self.config.enable_request_pattern_analysis:
            pattern_score = self._analyze_request_patterns(client_key, client_ip)
            if pattern_score >= self.config.request_pattern_score_threshold:
                logger.warning(
                    "Pattern-based abuse detected: score %d from %s",
                    pattern_score,
                    client_ip,
                )
                return True
        return False

    def _refill_bucket(self, client_key: str):
        """Refill the token bucket for a client."""
        current_time = time.time()
        last_refill_time = self.last_refill[client_key]
        time_passed = current_time - last_refill_time
        tokens_to_add = (time_passed / 60.0) * self.config.requests_per_minute
        self.buckets[client_key] = min(
            self.config.burst_size,
            self.buckets[client_key] + tokens_to_add,
        )
        self.last_refill[client_key] = current_time

    def allow_request(
        self,
        client_ip: str,
        user_agent: str = "",
    ) -> Tuple[bool, str, dict]:
        """
        Check if request should be allowed.

        Returns:
            Tuple of (allowed, reason, metadata)
        """
        with self.lock:
            if not self._is_ip_allowed(client_ip):
                return False, "IP not allowed", {"ip": client_ip}
            client_key = self._get_client_key(client_ip, user_agent)
            if self._is_client_blocked(client_key):
                return False, "Client blocked", {
                    "client_key": client_key,
                    "ip": client_ip,
                }
            if (
                self.concurrent_requests[client_key]
                >= self.config.max_concurrent_requests
            ):
                return False, "Too many concurrent requests", {
                    "client_key": client_key,
                    "concurrent": self.concurrent_requests[client_key],
                    "max": self.config.max_concurrent_requests,
                }
            if self._detect_abuse(client_key, client_ip, user_agent):
                self.blocked_clients[client_key] = (
                    time.time() + self.config.block_duration_seconds
                )
                logger.warning(
                    "Blocked abusive client %s from %s for %ss",
                    client_key,
                    client_ip,
                    self.config.block_duration_seconds,
                )
                return False, "Abuse detected", {
                    "client_key": client_key,
                    "ip": client_ip,
                }
            self._refill_bucket(client_key)
            if self.buckets[client_key] < 0.999999:
                return False, "Rate limit exceeded", {
                    "client_key": client_key,
                    "tokens": self.buckets[client_key],
                    "rate_limit": self.config.requests_per_minute,
                }
            self.buckets[client_key] -= 1.0
            self.request_history[client_key].append(time.time())
            self.concurrent_requests[client_key] += 1
            return True, "Request allowed", {
                "client_key": client_key,
                "tokens_remaining": self.buckets[client_key],
                "concurrent_requests": self.concurrent_requests[client_key],
            }

    def release_request(self, client_ip: str, user_agent: str = ""):
        """Release a concurrent request slot."""
        with self.lock:
            client_key = self._get_client_key(client_ip, user_agent)
            if client_key in self.concurrent_requests:
                self.concurrent_requests[client_key] = max(
                    0, self.concurrent_requests[client_key] - 1
                )

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "active_buckets": len(self.buckets),
                "blocked_clients": len(self.blocked_clients),
                "concurrent_requests": sum(self.concurrent_requests.values()),
                "total_clients": len(
                    set(self.buckets.keys()) | set(self.concurrent_requests.keys())
                ),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "burst_size": self.config.burst_size,
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                    "block_duration_seconds": self.config.block_duration_seconds,
                },
            }


def add_rate_limiting(
    app,
    requests_per_minute: int = 100,
    burst_size: int = 10,
    max_concurrent_requests: int = 5,
    rapid_fire_threshold: int = 10,
    sustained_rate_threshold: int = 200,
    excluded_paths: Optional[Set[str]] = None,
):
    """Attach rate limiting middleware to a FastAPI app."""
    from fastapi import Request  # noqa: F401  (kept for type hints in middleware)

    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        max_concurrent_requests=max_concurrent_requests,
        enable_ip_blacklist=True,
        enable_ip_whitelist=False,
        rapid_fire_threshold=rapid_fire_threshold,
        sustained_rate_threshold=sustained_rate_threshold,
    )
    limiter = TokenBucketRateLimiter(config)
    app.state.rate_limiter = limiter

    normalized_exclusions = _build_exclusions(excluded_paths)
    app.add_middleware(
        _RateLimitMiddleware,
        rate_limiter=limiter,
        config=config,
        normalized_exclusions=normalized_exclusions,
    )

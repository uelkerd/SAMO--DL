#!/usr/bin/env python3
"""
🔒 API Rate Limiter
==================
Token bucket algorithm implementation for API rate limiting with security features.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Optional, Tuple
import logging
import hashlib
import ipaddress
from dataclasses import dataclass
from datetime import datetime, timedelta

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
        # Create a fingerprint based on IP and user agent
        fingerprint = f"{client_ip}:{user_agent}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()
    
    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP is allowed based on whitelist/blacklist."""
        try:
            ip = ipaddress.ip_address(client_ip)
            
            # Check blacklist first
            if self.config.enable_ip_blacklist:
                if client_ip in self.config.blacklisted_ips:
                    logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
                    return False
            
            # Check whitelist
            if self.config.enable_ip_whitelist:
                if client_ip in self.config.whitelisted_ips:
                    return True
                else:
                    logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
                    return False
            
            return True
            
        except ValueError:
            logger.error(f"Invalid IP address: {client_ip}")
            return False
    
    def _is_client_blocked(self, client_key: str) -> bool:
        """Check if client is currently blocked."""
        if client_key in self.blocked_clients:
            block_until = self.blocked_clients[client_key]
            if time.time() < block_until:
                return True
            else:
                # Remove expired block
                del self.blocked_clients[client_key]
        return False
    
    def _analyze_user_agent(self, user_agent: str) -> int:
        """Analyze user agent for suspicious patterns. Returns score (0-10)."""
        if not user_agent:
            return 0
        
        score = 0
        ua_lower = user_agent.lower()
        
        # High-risk patterns (score +3 each)
        high_risk_patterns = [
            'sqlmap', 'nikto', 'nmap', 'scanner', 'crawler', 'spider',
            'bot', 'automation', 'script', 'python-requests', 'curl',
            'wget', 'httrack', 'grabber', 'harvester'
        ]
        
        # Medium-risk patterns (score +2 each)
        medium_risk_patterns = [
            'headless', 'phantom', 'selenium', 'webdriver', 'automated',
            'testing', 'monitoring', 'healthcheck', 'pingdom', 'uptimerobot'
        ]
        
        # Low-risk patterns (score +1 each)
        low_risk_patterns = [
            'bot', 'crawler', 'spider', 'indexer', 'feed', 'rss',
            'aggregator', 'monitor', 'checker'
        ]
        
        # Check high-risk patterns
        for pattern in high_risk_patterns:
            if pattern in ua_lower:
                score += 3
                logger.debug(f"High-risk UA pattern detected: {pattern}")
        
        # Check medium-risk patterns
        for pattern in medium_risk_patterns:
            if pattern in ua_lower:
                score += 2
                logger.debug(f"Medium-risk UA pattern detected: {pattern}")
        
        # Check low-risk patterns
        for pattern in low_risk_patterns:
            if pattern in ua_lower:
                score += 1
                logger.debug(f"Low-risk UA pattern detected: {pattern}")
        
        # Bonus for suspicious combinations
        if any(pattern in ua_lower for pattern in ['bot', 'crawler']) and any(pattern in ua_lower for pattern in ['python', 'curl', 'wget']):
            score += 2
            logger.debug("Suspicious UA combination detected")
        
        return min(score, 10)  # Cap at 10
    
    def _analyze_request_patterns(self, client_key: str, client_ip: str) -> int:
        """Analyze request patterns for suspicious behavior. Returns score (0-10)."""
        score = 0
        history = self.request_history[client_key]
        current_time = time.time()
        
        if len(history) < 5:  # Need minimum data for analysis
            return 0
        
        # Remove old requests
        recent_history = [req_time for req_time in history if current_time - req_time <= self.config.anomaly_detection_window]
        
        if len(recent_history) < 3:
            return 0
        
        # Check for burst patterns (many requests in short time)
        burst_windows = [1.0, 5.0, 10.0]  # 1s, 5s, 10s windows
        for window in burst_windows:
            burst_requests = [req_time for req_time in recent_history if current_time - req_time <= window]
            if len(burst_requests) > window * 2:  # More than 2 requests per second
                score += 2
                logger.debug(f"Burst pattern detected: {len(burst_requests)} requests in {window}s")
        
        # Check for regular intervals (automated behavior)
        if len(recent_history) >= 5:
            intervals = []
            for i in range(1, len(recent_history)):
                intervals.append(recent_history[i] - recent_history[i-1])
            
            # Check if intervals are too regular (automated)
            if len(intervals) >= 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                
                if variance < 0.1 and avg_interval < 2.0:  # Very regular, fast intervals
                    score += 3
                    logger.debug(f"Regular interval pattern detected: avg={avg_interval:.2f}s, variance={variance:.2f}")
        
        # Check for sustained high rate
        minute_requests = [req_time for req_time in recent_history if current_time - req_time <= 60.0]
        if len(minute_requests) > 50:  # More than 50 requests per minute
            score += 2
            logger.debug(f"Sustained high rate: {len(minute_requests)} requests per minute")
        
        return min(score, 10)  # Cap at 10
    
    def _detect_abuse(self, client_key: str, client_ip: str, user_agent: str = "") -> bool:
        """Enhanced abuse detection with user agent and pattern analysis."""
        # Basic rate-based detection (existing logic)
        history = self.request_history[client_key]
        current_time = time.time()
        
        # Remove old requests (older than 1 hour)
        while history and current_time - history[0] > 3600:
            history.popleft()
        
        # Check for rapid-fire requests
        recent_requests = [req_time for req_time in history if current_time - req_time <= self.config.rapid_fire_window]
        if len(recent_requests) > self.config.rapid_fire_threshold:
            logger.warning(f"Rate-based abuse detected: {len(recent_requests)} requests in {self.config.rapid_fire_window}s from {client_ip}")
            return True
        
        # Check for sustained high rate
        minute_requests = [req_time for req_time in history if current_time - req_time <= self.config.sustained_rate_window]
        if len(minute_requests) > self.config.sustained_rate_threshold:
            logger.warning(f"Rate-based abuse detected: {len(minute_requests)} requests in {self.config.sustained_rate_window}s from {client_ip}")
            return True
        
        # Enhanced anomaly detection
        if self.config.enable_user_agent_analysis:
            ua_score = self._analyze_user_agent(user_agent)
            if ua_score >= self.config.suspicious_user_agent_score_threshold:
                logger.warning(f"User agent abuse detected: score {ua_score} from {client_ip} (UA: {user_agent[:100]})")
                return True
        
        if self.config.enable_request_pattern_analysis:
            pattern_score = self._analyze_request_patterns(client_key, client_ip)
            if pattern_score >= self.config.request_pattern_score_threshold:
                logger.warning(f"Pattern-based abuse detected: score {pattern_score} from {client_ip}")
                return True
        
        return False
    
    def _refill_bucket(self, client_key: str):
        """Refill the token bucket for a client."""
        current_time = time.time()
        last_refill_time = self.last_refill[client_key]
        time_passed = current_time - last_refill_time
        
        # Calculate tokens to add
        tokens_to_add = (time_passed / 60.0) * self.config.requests_per_minute
        self.buckets[client_key] = min(
            self.config.burst_size,
            self.buckets[client_key] + tokens_to_add
        )
        self.last_refill[client_key] = current_time
    
    def allow_request(self, client_ip: str, user_agent: str = "") -> Tuple[bool, str, Dict]:
        """
        Check if request should be allowed.
        
        Returns:
            Tuple of (allowed, reason, metadata)
        """
        with self.lock:
            # Check IP allowlist/blocklist
            if not self._is_ip_allowed(client_ip):
                return False, "IP not allowed", {"ip": client_ip}
            
            client_key = self._get_client_key(client_ip, user_agent)
            
            # Check if client is blocked
            if self._is_client_blocked(client_key):
                return False, "Client blocked", {"client_key": client_key, "ip": client_ip}
            
            # Check concurrent requests
            if self.concurrent_requests[client_key] >= self.config.max_concurrent_requests:
                return False, "Too many concurrent requests", {
                    "client_key": client_key,
                    "concurrent": self.concurrent_requests[client_key],
                    "max": self.config.max_concurrent_requests
                }
            
            # Enhanced abuse detection with user agent
            if self._detect_abuse(client_key, client_ip, user_agent):
                self.blocked_clients[client_key] = time.time() + self.config.block_duration_seconds
                logger.warning(f"Blocked abusive client {client_key} from {client_ip} for {self.config.block_duration_seconds}s")
                return False, "Abuse detected", {"client_key": client_key, "ip": client_ip}
            
            # Refill bucket
            self._refill_bucket(client_key)
            
            # Check if tokens available
            if self.buckets[client_key] < 1.0:
                return False, "Rate limit exceeded", {
                    "client_key": client_key,
                    "tokens": self.buckets[client_key],
                    "rate_limit": self.config.requests_per_minute
                }
            
            # Consume token
            self.buckets[client_key] -= 1.0
            
            # Update request history
            self.request_history[client_key].append(time.time())
            
            # Increment concurrent requests
            self.concurrent_requests[client_key] += 1
            
            return True, "Request allowed", {
                "client_key": client_key,
                "tokens_remaining": self.buckets[client_key],
                "concurrent_requests": self.concurrent_requests[client_key]
            }
    
    def release_request(self, client_ip: str, user_agent: str = ""):
        """Release a concurrent request slot."""
        with self.lock:
            client_key = self._get_client_key(client_ip, user_agent)
            if client_key in self.concurrent_requests:
                self.concurrent_requests[client_key] = max(0, self.concurrent_requests[client_key] - 1)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "active_buckets": len(self.buckets),
                "blocked_clients": len(self.blocked_clients),
                "concurrent_requests": sum(self.concurrent_requests.values()),
                "total_clients": len(set(self.buckets.keys()) | set(self.concurrent_requests.keys())),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "burst_size": self.config.burst_size,
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                    "block_duration_seconds": self.config.block_duration_seconds
                }
            }
    
    def add_to_blacklist(self, ip: str):
        """Add IP to blacklist."""
        with self.lock:
            self.config.blacklisted_ips.add(ip)
            logger.info(f"Added {ip} to blacklist")
    
    def remove_from_blacklist(self, ip: str):
        """Remove IP from blacklist."""
        with self.lock:
            self.config.blacklisted_ips.discard(ip)
            logger.info(f"Removed {ip} from blacklist")
    
    def add_to_whitelist(self, ip: str):
        """Add IP to whitelist."""
        with self.lock:
            self.config.whitelisted_ips.add(ip)
            logger.info(f"Added {ip} to whitelist")
    
    def remove_from_whitelist(self, ip: str):
        """Remove IP from whitelist."""
        with self.lock:
            self.config.whitelisted_ips.discard(ip)
            logger.info(f"Removed {ip} from whitelist")


def add_rate_limiting(app):
    """Add rate limiting middleware to FastAPI app."""
    from fastapi import Request, HTTPException
    from fastapi.responses import JSONResponse
    
    # Create rate limiter instance
    config = RateLimitConfig(
        requests_per_minute=100,
        burst_size=10,
        max_concurrent_requests=5,
        enable_ip_blacklist=True,
        enable_ip_whitelist=False
    )
    rate_limiter = TokenBucketRateLimiter(config)
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Rate limiting middleware."""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Check rate limit
        allowed, reason, meta = rate_limiter.allow_request(client_ip, user_agent)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": reason,
                    "retry_after": meta.get("retry_after", 60)
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(meta.get("tokens_remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(meta.get("reset_time", 0))
        
        return response

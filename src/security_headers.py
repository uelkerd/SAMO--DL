#!/usr/bin/env python3
"""🛡️ Security Headers Middleware.
==============================
Flask middleware for adding security headers and implementing security policies.
"""

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List

import yaml
from flask import Flask, Response, g, request

logger = logging.getLogger(__name__)


@dataclass
class SecurityHeadersConfig:
    """Security headers configuration."""

    enable_csp: bool = True
    enable_hsts: bool = True
    enable_x_frame_options: bool = True
    enable_x_content_type_options: bool = True
    enable_x_xss_protection: bool = True
    enable_referrer_policy: bool = True
    enable_permissions_policy: bool = True
    enable_cross_origin_embedder_policy: bool = True
    enable_cross_origin_opener_policy: bool = True
    enable_cross_origin_resource_policy: bool = True
    enable_origin_agent_cluster: bool = True
    enable_strict_transport_security: bool = True
    enable_content_security_policy: bool = True
    enable_request_id: bool = True
    enable_correlation_id: bool = True
    # Enhanced user agent analysis
    enable_enhanced_ua_analysis: bool = True
    ua_suspicious_score_threshold: int = 4  # Score threshold for suspicious UAs
    ua_blocking_enabled: bool = False  # Whether to block suspicious UAs (vs just log)


class SecurityHeadersMiddleware:
    """Flask middleware for adding security headers and implementing security policies.

    Features:
    - Content Security Policy (CSP)
    - HTTP Strict Transport Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer Policy
    - Permissions Policy
    - Cross-Origin policies
    - Request correlation
    - Security monitoring
    """

    def __init__(self, app: Flask, config: SecurityHeadersConfig):
        self.app = app
        self.config = config
        # Load CSP from YAML config if available
        self.csp_policy = None
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "../configs/security.yaml"
            )
            with open(config_path) as f:
                security_config = yaml.safe_load(f)
                self.csp_policy = (
                    security_config.get("security_headers", {})
                    .get("headers", {})
                    .get("Content-Security-Policy")
                )
        except Exception as e:
            logger.warning("Could not load CSP from config: %s", e)

        # Register middleware
        app.before_request(self._before_request)
        app.after_request(self._after_request)

        # Generate nonce for CSP
        self._csp_nonce = secrets.token_hex(16)

    def _before_request(self):
        """Process request before handling."""
        # Generate request ID for correlation
        if self.config.enable_request_id:
            request_data = f"{time.time()}:{request.remote_addr}:{secrets.token_hex(8)}"
            g.request_id = hashlib.sha256(request_data.encode()).hexdigest()

        # Generate correlation ID
        if self.config.enable_correlation_id:
            g.correlation_id = request.headers.get("X-Correlation-ID", g.request_id)

        # Check for high-risk requests and block them
        if self.config.ua_blocking_enabled:
            user_agent = request.headers.get("User-Agent", "")
            ua_analysis = self._analyze_user_agent_enhanced(user_agent)

            if ua_analysis["risk_level"] in ["high", "very_high"]:
                # Store blocking information for after_request logging
                g.security_patterns = [
                    f"BLOCKED: High-risk user agent - {ua_analysis['category']} "
                    f"(score: {ua_analysis['score']})"
                ]
                g.block_reason = f"User agent risk level: {ua_analysis['risk_level']}"
                g.ua_analysis = ua_analysis

                # Return 403 Forbidden response
                from flask import make_response
                response = make_response(
                    "Access Forbidden - High-risk user agent detected", 403
                )
                response.headers["Content-Type"] = "text/plain"
                return response

        # Log security-relevant request information
        self._log_security_info()

        # Explicit return None for consistency (PEP8 compliance)
        return None

    def _after_request(self, response: Response) -> Response:
        """Process response after handling."""
        # Add security headers
        self._add_security_headers(response)

        # Add request correlation headers
        self._add_correlation_headers(response)

        # Log security-relevant response information
        self._log_response_security(response)

        # Log blocking information if request was blocked
        if hasattr(g, 'security_patterns'):
            logger.warning("Request blocked: %s", g.security_patterns)
            if hasattr(g, 'block_reason'):
                logger.warning("Block reason: %s", g.block_reason)
            if hasattr(g, 'ua_analysis'):
                logger.warning("User agent analysis: %s", g.ua_analysis)

        return response

    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        # Content Security Policy
        if self.config.enable_content_security_policy:
            csp_policy = self._build_csp_policy()
            response.headers["Content-Security-Policy"] = csp_policy

        # HTTP Strict Transport Security
        if self.config.enable_strict_transport_security:
            hsts_value = "max-age=31536000; includeSubDomains; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # X-Frame-Options
        if self.config.enable_x_frame_options:
            response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options
        if self.config.enable_x_content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection
        if self.config.enable_x_xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer Policy
        if self.config.enable_referrer_policy:
            referrer_policy = "strict-origin-when-cross-origin"
            response.headers["Referrer-Policy"] = referrer_policy

        # Permissions Policy
        if self.config.enable_permissions_policy:
            permissions_policy = self._build_permissions_policy()
            response.headers["Permissions-Policy"] = permissions_policy

        # Cross-Origin Embedder Policy
        if self.config.enable_cross_origin_embedder_policy:
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

        # Cross-Origin Opener Policy
        if self.config.enable_cross_origin_opener_policy:
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # Cross-Origin Resource Policy
        if self.config.enable_cross_origin_resource_policy:
            response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Origin-Agent-Cluster
        if self.config.enable_origin_agent_cluster:
            response.headers["Origin-Agent-Cluster"] = "?1"

    def _build_csp_policy(self) -> str:
        """Return CSP policy from config, or a secure default if not set."""
        if self.csp_policy:
            return self.csp_policy
        # Production-safe fallback default with comprehensive protection
        return (
            "default-src 'self'; "
            "script-src 'self'; "  # Only allow same-origin scripts
            "style-src 'self'; "  # Only allow same-origin styles
            "img-src 'self' data: https:; "  # Data URIs and HTTPS images
            "font-src 'self' data:; "  # Data URI fonts
            "connect-src 'self' https:; "  # Allow HTTPS connections
            "media-src 'self' https:; "  # Allow HTTPS media
            "object-src 'none'; "  # Block all plugins
            "base-uri 'self'; "  # Restrict base URI
            "form-action 'self'; "  # Restrict form submissions
            "frame-ancestors 'none'; "  # Block embedding in iframes
            "upgrade-insecure-requests; "  # Upgrade HTTP to HTTPS
            "block-all-mixed-content"  # Block mixed content
        )

    def _build_permissions_policy(self) -> str:
        """Build Permissions Policy."""
        policies = [
            "accelerometer=()",
            "ambient-light-sensor=()",
            "autoplay=()",
            "battery=()",
            "camera=()",
            "cross-origin-isolated=()",
            "display-capture=()",
            "document-domain=()",
            "encrypted-media=()",
            "execution-while-not-rendered=()",
            "execution-while-out-of-viewport=()",
            "fullscreen=()",
            "geolocation=()",
            "gyroscope=()",
            "keyboard-map=()",
            "magnetometer=()",
            "microphone=()",
            "midi=()",
            "navigation-override=()",
            "payment=()",
            "picture-in-picture=()",
            "publickey-credentials-get=()",
            "screen-wake-lock=()",
            "sync-xhr=()",
            "usb=()",
            "web-share=()",
            "xr-spatial-tracking=()",
        ]
        return ", ".join(policies)

    def _add_correlation_headers(self, response: Response):
        """Add request correlation headers."""
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id

        if hasattr(g, "correlation_id"):
            response.headers["X-Correlation-ID"] = g.correlation_id

    def _log_security_info(self):
        """Log security-relevant request information."""
        security_info = {
            "timestamp": time.time(),
            "request_id": getattr(g, "request_id", None),
            "correlation_id": getattr(g, "correlation_id", None),
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get("User-Agent", ""),
            "content_type": request.headers.get("Content-Type", ""),
            "content_length": request.headers.get("Content-Length", ""),
            "referer": request.headers.get("Referer", ""),
            "origin": request.headers.get("Origin", ""),
            "x_forwarded_for": request.headers.get("X-Forwarded-For", ""),
            "x_real_ip": request.headers.get("X-Real-IP", ""),
        }

        # Log suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns()
        if suspicious_patterns:
            security_info["suspicious_patterns"] = suspicious_patterns
            logger.warning("Security warning: %s", suspicious_patterns)

        logger.info(f"Security audit: {security_info}")

    def _analyze_user_agent_enhanced(self, user_agent: str) -> dict:
        """Enhanced user agent analysis with scoring and detailed categorization."""
        if not user_agent:
            return {
                "score": 0,
                "category": "empty",
                "patterns": [],
                "risk_level": "low",
            }

        score = 0
        patterns = []
        ua_lower = user_agent.lower()

        # Legitimate bot whitelist (negative scoring)
        legitimate_bots = [
            "googlebot",
            "bingbot",
            "slurp",
            "duckduckbot",
            "facebookexternalhit",
            "twitterbot",
            "linkedinbot",
            "whatsapp",
            "telegrambot",
            "discordbot",
            "slackbot",
            "github-camo",
            "github-actions",
            "vercel",
            "netlify",
            "uptimerobot",
            "pingdom",
            "statuscake",
            "monitor",
            "healthcheck",
        ]

        # High-risk patterns (score +3 each)
        high_risk_patterns = [
            "sqlmap",
            "nikto",
            "nmap",
            "scanner",
            "grabber",
            "harvester",
            "exploit",
            "vulnerability",
            "penetration",
            "security",
            "audit",
        ]

        # Medium-risk patterns (score +2 each)
        medium_risk_patterns = [
            "headless",
            "phantom",
            "selenium",
            "webdriver",
            "automated",
            "testing",
            "script",
            "python-requests",
            "curl",
            "wget",
            "httrack",
            "scraper",
            "crawler",
            "spider",
            "bot",
        ]

        # Low-risk patterns (score +1 each)
        low_risk_patterns = [
            "indexer",
            "feed",
            "rss",
            "aggregator",
            "monitor",
            "checker",
            "validator",
            "linter",
            "checker",
            "analyzer",
        ]

        # Check legitimate bots first (negative scoring)
        for bot in legitimate_bots:
            if bot in ua_lower:
                score -= 2
                patterns.append(f"legitimate_bot:{bot}")
                logger.debug("Legitimate bot detected: %s", bot)

        # Check high-risk patterns
        for pattern in high_risk_patterns:
            if pattern in ua_lower:
                score += 3
                patterns.append(f"high_risk:{pattern}")
                logger.debug("High-risk UA pattern detected: %s", pattern)

        # Check medium-risk patterns
        for pattern in medium_risk_patterns:
            if pattern in ua_lower:
                score += 2
                patterns.append(f"medium_risk:{pattern}")
                logger.debug("Medium-risk UA pattern detected: %s", pattern)

        # Check low-risk patterns
        for pattern in low_risk_patterns:
            if pattern in ua_lower:
                score += 1
                patterns.append(f"low_risk:{pattern}")
                logger.debug("Low-risk UA pattern detected: %s", pattern)

        # Bonus for suspicious combinations
        bot_patterns = ["bot", "crawler", "spider"]
        script_patterns = ["python", "curl", "wget", "script"]

        if any(pattern in ua_lower for pattern in bot_patterns) and any(
            pattern in ua_lower for pattern in script_patterns
        ):
            score += 2
            patterns.append("suspicious_combination")
            logger.debug("Suspicious UA combination detected")

        # Check for missing or generic user agents
        if user_agent in ["", "null", "undefined", "unknown", "anonymous"]:
            score += 2
            patterns.append("missing_generic_ua")
            logger.debug("Missing or generic user agent detected")

        # Determine category and risk level
        if score <= -1:
            category = "legitimate_bot"
            risk_level = "very_low"
        elif score <= 1:
            category = "normal"
            risk_level = "low"
        elif score <= 3:
            category = "suspicious"
            risk_level = "medium"
        elif score <= 6:
            category = "high_risk"
            risk_level = "high"
        else:
            category = "malicious"
            risk_level = "very_high"

        return {
            "score": max(0, score),  # Don't return negative scores
            "category": category,
            "patterns": patterns,
            "risk_level": risk_level,
            "user_agent": user_agent[:100],  # Truncate for logging
        }

    def _detect_suspicious_patterns(self) -> List[str]:
        """Enhanced suspicious pattern detection with user agent analysis."""
        patterns = []

        # Check for suspicious headers
        suspicious_headers = [
            "X-Forwarded-Host",
            "X-Original-URL",
            "X-Rewrite-URL",
            "X-Custom-IP-Authorization",
        ]

        for header in suspicious_headers:
            if header in request.headers:
                patterns.append(f"Suspicious header: {header}")

        # Check for suspicious query parameters
        suspicious_params = [
            "cmd",
            "exec",
            "system",
            "eval",
            "script",
            "union",
            "select",
            "insert",
            "update",
            "delete",
        ]

        for param in suspicious_params:
            if param in request.args:
                patterns.append(f"Suspicious query param: {param}")

        # Enhanced user agent analysis
        if self.config.enable_enhanced_ua_analysis:
            user_agent = request.headers.get("User-Agent", "")
            ua_analysis = self._analyze_user_agent_enhanced(user_agent)

            if ua_analysis["score"] >= self.config.ua_suspicious_score_threshold:
                ua_msg = (
                    f"Suspicious user agent: {ua_analysis['category']} "
                    f"(score: {ua_analysis['score']})"
                )
                patterns.append(ua_msg)

                # Log detailed analysis
                logger.warning("User agent analysis: %s", ua_analysis)

                # Note: High-risk user agents are now blocked in _before_request
                # This is just for logging and pattern detection

        return patterns

    def _log_response_security(self, response: Response):
        """Log security-relevant response information."""
        security_info = {
            "timestamp": time.time(),
            "request_id": getattr(g, "request_id", None),
            "correlation_id": getattr(g, "correlation_id", None),
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "security_headers": {
                "csp": response.headers.get("Content-Security-Policy", ""),
                "hsts": response.headers.get("Strict-Transport-Security", ""),
                "x_frame_options": response.headers.get("X-Frame-Options", ""),
                "x_content_type_options": response.headers.get(
                    "X-Content-Type-Options", ""
                ),
                "x_xss_protection": response.headers.get("X-XSS-Protection", ""),
                "referrer_policy": response.headers.get("Referrer-Policy", ""),
                "permissions_policy": response.headers.get("Permissions-Policy", ""),
            },
        }

        logger.info("Response security: %s", security_info)

    def get_security_stats(self) -> Dict:
        """Get security headers statistics."""
        return {
            "config": {
                "enable_csp": self.config.enable_content_security_policy,
                "enable_hsts": self.config.enable_strict_transport_security,
                "enable_x_frame_options": self.config.enable_x_frame_options,
                "enable_x_content_type_options": (
                    self.config.enable_x_content_type_options
                ),
                "enable_x_xss_protection": self.config.enable_x_xss_protection,
                "enable_referrer_policy": self.config.enable_referrer_policy,
                "enable_permissions_policy": self.config.enable_permissions_policy,
                "enable_cross_origin_embedder_policy": (
                    self.config.enable_cross_origin_embedder_policy
                ),
                "enable_cross_origin_opener_policy": (
                    self.config.enable_cross_origin_opener_policy
                ),
                "enable_cross_origin_resource_policy": (
                    self.config.enable_cross_origin_resource_policy
                ),
                "enable_origin_agent_cluster": self.config.enable_origin_agent_cluster,
                "enable_request_id": self.config.enable_request_id,
                "enable_correlation_id": self.config.enable_correlation_id,
                "enable_enhanced_ua_analysis": self.config.enable_enhanced_ua_analysis,
                "ua_suspicious_score_threshold": (
                    self.config.ua_suspicious_score_threshold
                ),
                "ua_blocking_enabled": self.config.ua_blocking_enabled,
            },
            "csp_nonce": self._csp_nonce,
        }

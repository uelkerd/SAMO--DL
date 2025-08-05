#!/usr/bin/env python3
"""
🛡️ Security Headers Middleware
==============================
Flask middleware for adding security headers and implementing security policies.
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from flask import Flask, request, Response, g
import time
import hashlib
import secrets

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

class SecurityHeadersMiddleware:
    """
    Flask middleware for adding security headers and implementing security policies.
    
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
        
        # Register middleware
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        
        # Generate nonce for CSP
        self._csp_nonce = secrets.token_hex(16)
    
    def _before_request(self):
        """Process request before handling."""
        # Generate request ID for correlation
        if self.config.enable_request_id:
            g.request_id = hashlib.sha256(
                f"{time.time()}:{request.remote_addr}:{secrets.token_hex(8)}".encode()
            ).hexdigest()[:16]
        
        # Generate correlation ID
        if self.config.enable_correlation_id:
            g.correlation_id = request.headers.get('X-Correlation-ID', g.request_id)
        
        # Log security-relevant request information
        self._log_security_info()
    
    def _after_request(self, response: Response) -> Response:
        """Process response after handling."""
        # Add security headers
        self._add_security_headers(response)
        
        # Add request correlation headers
        self._add_correlation_headers(response)
        
        # Log security-relevant response information
        self._log_response_security(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        # Content Security Policy
        if self.config.enable_content_security_policy:
            csp_policy = self._build_csp_policy()
            response.headers['Content-Security-Policy'] = csp_policy
        
        # HTTP Strict Transport Security
        if self.config.enable_strict_transport_security:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        
        # X-Frame-Options
        if self.config.enable_x_frame_options:
            response.headers['X-Frame-Options'] = 'DENY'
        
        # X-Content-Type-Options
        if self.config.enable_x_content_type_options:
            response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # X-XSS-Protection
        if self.config.enable_x_xss_protection:
            response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer Policy
        if self.config.enable_referrer_policy:
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy
        if self.config.enable_permissions_policy:
            permissions_policy = self._build_permissions_policy()
            response.headers['Permissions-Policy'] = permissions_policy
        
        # Cross-Origin Embedder Policy
        if self.config.enable_cross_origin_embedder_policy:
            response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        
        # Cross-Origin Opener Policy
        if self.config.enable_cross_origin_opener_policy:
            response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        
        # Cross-Origin Resource Policy
        if self.config.enable_cross_origin_resource_policy:
            response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        
        # Origin-Agent-Cluster
        if self.config.enable_origin_agent_cluster:
            response.headers['Origin-Agent-Cluster'] = '?1'
    
    def _build_csp_policy(self) -> str:
        """Build Content Security Policy."""
        directives = []
        
        # Default source
        directives.append("default-src 'self'")
        
        # Script sources
        directives.append("script-src 'self' 'unsafe-inline' 'unsafe-eval'")
        
        # Style sources
        directives.append("style-src 'self' 'unsafe-inline'")
        
        # Image sources
        directives.append("img-src 'self' data: https:")
        
        # Font sources
        directives.append("font-src 'self' data:")
        
        # Connect sources (for API calls)
        directives.append("connect-src 'self'")
        
        # Frame sources
        directives.append("frame-src 'none'")
        
        # Object sources
        directives.append("object-src 'none'")
        
        # Base URI
        directives.append("base-uri 'self'")
        
        # Form action
        directives.append("form-action 'self'")
        
        # Frame ancestors
        directives.append("frame-ancestors 'none'")
        
        # Upgrade insecure requests
        directives.append("upgrade-insecure-requests")
        
        return "; ".join(directives)
    
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
            "xr-spatial-tracking=()"
        ]
        return ", ".join(policies)
    
    def _add_correlation_headers(self, response: Response):
        """Add request correlation headers."""
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        if hasattr(g, 'correlation_id'):
            response.headers['X-Correlation-ID'] = g.correlation_id
    
    def _log_security_info(self):
        """Log security-relevant request information."""
        security_info = {
            'timestamp': time.time(),
            'request_id': getattr(g, 'request_id', None),
            'correlation_id': getattr(g, 'correlation_id', None),
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'content_type': request.headers.get('Content-Type', ''),
            'content_length': request.headers.get('Content-Length', ''),
            'referer': request.headers.get('Referer', ''),
            'origin': request.headers.get('Origin', ''),
            'x_forwarded_for': request.headers.get('X-Forwarded-For', ''),
            'x_real_ip': request.headers.get('X-Real-IP', ''),
        }
        
        # Log suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns()
        if suspicious_patterns:
            security_info['suspicious_patterns'] = suspicious_patterns
            logger.warning(f"Security warning: {suspicious_patterns}")
        
        logger.info(f"Security audit: {security_info}")
    
    def _detect_suspicious_patterns(self) -> List[str]:
        """Detect suspicious patterns in request."""
        patterns = []
        
        # Check for suspicious headers
        suspicious_headers = [
            'X-Forwarded-Host',
            'X-Original-URL',
            'X-Rewrite-URL',
            'X-Custom-IP-Authorization'
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                patterns.append(f"Suspicious header: {header}")
        
        # Check for suspicious query parameters
        suspicious_params = [
            'cmd', 'exec', 'system', 'eval', 'script',
            'union', 'select', 'insert', 'update', 'delete'
        ]
        
        for param in suspicious_params:
            if param in request.args:
                patterns.append(f"Suspicious query param: {param}")
        
        # Check for suspicious user agents
        suspicious_user_agents = [
            'sqlmap', 'nikto', 'nmap', 'wget', 'curl',
            'python-requests', 'scanner', 'bot'
        ]
        
        user_agent = request.headers.get('User-Agent', '').lower()
        for agent in suspicious_user_agents:
            if agent in user_agent:
                patterns.append(f"Suspicious user agent: {agent}")
        
        return patterns
    
    def _log_response_security(self, response: Response):
        """Log security-relevant response information."""
        security_info = {
            'timestamp': time.time(),
            'request_id': getattr(g, 'request_id', None),
            'correlation_id': getattr(g, 'correlation_id', None),
            'status_code': response.status_code,
            'content_type': response.headers.get('Content-Type', ''),
            'content_length': response.headers.get('Content-Length', ''),
            'security_headers': {
                'csp': response.headers.get('Content-Security-Policy', ''),
                'hsts': response.headers.get('Strict-Transport-Security', ''),
                'x_frame_options': response.headers.get('X-Frame-Options', ''),
                'x_content_type_options': response.headers.get('X-Content-Type-Options', ''),
                'x_xss_protection': response.headers.get('X-XSS-Protection', ''),
                'referrer_policy': response.headers.get('Referrer-Policy', ''),
                'permissions_policy': response.headers.get('Permissions-Policy', ''),
            }
        }
        
        logger.info(f"Response security: {security_info}")
    
    def get_security_stats(self) -> Dict:
        """Get security headers statistics."""
        return {
            "config": {
                "enable_csp": self.config.enable_content_security_policy,
                "enable_hsts": self.config.enable_strict_transport_security,
                "enable_x_frame_options": self.config.enable_x_frame_options,
                "enable_x_content_type_options": self.config.enable_x_content_type_options,
                "enable_x_xss_protection": self.config.enable_x_xss_protection,
                "enable_referrer_policy": self.config.enable_referrer_policy,
                "enable_permissions_policy": self.config.enable_permissions_policy,
                "enable_cross_origin_embedder_policy": self.config.enable_cross_origin_embedder_policy,
                "enable_cross_origin_opener_policy": self.config.enable_cross_origin_opener_policy,
                "enable_cross_origin_resource_policy": self.config.enable_cross_origin_resource_policy,
                "enable_origin_agent_cluster": self.config.enable_origin_agent_cluster,
                "enable_request_id": self.config.enable_request_id,
                "enable_correlation_id": self.config.enable_correlation_id,
            },
            "csp_nonce": self._csp_nonce
        } 
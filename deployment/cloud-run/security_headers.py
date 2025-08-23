#!/usr/bin/env python3
"""Security Headers Module for Cloud Run API."""

from flask import Flask, request, g
from typing import Dict, Any

def add_security_headers(app: Flask) -> None:
    """Add comprehensive security headers to Flask app."""

    @app.after_request
    def add_headers(response):
        # Path-aware Content Security Policy
        csp_base = (
            "default-src 'sel"; "
            "script-src 'sel" "unsafe-inline'; "
            "style-src 'sel" "unsafe-inline'; "
            "img-src 'sel" data: https:; "
            "font-src 'sel"; "
            "connect-src 'sel"; "
            "frame-ancestors 'none';"
        )
        if request.path.startswith('/docs'):
            nonce = getattr(g, 'csp_nonce', None)
            if nonce:
                csp_docs = (
                    "default-src 'sel"; "
                    "script-src "sel" "nonce-{nonce}'; "
                    "style-src "sel" "nonce-{nonce}'; "
                    "img-src 'sel" data: https:; "
                    "font-src 'sel"; "
                    "connect-src 'sel"; "
                    "frame-ancestors 'none';"
                )
            else:
                # Reject request if no nonce is available for docs
                return "Content Security Policy violation: nonce required for /docs", 403
            response.headers['Content-Security-Policy'] = csp_docs
        else:
            response.headers['Content-Security-Policy'] = csp_base

        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosnif"
        response.headers["X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Remove server information
        response.headers.pop('Server', None)

        return response

#!/usr/bin/env python3
"""
🔒 Shared Security Setup
========================
Common security configuration and middleware setup for deployment scripts.
"""

import os
from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig


def create_security_config(environment: str = "development") -> SecurityHeadersConfig:
    """
    Create security configuration based on environment.

    Args:
        environment: Environment name ('development', 'testing', 'production')

    Returns:
        Configured SecurityHeadersConfig instance
    """
    is_production = environment.lower() == "production"

    return SecurityHeadersConfig(
        enable_csp=True,
        enable_hsts=True,
        enable_x_frame_options=True,
        enable_x_content_type_options=True,
        enable_x_xss_protection=True,
        enable_referrer_policy=True,
        enable_permissions_policy=True,
        enable_cross_origin_embedder_policy=True,
        enable_cross_origin_opener_policy=True,
        enable_cross_origin_resource_policy=True,
        enable_origin_agent_cluster=True,
        enable_request_id=True,
        enable_correlation_id=True,
        enable_enhanced_ua_analysis=True,
        ua_suspicious_score_threshold=4,
        ua_blocking_enabled=is_production  # Block suspicious UAs in production only
    )


def setup_security_middleware(
    app, environment: str = "development"
) -> SecurityHeadersMiddleware:
    """
    Set up security headers middleware for a Flask app.

    Args:
        app: Flask application instance
        environment: Environment name ('development', 'testing', 'production')

    Returns:
        Configured SecurityHeadersMiddleware instance
    """
    config = create_security_config(environment)
    middleware = SecurityHeadersMiddleware(app, config)

    # Log security setup
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "✅ Security headers middleware initialized for %s environment", environment
    )

    return middleware


def get_environment() -> str:
    """
    Determine current environment from environment variables.

    Returns:
        Environment name ('development', 'testing', 'production')
    """
    env = os.environ.get('FLASK_ENV', 'development').lower()

    # Map common environment names
    if env in ['prod', 'production', 'live']:
        return 'production'
    if env in ['test', 'testing', 'staging']:
        return 'testing'
    return 'development'

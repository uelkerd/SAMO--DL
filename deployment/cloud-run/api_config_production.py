#!/usr/bin/env python3
"""
Production API Configuration for SAMO Cloud Run
Optimized rate limiting and model loading settings
"""

import os
from typing import Dict, Any, ClassVar

class ProductionConfig:
    """Production configuration for SAMO API."""

    # Rate Limiting Configuration (More permissive for production)
    RATE_LIMIT_CONFIG: ClassVar[Dict[str, Any]] = {
        "requests_per_minute": 300,  # Increased from 60
        "burst_size": 50,           # Increased from 10
        "window_size_seconds": 60,
        "block_duration_seconds": 120,  # Reduced from 300
        "max_concurrent_requests": 20,  # Increased from 5

        # Abuse detection (More lenient)
        "rapid_fire_threshold": 30,     # Increased from 10
        "sustained_rate_threshold": 600, # Increased from 200
        "rapid_fire_window": 1.0,
        "sustained_rate_window": 60.0,

        # Disable some strict checks for production
        "enable_user_agent_analysis": False,
        "enable_request_pattern_analysis": False,
    }

    # Model Configuration
    MODEL_CONFIG: ClassVar[Dict[str, Any]] = {
        "emotion_model_id": os.getenv("EMOTION_MODEL_ID", "0xmnrv/samo"),
        "voice_model_size": os.getenv("VOICE_MODEL_SIZE", "base"),
        "text_model_size": os.getenv("TEXT_MODEL_SIZE", "t5-small"),
        "enable_model_fallbacks": True,
        "lazy_model_loading": True,
        "model_timeout_seconds": 300,
    }

    # Authentication Configuration
    AUTH_CONFIG: ClassVar[Dict[str, Any]] = {
        "jwt_secret": os.getenv("JWT_SECRET", "your-production-secret-key"),
        "jwt_expiry_minutes": int(os.getenv("JWT_EXPIRY_MINUTES", "30")),
        "enable_registration": os.getenv("ENABLE_REGISTRATION", "true").lower() == "true",
        "require_email_verification": False,  # Simplified for demo
    }

    # Logging Configuration
    LOGGING_CONFIG: ClassVar[Dict[str, Any]] = {
        "level": "INFO",
        "enable_access_logs": True,
        "enable_performance_logs": True,
        "log_format": "json",  # Better for Cloud Run
    }

    @classmethod
    def get_rate_limit_config(cls) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return cls.RATE_LIMIT_CONFIG.copy()

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration."""
        return cls.MODEL_CONFIG.copy()

    @classmethod
    def get_auth_config(cls) -> Dict[str, Any]:
        """Get authentication configuration."""
        return cls.AUTH_CONFIG.copy()

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return cls.LOGGING_CONFIG.copy()

# Environment-specific overrides
def get_production_overrides() -> Dict[str, Any]:
    """Get production-specific overrides."""
    overrides = {}

    # Cloud Run specific settings
    if os.getenv("K_SERVICE"):  # Running on Cloud Run
        overrides.update({
            "rate_limit_requests_per_minute": 500,  # Higher for Cloud Run
            "rate_limit_burst_size": 100,
            "enable_health_check_bypass": True,
        })

    # Performance mode
    if os.getenv("PERFORMANCE_MODE") == "high":
        overrides.update({
            "rate_limit_requests_per_minute": 1000,
            "rate_limit_burst_size": 200,
            "max_concurrent_requests": 50,
        })

    return overrides

# Usage example for API initialization
def configure_production_api(_app):
    """Configure API with production settings."""
    config = ProductionConfig()
    overrides = get_production_overrides()

    # Apply configurations
    rate_config = config.get_rate_limit_config()
    rate_config.update(overrides)

    return {
        "rate_limiting": rate_config,
        "models": config.get_model_config(),
        "auth": config.get_auth_config(),
        "logging": config.get_logging_config()
    }

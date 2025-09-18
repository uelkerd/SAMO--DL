"""
Security-first host binding configuration for SAMO-DL applications.

This module provides secure host binding logic that prevents accidental
exposure to all network interfaces during development while allowing
proper containerized deployment in production environments.
"""
import os
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Security constants
DEFAULT_SECURE_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
ALL_INTERFACES_HOST = "0.0.0.0"

# Environment variables that indicate production/containerized deployment
PRODUCTION_INDICATORS = {
    "PRODUCTION": "true",
    "DOCKER_CONTAINER": "true", 
    "CLOUD_RUN_SERVICE": "true",
    "KUBERNETES_SERVICE_HOST": "true",
    "CONTAINER": "true",
    "BIND_ALL_INTERFACES": "true"
}

# Environment variables that indicate development mode
DEVELOPMENT_INDICATORS = {
    "DEVELOPMENT": "true",
    "DEBUG": "true",
    "LOCAL_DEVELOPMENT": "true"
}


def is_production_environment() -> bool:
    """
    Determine if the application is running in a production environment.
    
    Returns:
        bool: True if running in production, False otherwise
    """
    # Check for explicit production indicators
    for env_var, expected_value in PRODUCTION_INDICATORS.items():
        if os.environ.get(env_var) == expected_value:
            return True
    
    # Check for containerized environment indicators
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
        
    return False


def is_development_environment() -> bool:
    """
    Determine if the application is running in a development environment.
    
    Returns:
        bool: True if running in development, False otherwise
    """
    for env_var, expected_value in DEVELOPMENT_INDICATORS.items():
        if os.environ.get(env_var) == expected_value:
            return True
    return False


def get_secure_host_binding(default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """
    Get secure host binding configuration based on environment.
    
    This function implements a security-first approach:
    1. Defaults to localhost (127.0.0.1) for maximum security
    2. Only binds to all interfaces (0.0.0.0) in explicitly configured production environments
    3. Provides comprehensive logging for security auditing
    
    Args:
        default_port: Default port number if not specified in environment
        
    Returns:
        Tuple[str, int]: (host, port) configuration
        
    Security Notes:
        - 127.0.0.1: Only accessible from localhost (secure for development)
        - 0.0.0.0: Accessible from all network interfaces (required for containers)
    """
    # Get port from environment or use default
    port = int(os.environ.get("PORT", default_port))
    
    # Check for explicitly configured host
    explicit_host = os.environ.get("HOST")
    if explicit_host:
        logger.info("Using explicitly configured host: %s", explicit_host)
        if explicit_host == ALL_INTERFACES_HOST:
            logger.warning("⚠️  EXPLICIT CONFIGURATION: Binding to all interfaces (0.0.0.0)")
            logger.warning("🔒 Ensure proper network security and firewall rules are in place")
        return explicit_host, port
    
    # Security-first default: localhost only
    host = DEFAULT_SECURE_HOST
    
    # Only bind to all interfaces in production environments
    if is_production_environment():
        host = ALL_INTERFACES_HOST
        logger.warning("⚠️  PRODUCTION MODE: Binding to all interfaces (0.0.0.0)")
        logger.warning("🔒 Containerized deployment detected - external access required")
        logger.warning("🚨 SECURITY: Server accessible from all network interfaces")
        logger.warning("🚨 Ensure proper authentication, authorization, and network security")
    else:
        logger.info("🔒 DEVELOPMENT MODE: Binding to localhost only (%s)", host)
        logger.info("✅ External access blocked - only localhost connections allowed")
        logger.info("💡 To enable external access, set production environment variables")
    
    return host, port


def validate_host_binding(host: str, port: int) -> None:
    """
    Validate host binding configuration and log security implications.
    
    Args:
        host: Host address to bind to
        port: Port number to bind to
        
    Raises:
        ValueError: If host binding configuration is invalid
    """
    if not host or not isinstance(host, str):
        raise ValueError("Host must be a non-empty string")
    
    if not isinstance(port, int) or port <= 0 or port > 65535:
        raise ValueError("Port must be an integer between 1 and 65535")
    
    if host == ALL_INTERFACES_HOST:
        logger.warning("🚨 SECURITY WARNING: Server will be accessible from all network interfaces")
        logger.warning("🚨 Ensure proper network security, firewall rules, and authentication")
        logger.warning("🚨 Consider using a reverse proxy or load balancer for production")
    elif host == DEFAULT_SECURE_HOST:
        logger.info("✅ SECURE: Server bound to localhost only")
        logger.info("✅ External network access blocked")
    else:
        logger.warning("⚠️  CUSTOM HOST: Using non-standard host binding: %s", host)
        logger.warning("⚠️  Verify this configuration meets your security requirements")


def get_binding_security_summary(host: str, port: int) -> str:
    """
    Get a security summary of the host binding configuration.
    
    Args:
        host: Host address
        port: Port number
        
    Returns:
        str: Security summary message
    """
    if host == ALL_INTERFACES_HOST:
        return f"⚠️  SECURITY: Server accessible from all interfaces on port {port}"
    elif host == DEFAULT_SECURE_HOST:
        return f"✅ SECURE: Server bound to localhost only on port {port}"
    else:
        return f"⚠️  CUSTOM: Server bound to {host} on port {port}"

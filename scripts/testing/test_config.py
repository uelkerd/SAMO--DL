#!/usr/bin/env python3
"""Centralized Test Configuration Provides consistent configuration for all testing
scripts."""

import os
import argparse
import secrets
from typing import Optional


class TestConfig:
    """Centralized configuration for all testing scripts."""
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or self._get_base_url()
        self.api_key = api_key or self._get_api_key()
    
    def _get_base_url(self) -> str:
        """Get base URL from environment or command line arguments."""
        # Priority: CLI arg > environment variable > default
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--base-url', help='API base URL')
        args, _ = parser.parse_known_args()
        
        if args.base_url:
            return args.base_url.rstrip('/')
        
        # Check multiple environment variables for flexibility
        env_url = (os.environ.get("API_BASE_URL") or 
                  os.environ.get("CLOUD_RUN_API_URL") or 
                  os.environ.get("MODEL_API_BASE_URL"))
        
        if env_url:
            return env_url.rstrip('/')
        
        # If no URL is provided, raise an error to force explicit configuration
        raise ValueError(
            "No API base URL provided. Please set one of:\n"
            "  - API_BASE_URL environment variable\n"
            "  - CLOUD_RUN_API_URL environment variable\n"
            "  - MODEL_API_BASE_URL environment variable\n"
            "  - --base-url command line argument"
        )
    
    def _get_api_key(self) -> str:
        """Get API key from environment or generate securely."""
        # Priority: environment variable > secure generation
        api_key = os.environ.get("API_KEY")
        if api_key:
            return api_key
        
        # Fallback: generate a secure random key
        return f"samo-admin-key-{secrets.token_urlsafe(32)}"
    
    def get_headers(self) -> dict:
        """Get standard headers for API requests."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def get_rate_limit_requests(self) -> int:
        """Get number of requests for rate limiting tests."""
        return int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))


class APIClient:
    """Centralized API client with consistent error handling."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.base_url = config.base_url
        self.headers = config.get_headers()
    
    def get(self, endpoint: str, **kwargs) -> dict:
        """Make GET request with consistent error handling."""
        import requests
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.headers, **kwargs.get('headers', {})}
        
        try:
            response = requests.get(url, headers=headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"GET {endpoint} failed: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from {endpoint}: {str(e)}")
    
    def post(self, endpoint: str, data: dict, **kwargs) -> dict:
        """Make POST request with consistent error handling."""
        import requests
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {**self.headers, **kwargs.get('headers', {})}
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"POST {endpoint} failed: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from {endpoint}: {str(e)}")


def create_test_config() -> TestConfig:
    """Factory function to create test configuration."""
    return TestConfig()


def create_api_client() -> APIClient:
    """Factory function to create API client."""
    config = create_test_config()
    return APIClient(config)
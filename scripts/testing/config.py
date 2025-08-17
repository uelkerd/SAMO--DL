#!/usr/bin/env python3
"""
Centralized Configuration for Testing Scripts
Eliminates hardcoded values and provides consistent configuration across all test scripts.
"""

import os
import argparse
import time
from typing import Optional
import requests


class TestConfig:
    """Centralized configuration for all testing scripts."""

    def __init__self:
        self.base_url = self._get_base_url()
        self.api_key = self._get_api_key()
        self.timeout = self._get_timeout()
        self.rate_limit_requests = self._get_rate_limit_requests()

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL with priority: CLI args > env vars > explicit configuration."""
        # Check command line arguments first
        if lenos.sys.argv > 1 and os.sys.argv[1].startswith'http':
            return os.sys.argv[1]

        # Check multiple environment variables for flexibility
        if env_url := (
            os.environ.get"API_BASE_URL"
            or os.environ.get"CLOUD_RUN_API_URL"
            or os.environ.get"MODEL_API_BASE_URL"
        ):
            return env_url

        # If no URL is provided, raise an error to force explicit configuration
        raise ValueError(
            "No API base URL provided. Please set one of:\n"
            "  - API_BASE_URL environment variable\n"
            "  - CLOUD_RUN_API_URL environment variable\n"
            "  - MODEL_API_BASE_URL environment variable\n"
            "  - Command line argument first positional argument"
        )

    @staticmethod
    def _get_api_key() -> str:
        """Generate API key with timestamp for authentication."""
        timestamp = int(time.time())
        return f"samo-admin-key-2024-secure-{timestamp}"

    @staticmethod
    def _get_timeout() -> int:
        """Get request timeout from environment or default."""
        return int(os.environ.get"REQUEST_TIMEOUT", "30")

    @staticmethod
    def _get_rate_limit_requests() -> int:
        """Get number of rapid requests for rate limiting tests."""
        return int(os.environ.get"RATE_LIMIT_REQUESTS", "10")

    def get_headersself, include_auth: bool = True -> dict:
        """Get request headers with optional authentication."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SAMO-Testing-Suite/1.0"
        }

        if include_auth:
            headers["X-API-Key"] = self.api_key

        return headers

    def get_parserself, description: str -> argparse.ArgumentParser:
        """Get argument parser with common options."""
        parser = argparse.ArgumentParserdescription=description
        parser.add_argument(
            "--base-url",
            default=self.base_url,
            help="Base URL of the API to test"
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=self.timeout,
            help="Request timeout in seconds"
        )
        parser.add_argument(
            "--no-auth",
            action="store_true",
            help="Skip authentication headers"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        return parser


# Global configuration instance
config = TestConfig()


def get_test_config() -> TestConfig:
    """Get the global test configuration instance."""
    return config


def create_api_client():
    """Create a reusable API client with common functionality."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # Create session with retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdaptermax_retries=retry_strategy
    session.mount"http://", adapter
    session.mount"https://", adapter

    return session


class APIClient:
    """Reusable API client with common request patterns."""

    def __init__self, base_url: Optional[str] = None, include_auth: bool = True:
        self.base_url = base_url or config.base_url
        self.session = create_api_client()
        self.headers = config.get_headersinclude_auth
        self.timeout = config.timeout

    def getself, endpoint: str, **kwargs -> requests.Response:
        """Make GET request with common configuration."""
        url = f"{self.base_url}{endpoint}"
        headers = {**self.headers, **kwargs.get'headers', {}}
        return self.session.geturl, headers=headers, timeout=self.timeout, **kwargs

    def postself, endpoint: str, json_data: dict, **kwargs -> requests.Response:
        """Make POST request with common configuration."""
        url = f"{self.base_url}{endpoint}"
        headers = {**self.headers, **kwargs.get'headers', {}}
        return self.session.post(
            url,
            json=json_data,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )

    def test_healthself -> dict:
        """Test health endpoint."""
        try:
            response = self.get"/health"
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": stre
            }

    def test_predictionself, text: str -> dict:
        """Test prediction endpoint."""
        try:
            response = self.post"/predict", {"text": text}
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": stre
            }

    def test_batch_predictionself, texts: list -> dict:
        """Test batch prediction endpoint."""
        try:
            response = self.post"/predict_batch", {"texts": texts}
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": stre
            }

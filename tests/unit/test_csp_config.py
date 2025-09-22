#!/usr/bin/env python3
"""🧪 CSP Configuration Tests.
==========================
Tests for Content Security Policy configuration and loading.
"""

import os
import sys
import tempfile

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import unittest
from unittest.mock import patch

from security_headers import SecurityHeadersConfig, SecurityHeadersMiddleware


class TestCSPConfiguration(unittest.TestCase):
    """Test CSP configuration loading and fallback."""

    def setUp(self):
        """Set up test fixtures."""
        from flask import Flask

        self.app = Flask(__name__)
        self.config = SecurityHeadersConfig(
            enable_csp=True,
            enable_content_security_policy=True,
        )

    def test_csp_loaded_from_config_file(self):
        """Test that CSP is loaded from config file when available."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "security_headers": {
                        "headers": {
                            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'nonce-test'; style-src 'self'",
                        },
                    },
                },
                f,
            )
            config_path = f.name

        try:
            # Mock the config file path
            with patch("os.path.join", return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that CSP was loaded from config
                csp_policy = middleware._build_csp_policy()
                assert "script-src 'self' 'nonce-test'" in csp_policy
                assert "style-src 'self'" in csp_policy

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_fallback_to_secure_default(self):
        """Test that CSP falls back to secure default when config file is missing."""
        # Mock file not found
        with patch(
            "builtins.open",
            side_effect=FileNotFoundError("Config file not found"),
        ):
            middleware = SecurityHeadersMiddleware(self.app, self.config)

            # Check that secure default is used
            csp_policy = middleware._build_csp_policy()
            assert "default-src 'self'" in csp_policy
            assert "script-src 'self'" in csp_policy
            assert "style-src 'self'" in csp_policy
            assert "object-src 'none'" in csp_policy
            assert "base-uri 'self'" in csp_policy
            assert "form-action 'self'" in csp_policy

    def test_csp_fallback_on_invalid_yaml(self):
        """Test that CSP falls back to secure default when YAML is invalid."""
        # Create a temporary config file with invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            # Mock the config file path
            with patch("os.path.join", return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that secure default is used
                csp_policy = middleware._build_csp_policy()
                assert "default-src 'self'" in csp_policy
                assert "script-src 'self'" in csp_policy

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_fallback_on_missing_csp_key(self):
        """Test that CSP falls back to secure default when CSP key is missing from
        config."""
        # Create a temporary config file without CSP
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "security_headers": {
                        "headers": {
                            "X-Frame-Options": "DENY",
                        },
                    },
                },
                f,
            )
            config_path = f.name

        try:
            # Mock the config file path
            with patch("os.path.join", return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that secure default is used
                csp_policy = middleware._build_csp_policy()
                assert "default-src 'self'" in csp_policy
                assert "script-src 'self'" in csp_policy

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_policy_formatting(self):
        """Test that CSP policy is properly formatted."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Check that policy is a string
        assert isinstance(csp_policy, str)

        # Check that policy contains required directives
        directives = csp_policy.split("; ")
        assert len(directives) > 5  # Should have multiple directives

        # Check for required directives
        directive_names = [d.split(" ")[0] for d in directives]
        assert "default-src" in directive_names
        assert "script-src" in directive_names
        assert "style-src" in directive_names
        assert "object-src" in directive_names

    def test_csp_policy_security(self):
        """Test that CSP policy contains secure defaults."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Check for secure defaults
        assert "object-src 'none'" in csp_policy  # No plugins
        assert "base-uri 'self'" in csp_policy  # Restrict base URI
        assert "form-action 'self'" in csp_policy  # Restrict form submissions

        # Should NOT contain unsafe directives
        assert "'unsafe-inline'" not in csp_policy
        assert "'unsafe-eval'" not in csp_policy

    def test_csp_disabled_when_config_disabled(self):
        """Test that CSP is not added when disabled in config."""
        config = SecurityHeadersConfig(
            enable_csp=False,
            enable_content_security_policy=False,
        )

        middleware = SecurityHeadersMiddleware(self.app, config)

        # Mock response
        from flask import Response

        response = Response()

        # Add security headers
        middleware._add_security_headers(response)

        # Check that CSP header is not set
        assert "Content-Security-Policy" not in response.headers

    def test_csp_header_set_when_enabled(self):
        """Test that CSP header is set when enabled."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)

        # Mock response
        from flask import Response

        response = Response()

        # Add security headers
        middleware._add_security_headers(response)

        # Check that CSP header is set
        assert "Content-Security-Policy" in response.headers
        csp_value = response.headers["Content-Security-Policy"]
        assert isinstance(csp_value, str)
        assert len(csp_value) > 0

    def test_enhanced_csp_policy_directives(self):
        """Test that enhanced CSP policy contains all required security directives."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Define all required CSP directives with descriptions
        required_directives = [
            ("default-src 'self'", "Default source restriction"),
            ("script-src 'self'", "Script source restriction"),
            ("style-src 'self'", "Style source restriction"),
            ("object-src 'none'", "Block all plugins"),
            ("base-uri 'self'", "Restrict base URI"),
            ("form-action 'self'", "Restrict form submissions"),
            ("frame-ancestors 'none'", "Block iframe embedding"),
            ("upgrade-insecure-requests", "Force HTTPS"),
            ("block-all-mixed-content", "Block mixed content"),
            ("img-src 'self' data: https:", "Allow data URIs and HTTPS images"),
            ("font-src 'self' data:", "Allow data URI fonts"),
            ("connect-src 'self' https:", "Allow HTTPS connections"),
            ("media-src 'self' https:", "Allow HTTPS media"),
        ]

        # Test all directives in a single loop
        for directive, description in required_directives:
            assert (
                directive in csp_policy
            ), f"Missing CSP directive: {description} ({directive})"

    def test_csp_policy_production_ready(self):
        """Test that CSP policy is production-ready with comprehensive security."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Production security checks with descriptions
        production_security = [
            ("object-src 'none'", "Block all plugins"),
            ("frame-ancestors 'none'", "Block iframe embedding"),
            ("base-uri 'self'", "Restrict base URI"),
            ("form-action 'self'", "Restrict form submissions"),
            ("upgrade-insecure-requests", "Force HTTPS"),
            ("block-all-mixed-content", "Block mixed content"),
        ]

        # Test all production security features in a single loop
        for directive, description in production_security:
            assert (
                directive in csp_policy
            ), f"Production security missing: {description} ({directive})"


if __name__ == "__main__":
    unittest.main()

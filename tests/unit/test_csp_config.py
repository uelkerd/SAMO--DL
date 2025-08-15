#!/usr/bin/env python3
"""
🧪 CSP Configuration Tests
==========================
Tests for Content Security Policy configuration and loading.
"""

import sys
import os
import tempfile
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import unittest
from unittest.mock import patch

from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig

class TestCSPConfiguration(unittest.TestCase):
    """Test CSP configuration loading and fallback."""

    def setUp(self):
        """Set up test fixtures."""
        from flask import Flask
        self.app = Flask(__name__)
        self.config = SecurityHeadersConfig(
            enable_csp=True,
            enable_content_security_policy=True
        )

    def test_csp_loaded_from_config_file(self):
        """Test that CSP is loaded from config file when available."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'security_headers': {
                    'headers': {
                        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'nonce-test'; style-src 'self'"
                    }
                }
            }, f)
            config_path = f.name

        try:
            # Mock the config file path
            with patch('os.path.join', return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that CSP was loaded from config
                csp_policy = middleware._build_csp_policy()
                self.assertIn("script-src 'self' 'nonce-test'", csp_policy)
                self.assertIn("style-src 'self'", csp_policy)

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_fallback_to_secure_default(self):
        """Test that CSP falls back to secure default when config file is missing."""
        # Mock file not found
        with patch('builtins.open', side_effect=FileNotFoundError("Config file not found")):
            middleware = SecurityHeadersMiddleware(self.app, self.config)

            # Check that secure default is used
            csp_policy = middleware._build_csp_policy()
            self.assertIn("default-src 'self'", csp_policy)
            self.assertIn("script-src 'self'", csp_policy)
            self.assertIn("style-src 'self'", csp_policy)
            self.assertIn("object-src 'none'", csp_policy)
            self.assertIn("base-uri 'self'", csp_policy)
            self.assertIn("form-action 'self'", csp_policy)

    def test_csp_fallback_on_invalid_yaml(self):
        """Test that CSP falls back to secure default when YAML is invalid."""
        # Create a temporary config file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            # Mock the config file path
            with patch('os.path.join', return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that secure default is used
                csp_policy = middleware._build_csp_policy()
                self.assertIn("default-src 'self'", csp_policy)
                self.assertIn("script-src 'self'", csp_policy)

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_fallback_on_missing_csp_key(self):
        """Test that CSP falls back to secure default when CSP key is missing from config."""
        # Create a temporary config file without CSP
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'security_headers': {
                    'headers': {
                        'X-Frame-Options': 'DENY'
                    }
                }
            }, f)
            config_path = f.name

        try:
            # Mock the config file path
            with patch('os.path.join', return_value=config_path):
                middleware = SecurityHeadersMiddleware(self.app, self.config)

                # Check that secure default is used
                csp_policy = middleware._build_csp_policy()
                self.assertIn("default-src 'self'", csp_policy)
                self.assertIn("script-src 'self'", csp_policy)

        finally:
            # Clean up
            os.unlink(config_path)

    def test_csp_policy_formatting(self):
        """Test that CSP policy is properly formatted."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Check that policy is a string
        self.assertIsInstance(csp_policy, str)

        # Check that policy contains required directives
        directives = csp_policy.split('; ')
        self.assertGreater(len(directives), 5)  # Should have multiple directives

        # Check for required directives
        directive_names = [d.split(' ')[0] for d in directives]
        self.assertIn('default-src', directive_names)
        self.assertIn('script-src', directive_names)
        self.assertIn('style-src', directive_names)
        self.assertIn('object-src', directive_names)

    def test_csp_policy_security(self):
        """Test that CSP policy contains secure defaults."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)
        csp_policy = middleware._build_csp_policy()

        # Check for secure defaults
        self.assertIn("object-src 'none'", csp_policy)  # No plugins
        self.assertIn("base-uri 'self'", csp_policy)    # Restrict base URI
        self.assertIn("form-action 'self'", csp_policy) # Restrict form submissions

        # Should NOT contain unsafe directives
        self.assertNotIn("'unsafe-inline'", csp_policy)
        self.assertNotIn("'unsafe-eval'", csp_policy)

    def test_csp_disabled_when_config_disabled(self):
        """Test that CSP is not added when disabled in config."""
        config = SecurityHeadersConfig(
            enable_csp=False,
            enable_content_security_policy=False
        )

        middleware = SecurityHeadersMiddleware(self.app, config)

        # Mock response
        from flask import Response
        response = Response()

        # Add security headers
        middleware._add_security_headers(response)

        # Check that CSP header is not set
        self.assertNotIn('Content-Security-Policy', response.headers)

    def test_csp_header_set_when_enabled(self):
        """Test that CSP header is set when enabled."""
        middleware = SecurityHeadersMiddleware(self.app, self.config)

        # Mock response
        from flask import Response
        response = Response()

        # Add security headers
        middleware._add_security_headers(response)

        # Check that CSP header is set
        self.assertIn('Content-Security-Policy', response.headers)
        csp_value = response.headers['Content-Security-Policy']
        self.assertIsInstance(csp_value, str)
        self.assertGreater(len(csp_value), 0)

if __name__ == '__main__':
    unittest.main()

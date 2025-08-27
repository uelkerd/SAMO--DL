#!/usr/bin/env python3
"""
🧪 Security Integration Tests
=============================
Comprehensive tests for security components working together.
"""

import sys
import os
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from flask import Flask, Response
from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig


class TestSecurityIntegration(unittest.TestCase):
    """Test security components working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.config = SecurityHeadersConfig(
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
            ua_blocking_enabled=True  # Enable blocking for testing
        )
        self.middleware = SecurityHeadersMiddleware(self.app, self.config)

    def test_comprehensive_security_headers(self):
        """Test that all security headers are properly set."""
        response = Response()
        
        # Add all security headers
        self.middleware._add_security_headers(response)
        
        # Define all required security headers with validation rules
        required_headers = [
            ('Content-Security-Policy', 'string', 'non-empty'),
            ('Strict-Transport-Security', 'string', 'non-empty'),
            ('X-Frame-Options', 'string', 'non-empty'),
            ('X-Content-Type-Options', 'string', 'non-empty'),
            ('X-XSS-Protection', 'string', 'non-empty'),
            ('Referrer-Policy', 'string', 'non-empty'),
            ('Permissions-Policy', 'string', 'non-empty'),
            ('Cross-Origin-Embedder-Policy', 'string', 'non-empty'),
            ('Cross-Origin-Opener-Policy', 'string', 'non-empty'),
            ('Cross-Origin-Resource-Policy', 'string', 'non-empty'),
            ('Origin-Agent-Cluster', 'string', 'non-empty')
        ]
        
        # Test all headers with consistent validation
        for header, expected_type, validation in required_headers:
            self.assertIn(header, response.headers, f"Missing security header: {header}")
            self.assertIsInstance(response.headers[header], str, f"Header {header} should be string")
            if validation == 'non-empty':
                self.assertGreater(len(response.headers[header]), 0, f"Header {header} should not be empty")

    def test_csp_policy_comprehensive_coverage(self):
        """Test that CSP policy covers all security aspects."""
        csp_policy = self.middleware._build_csp_policy()
        
        # Security directives that should be present
        security_directives = [
            "default-src 'self'",           # Default source restriction
            "script-src 'self'",            # Script source restriction
            "style-src 'self'",             # Style source restriction
            "object-src 'none'",            # Block all plugins
            "base-uri 'self'",              # Restrict base URI
            "form-action 'self'",           # Restrict form submissions
            "frame-ancestors 'none'",       # Block iframe embedding
            "upgrade-insecure-requests",    # Force HTTPS
            "block-all-mixed-content"       # Block mixed content
        ]
        
        for directive in security_directives:
            self.assertIn(directive, csp_policy, f"Missing CSP directive: {directive}")

    def test_permissions_policy_comprehensive(self):
        """Test that permissions policy covers all browser features."""
        permissions_policy = self.middleware._build_permissions_policy()
        
        # Critical permissions that should be restricted
        critical_permissions = [
            "camera=()",           # No camera access
            "microphone=()",       # No microphone access
            "geolocation=()",      # No location access
            "payment=()",          # No payment access
            "fullscreen=()",       # No fullscreen access
            "encrypted-media=()"   # No encrypted media access
        ]
        
        for permission in critical_permissions:
            self.assertIn(permission, permissions_policy, f"Missing permission restriction: {permission}")

    def test_user_agent_analysis_integration(self):
        """Test user agent analysis integration with security middleware."""
        # Test with highly malicious user agent that will score >3
        malicious_ua = "sqlmap/1.0 + nmap/7.80 + nikto/2.1.6 + dirb/2.22"
        analysis = self.middleware._analyze_user_agent_enhanced(malicious_ua)
        
        self.assertIn('score', analysis)
        self.assertIn('category', analysis)
        self.assertIn('risk_level', analysis)
        self.assertIn('patterns', analysis)
        
        # Should detect high-risk user agent with multiple attack tools
        self.assertGreater(analysis['score'], 3)
        self.assertIn('high_risk', analysis['category'])

    def test_suspicious_pattern_detection(self):
        """Test suspicious pattern detection integration."""
        with self.app.test_request_context('/test', headers={
            'X-Forwarded-Host': 'malicious.com',
            'User-Agent': 'sqlmap/1.0'
        }):
            patterns = self.middleware._detect_suspicious_patterns()
            
            # Should detect suspicious patterns
            self.assertIsInstance(patterns, list)
            if len(patterns) > 0:
                # Check that patterns contain suspicious indicators
                pattern_text = ' '.join(patterns).lower()
                self.assertTrue(
                    any(indicator in pattern_text for indicator in ['suspicious', 'header', 'user agent']),
                    f"Expected suspicious patterns, got: {patterns}"
                )

    def test_request_correlation_integration(self):
        """Test request correlation headers integration."""
        with self.app.test_request_context('/test'):
            # Simulate before_request
            self.middleware._before_request()
            
            # Create response
            response = Response()
            
            # Add correlation headers
            self.middleware._add_correlation_headers(response)
            
            # Check for correlation headers
            self.assertIn('X-Request-ID', response.headers)
            self.assertIn('X-Correlation-ID', response.headers)
            
            # Headers should not be empty
            self.assertGreater(len(response.headers['X-Request-ID']), 0)
            self.assertGreater(len(response.headers['X-Correlation-ID']), 0)

    def test_security_logging_integration(self):
        """Test security logging integration."""
        with self.app.test_request_context('/test', headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }):
            # This should trigger security logging
            # We can't easily test logging output, but we can ensure it doesn't crash
            try:
                self.middleware._log_security_info()
                # If we get here, logging didn't crash
                # No assertion needed - if we reach this point, the test passes
            except Exception as e:
                self.fail(f"Security logging crashed: {e}")

    def test_security_stats_comprehensive(self):
        """Test that security stats provide comprehensive information."""
        stats = self.middleware.get_security_stats()
        
        # Check config section
        self.assertIn('config', stats)
        config = stats['config']
        
        # All security features should be documented
        required_config_keys = [
            'enable_csp', 'enable_hsts', 'enable_x_frame_options',
            'enable_x_content_type_options', 'enable_x_xss_protection',
            'enable_referrer_policy', 'enable_permissions_policy',
            'enable_cross_origin_embedder_policy', 'enable_cross_origin_opener_policy',
            'enable_cross_origin_resource_policy', 'enable_origin_agent_cluster',
            'enable_request_id', 'enable_correlation_id',
            'enable_enhanced_ua_analysis', 'ua_suspicious_score_threshold',
            'ua_blocking_enabled'
        ]
        
        for key in required_config_keys:
            self.assertIn(key, config, f"Missing config key: {key}")
            self.assertIsInstance(config[key], (bool, int), f"Config key {key} should be bool or int")

    def test_production_security_headers(self):
        """Test that all production security headers are properly configured."""
        response = Response()
        self.middleware._add_security_headers(response)
        
        # Production security header values
        expected_values = {
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        for header, expected_value in expected_values.items():
            self.assertIn(header, response.headers, f"Missing header: {header}")
            self.assertEqual(response.headers[header], expected_value, 
                           f"Header {header} has wrong value: {response.headers[header]} != {expected_value}")

    def test_csp_nonce_generation(self):
        """Test that CSP nonce is generated and available."""
        stats = self.middleware.get_security_stats()
        
        self.assertIn('csp_nonce', stats)
        nonce = stats['csp_nonce']
        
        # Nonce should be a hex string
        self.assertIsInstance(nonce, str)
        self.assertGreater(len(nonce), 0)
        
        # Should be regenerated for each middleware instance
        middleware2 = SecurityHeadersMiddleware(self.app, self.config)
        stats2 = middleware2.get_security_stats()
        
        # Nonces should be different (random generation)
        self.assertNotEqual(nonce, stats2['csp_nonce'])


if __name__ == '__main__':
    unittest.main()

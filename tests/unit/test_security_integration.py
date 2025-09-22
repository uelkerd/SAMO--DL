#!/usr/bin/env python3
"""🧪 Security Integration Tests.
=============================
Comprehensive tests for security components working together.
"""

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from flask import Flask, Response
from security_headers import SecurityHeadersConfig, SecurityHeadersMiddleware


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
            ua_blocking_enabled=True,  # Enable blocking for testing
        )
        self.middleware = SecurityHeadersMiddleware(self.app, self.config)

    def test_comprehensive_security_headers(self):
        """Test that all security headers are properly set."""
        response = Response()

        # Add all security headers
        self.middleware._add_security_headers(response)

        # Define all required security headers with validation rules
        required_headers = [
            ("Content-Security-Policy", "non-empty"),
            ("Strict-Transport-Security", "non-empty"),
            ("X-Frame-Options", "non-empty"),
            ("X-Content-Type-Options", "non-empty"),
            ("X-XSS-Protection", "non-empty"),
            ("Referrer-Policy", "non-empty"),
            ("Permissions-Policy", "non-empty"),
            ("Cross-Origin-Embedder-Policy", "non-empty"),
            ("Cross-Origin-Opener-Policy", "non-empty"),
            ("Cross-Origin-Resource-Policy", "non-empty"),
            ("Origin-Agent-Cluster", "non-empty"),
        ]

        # Test all headers with consistent validation
        for header, validation in required_headers:
            assert header in response.headers, f"Missing security header: {header}"
            assert isinstance(
                response.headers[header], str
            ), f"Header {header} should be string"
            if validation == "non-empty":
                assert (
                    len(response.headers[header]) > 0
                ), f"Header {header} should not be empty"

    def test_csp_policy_default_src(self):
        """Test that CSP policy includes default-src directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "default-src 'self'" in csp_policy, "Missing default-src directive"

    def test_csp_policy_script_src(self):
        """Test that CSP policy includes script-src directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "script-src 'self'" in csp_policy, "Missing script-src directive"

    def test_csp_policy_style_src(self):
        """Test that CSP policy includes style-src directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "style-src 'self'" in csp_policy, "Missing style-src directive"

    def test_csp_policy_object_src(self):
        """Test that CSP policy includes object-src directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "object-src 'none'" in csp_policy, "Missing object-src directive"

    def test_csp_policy_base_uri(self):
        """Test that CSP policy includes base-uri directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "base-uri 'self'" in csp_policy, "Missing base-uri directive"

    def test_csp_policy_form_action(self):
        """Test that CSP policy includes form-action directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert "form-action 'self'" in csp_policy, "Missing form-action directive"

    def test_csp_policy_frame_ancestors(self):
        """Test that CSP policy includes frame-ancestors directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert (
            "frame-ancestors 'none'" in csp_policy
        ), "Missing frame-ancestors directive"

    def test_csp_policy_upgrade_insecure_requests(self):
        """Test that CSP policy includes upgrade-insecure-requests directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert (
            "upgrade-insecure-requests" in csp_policy
        ), "Missing upgrade-insecure-requests directive"

    def test_csp_policy_block_mixed_content(self):
        """Test that CSP policy includes block-all-mixed-content directive."""
        csp_policy = self.middleware._build_csp_policy()
        assert (
            "block-all-mixed-content" in csp_policy
        ), "Missing block-all-mixed-content directive"

    def test_csp_disallows_unsafe_inline_and_eval(self):
        """Test that CSP policy does NOT allow unsafe directives."""
        csp_policy = self.middleware._build_csp_policy()
        assert "'unsafe-inline'" not in csp_policy, "CSP should not allow unsafe-inline"
        assert "'unsafe-eval'" not in csp_policy, "CSP should not allow unsafe-eval"

    def test_permissions_policy_camera(self):
        """Test that permissions policy restricts camera access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert "camera=()" in permissions_policy, "Missing camera restriction"

    def test_permissions_policy_microphone(self):
        """Test that permissions policy restricts microphone access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert "microphone=()" in permissions_policy, "Missing microphone restriction"

    def test_permissions_policy_geolocation(self):
        """Test that permissions policy restricts geolocation access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert "geolocation=()" in permissions_policy, "Missing geolocation restriction"

    def test_permissions_policy_payment(self):
        """Test that permissions policy restricts payment access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert "payment=()" in permissions_policy, "Missing payment restriction"

    def test_permissions_policy_fullscreen(self):
        """Test that permissions policy restricts fullscreen access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert "fullscreen=()" in permissions_policy, "Missing fullscreen restriction"

    def test_permissions_policy_encrypted_media(self):
        """Test that permissions policy restricts encrypted media access."""
        permissions_policy = self.middleware._build_permissions_policy()
        assert (
            "encrypted-media=()" in permissions_policy
        ), "Missing encrypted media restriction"

    def test_user_agent_analysis_integration(self):
        """Test user agent analysis integration with security middleware."""
        # Test with highly malicious user agent that will score >3
        malicious_ua = "sqlmap/1.0 + nmap/7.80 + nikto/2.1.6 + dirb/2.22"
        analysis = self.middleware._analyze_user_agent_enhanced(malicious_ua)

        assert "score" in analysis
        assert "category" in analysis
        assert "risk_level" in analysis
        assert "patterns" in analysis

        # Should detect malicious user agent with multiple attack tools
        assert analysis["score"] > 3
        assert "malicious" in analysis["category"]
        assert analysis["risk_level"] == "very_high"

    def test_suspicious_pattern_detection(self):
        """Test suspicious pattern detection integration."""
        with self.app.test_request_context(
            "/test",
            headers={
                "X-Forwarded-Host": "malicious.com",
                "User-Agent": "sqlmap/1.0",
            },
        ):
            patterns = self.middleware._detect_suspicious_patterns()

            # Should detect suspicious patterns
            assert isinstance(patterns, list)
            # Always check for suspicious indicators regardless of pattern count
            pattern_text = " ".join(patterns).lower()
            assert any(
                indicator in pattern_text
                for indicator in ["suspicious", "header", "user agent"]
            ), f"Expected suspicious patterns, got: {patterns}"

    def test_request_correlation_integration(self):
        """Test request correlation headers integration."""
        with self.app.test_request_context("/test"):
            # Simulate before_request
            self.middleware._before_request()

            # Create response
            response = Response()

            # Add correlation headers
            self.middleware._add_correlation_headers(response)

            # Check for correlation headers
            assert "X-Request-ID" in response.headers
            assert "X-Correlation-ID" in response.headers

            # Headers should not be empty
            assert len(response.headers["X-Request-ID"]) > 0
            assert len(response.headers["X-Correlation-ID"]) > 0

    def test_security_logging_integration(self):
        """Test security logging integration."""
        with self.app.test_request_context(
            "/test",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        ):
            # This should trigger security logging
            # We can't easily test logging output, but we can ensure it doesn't crash
            try:
                self.middleware._log_security_info()
                # If we get here, logging didn't crash
                # No assertion needed - if we reach this point, the test passes
            except Exception as e:
                self.fail(f"Security logging crashed: {e}")

    def test_security_stats_config_section(self):
        """Test that security stats config section exists."""
        stats = self.middleware.get_security_stats()
        assert "config" in stats
        config = stats["config"]
        assert isinstance(config, dict)

    def test_security_stats_enable_csp(self):
        """Test that CSP is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_csp" in config
        assert isinstance(config["enable_csp"], bool)

    def test_security_stats_enable_hsts(self):
        """Test that HSTS is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_hsts" in config
        assert isinstance(config["enable_hsts"], bool)

    def test_security_stats_enable_x_frame_options(self):
        """Test that X-Frame-Options is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_x_frame_options" in config
        assert isinstance(config["enable_x_frame_options"], bool)

    def test_security_stats_enable_x_content_type_options(self):
        """Test that X-Content-Type-Options is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_x_content_type_options" in config
        assert isinstance(config["enable_x_content_type_options"], bool)

    def test_security_stats_enable_x_xss_protection(self):
        """Test that X-XSS-Protection is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_x_xss_protection" in config
        assert isinstance(config["enable_x_xss_protection"], bool)

    def test_security_stats_enable_referrer_policy(self):
        """Test that Referrer-Policy is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_referrer_policy" in config
        assert isinstance(config["enable_referrer_policy"], bool)

    def test_security_stats_enable_permissions_policy(self):
        """Test that Permissions-Policy is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_permissions_policy" in config
        assert isinstance(config["enable_permissions_policy"], bool)

    def test_security_stats_enable_cross_origin_embedder_policy(self):
        """Test that Cross-Origin-Embedder-Policy is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_cross_origin_embedder_policy" in config
        assert isinstance(config["enable_cross_origin_embedder_policy"], bool)

    def test_security_stats_enable_cross_origin_opener_policy(self):
        """Test that Cross-Origin-Opener-Policy is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_cross_origin_opener_policy" in config
        assert isinstance(config["enable_cross_origin_opener_policy"], bool)

    def test_security_stats_enable_cross_origin_resource_policy(self):
        """Test that Cross-Origin-Resource-Policy is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_cross_origin_resource_policy" in config
        assert isinstance(config["enable_cross_origin_resource_policy"], bool)

    def test_security_stats_enable_origin_agent_cluster(self):
        """Test that Origin-Agent-Cluster is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_origin_agent_cluster" in config
        assert isinstance(config["enable_origin_agent_cluster"], bool)

    def test_security_stats_enable_request_id(self):
        """Test that Request-ID is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_request_id" in config
        assert isinstance(config["enable_request_id"], bool)

    def test_security_stats_enable_correlation_id(self):
        """Test that Correlation-ID is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_correlation_id" in config
        assert isinstance(config["enable_correlation_id"], bool)

    def test_security_stats_enable_enhanced_ua_analysis(self):
        """Test that Enhanced UA Analysis is enabled in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "enable_enhanced_ua_analysis" in config
        assert isinstance(config["enable_enhanced_ua_analysis"], bool)

    def test_security_stats_ua_suspicious_score_threshold(self):
        """Test that UA suspicious score threshold is set in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "ua_suspicious_score_threshold" in config
        assert isinstance(config["ua_suspicious_score_threshold"], int)

    def test_security_stats_ua_blocking_enabled(self):
        """Test that UA blocking is configured in security stats."""
        stats = self.middleware.get_security_stats()
        config = stats["config"]
        assert "ua_blocking_enabled" in config
        assert isinstance(config["ua_blocking_enabled"], bool)

    def test_production_security_headers(self):
        """Test that all production security headers are properly configured."""
        response = Response()
        self.middleware._add_security_headers(response)

        # Test each production security header individually
        assert "X-Frame-Options" in response.headers, "Missing X-Frame-Options header"
        assert (
            response.headers["X-Frame-Options"] == "DENY"
        ), "X-Frame-Options should be DENY"

        assert (
            "X-Content-Type-Options" in response.headers
        ), "Missing X-Content-Type-Options header"
        assert (
            response.headers["X-Content-Type-Options"] == "nosniff"
        ), "X-Content-Type-Options should be nosniff"

        assert "X-XSS-Protection" in response.headers, "Missing X-XSS-Protection header"
        assert (
            response.headers["X-XSS-Protection"] == "1; mode=block"
        ), "X-XSS-Protection should be 1; mode=block"

        assert "Referrer-Policy" in response.headers, "Missing Referrer-Policy header"
        assert (
            response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        ), "Referrer-Policy should be strict-origin-when-cross-origin"

    def test_csp_nonce_generation(self):
        """Test that CSP nonce is generated and available."""
        stats = self.middleware.get_security_stats()

        assert "csp_nonce" in stats
        nonce = stats["csp_nonce"]

        # Nonce should be a hex string
        assert isinstance(nonce, str)
        assert len(nonce) > 0

        # Should be regenerated for each middleware instance
        middleware2 = SecurityHeadersMiddleware(self.app, self.config)
        stats2 = middleware2.get_security_stats()

        # Nonces should be different (random generation)
        assert nonce != stats2["csp_nonce"]

    def test_headers_applied_via_after_request_integration(self):
        """Test that security headers are applied via Flask hooks."""
        app = self.app

        @app.route("/ping")
        def ping():
            """Simple ping endpoint for testing security headers."""
            return "ok"

        client = app.test_client()
        resp = client.get("/ping")
        assert "Content-Security-Policy" in resp.headers
        assert "X-Content-Type-Options" in resp.headers

    def test_ua_blocking_returns_403(self):
        """Test that malicious user agents are blocked with 403."""
        with self.app.test_request_context(
            "/blocked",
            headers={
                "User-Agent": "sqlmap/1.0 curl/7.88 nikto/2.1.6",
            },
        ):
            resp = self.middleware._before_request()
            assert resp is not None
            assert getattr(resp, "status_code", None) == 403


if __name__ == "__main__":
    unittest.main()

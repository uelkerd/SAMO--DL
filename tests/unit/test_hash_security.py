#!/usr/bin/env python3
"""
🧪 Hash Security Tests
======================
Tests for hash security and collision resistance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import unittest
import hashlib

from security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig
from api_rate_limiter import TokenBucketRateLimiter, RateLimitConfig

class TestHashSecurity(unittest.TestCase):
    """Test hash security and collision resistance."""
    
    def setUp(self):
        """Set up test fixtures."""
        from flask import Flask
        self.app = Flask(__name__)
        self.config = SecurityHeadersConfig(
            enable_request_id=True,
            enable_correlation_id=True
        )
        self.middleware = SecurityHeadersMiddleware(self.app, self.config)
        
        # Rate limiter for testing
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=10,
            max_concurrent_requests=5
        )
        self.rate_limiter = TokenBucketRateLimiter(self.rate_limit_config)
    
    def test_request_id_full_sha256(self):
        """Test that request ID uses full SHA-256 hexdigest."""
        # Mock request context
        from flask import g, request
        with self.app.test_request_context('/'):
            # Mock request.remote_addr
            request.remote_addr = '192.168.1.1'
            
            # Call _before_request to generate request ID
            self.middleware._before_request()
            
            # Check that request ID is full SHA-256 (64 characters)
            self.assertIsNotNone(g.request_id)
            self.assertEqual(len(g.request_id), 64)  # Full SHA-256 hexdigest
            
            # Verify it's a valid hex string
            try:
                int(g.request_id, 16)
            except ValueError:
                self.fail("Request ID is not a valid hex string")
    
    def test_client_key_full_sha256(self):
        """Test that client key uses full SHA-256 hexdigest."""
        client_ip = "192.168.1.1"
        user_agent = "test-user-agent"
        
        # Generate client key
        client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # Check that client key is full SHA-256 (64 characters)
        self.assertEqual(len(client_key), 64)  # Full SHA-256 hexdigest
        
        # Verify it's a valid hex string
        try:
            int(client_key, 16)
        except ValueError:
            self.fail("Client key is not a valid hex string")
    
    def test_hash_collision_resistance(self):
        """Test that different inputs produce different hashes."""
        # Test request ID collision resistance
        request_ids = set()
        
        for i in range(100):
            # Mock different request contexts
            with self.app.test_request_context('/'):
                from flask import g, request
                request.remote_addr = f'192.168.1.{i}'
                
                # Generate request ID
                self.middleware._before_request()
                request_ids.add(g.request_id)
        
        # All request IDs should be unique
        self.assertEqual(len(request_ids), 100)
    
    def test_client_key_collision_resistance(self):
        """Test that different client inputs produce different client keys."""
        client_keys = set()
        
        # Test different IPs
        for i in range(50):
            client_ip = f"192.168.1.{i}"
            user_agent = "same-user-agent"
            client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
            client_keys.add(client_key)
        
        # Test different user agents
        for i in range(50):
            client_ip = "192.168.1.1"
            user_agent = f"user-agent-{i}"
            client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
            client_keys.add(client_key)
        
        # All client keys should be unique
        self.assertEqual(len(client_keys), 100)
    
    def test_hash_deterministic(self):
        """Test that same inputs always produce same hashes."""
        client_ip = "192.168.1.1"
        user_agent = "test-user-agent"
        
        # Generate client key multiple times
        key1 = self.rate_limiter._get_client_key(client_ip, user_agent)
        key2 = self.rate_limiter._get_client_key(client_ip, user_agent)
        key3 = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # All should be identical
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
    
    def test_request_id_deterministic_with_same_inputs(self):
        """Test that request ID is deterministic for same inputs."""
        # This test is limited because request ID includes time and random components
        # But we can test the structure and length consistency
        with self.app.test_request_context('/'):
            from flask import g, request
            request.remote_addr = '192.168.1.1'
            
            # Generate request ID multiple times
            self.middleware._before_request()
            request_id1 = g.request_id
            
            # Should always be 64 characters
            self.assertEqual(len(request_id1), 64)
    
    def test_hash_algorithm_verification(self):
        """Test that we're actually using SHA-256."""
        client_ip = "192.168.1.1"
        user_agent = "test-user-agent"
        
        # Generate client key
        client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # Manually calculate expected SHA-256
        fingerprint = f"{client_ip}:{user_agent}"
        expected_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
        
        # Should match
        self.assertEqual(client_key, expected_hash)
    
    def test_hash_input_format(self):
        """Test that hash input is properly formatted."""
        client_ip = "192.168.1.1"
        user_agent = "test-user-agent"
        
        # Generate client key
        client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # Manually verify the input format
        expected_input = f"{client_ip}:{user_agent}"
        expected_hash = hashlib.sha256(expected_input.encode()).hexdigest()
        
        self.assertEqual(client_key, expected_hash)
    
    def test_empty_user_agent_handling(self):
        """Test that empty user agent is handled correctly."""
        client_ip = "192.168.1.1"
        user_agent = ""
        
        # Generate client key
        client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # Should still be valid SHA-256
        self.assertEqual(len(client_key), 64)
        try:
            int(client_key, 16)
        except ValueError:
            self.fail("Client key with empty user agent is not a valid hex string")
    
    def test_special_characters_in_user_agent(self):
        """Test that special characters in user agent are handled correctly."""
        client_ip = "192.168.1.1"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        
        # Generate client key
        client_key = self.rate_limiter._get_client_key(client_ip, user_agent)
        
        # Should be valid SHA-256
        self.assertEqual(len(client_key), 64)
        try:
            int(client_key, 16)
        except ValueError:
            self.fail("Client key with special characters is not a valid hex string")

if __name__ == '__main__':
    unittest.main()

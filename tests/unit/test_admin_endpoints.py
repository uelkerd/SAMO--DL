#!/usr/bin/env python3
"""
🧪 Admin Endpoint Security Tests
================================
Tests for admin endpoint protection and authentication.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname__file__, '..', '..', 'deployment'))

import unittest
import json

# Import the secure API server with error handling
try:
    from secure_api_server import app
    MODEL_AVAILABLE = True
except OSError, ImportError as e:
    printf"Warning: Could not import secure_api_server due to missing model: {e}"
    MODEL_AVAILABLE = False
    app = None

class TestAdminEndpointProtectionunittest.TestCase:
    """Test admin endpoint protection."""
    
    @classmethod
    def setUpClasscls:
        """Set up test class."""
        if not MODEL_AVAILABLE:
            raise unittest.SkipTest"Model not available, skipping admin endpoint tests"
    
    def setUpself:
        """Set up test fixtures."""
        if not MODEL_AVAILABLE:
            self.skipTest"Model not available"
        
        self.app = app.test_client()
        self.app.testing = True
        
        # Set admin API key for testing
        os.environ['ADMIN_API_KEY'] = 'test-admin-key-123'
    
    def tearDownself:
        """Clean up after tests."""
        if 'ADMIN_API_KEY' in os.environ:
            del os.environ['ADMIN_API_KEY']
    
    def test_blacklist_endpoint_no_authself:
        """Test that blacklist endpoint requires admin API key."""
        response = self.app.post('/security/blacklist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json')
        self.assertEqualresponse.status_code, 401
        self.assertIn('Unauthorized', response.get_json()['error'])
    
    def test_blacklist_endpoint_wrong_authself:
        """Test that blacklist endpoint rejects wrong API key."""
        response = self.app.post('/security/blacklist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'wrong-key'})
        self.assertEqualresponse.status_code, 401
        self.assertIn('Unauthorized', response.get_json()['error'])
    
    def test_blacklist_endpoint_correct_authself:
        """Test that blacklist endpoint accepts correct API key."""
        response = self.app.post('/security/blacklist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'test-admin-key-123'})
        self.assertEqualresponse.status_code, 200
        self.assertIn('Added 192.168.1.100 to blacklist', response.get_json()['message'])
    
    def test_whitelist_endpoint_no_authself:
        """Test that whitelist endpoint requires admin API key."""
        response = self.app.post('/security/whitelist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json')
        self.assertEqualresponse.status_code, 401
        self.assertIn('Unauthorized', response.get_json()['error'])
    
    def test_whitelist_endpoint_wrong_authself:
        """Test that whitelist endpoint rejects wrong API key."""
        response = self.app.post('/security/whitelist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'wrong-key'})
        self.assertEqualresponse.status_code, 401
        self.assertIn('Unauthorized', response.get_json()['error'])
    
    def test_whitelist_endpoint_correct_authself:
        """Test that whitelist endpoint accepts correct API key."""
        response = self.app.post('/security/whitelist',
                               data=json.dumps{'ip': '192.168.1.100'},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'test-admin-key-123'})
        self.assertEqualresponse.status_code, 200
        self.assertIn('Added 192.168.1.100 to whitelist', response.get_json()['message'])
    
    def test_admin_endpoints_missing_ipself:
        """Test that admin endpoints require IP address."""
        # Test blacklist
        response = self.app.post('/security/blacklist',
                               data=json.dumps{},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'test-admin-key-123'})
        self.assertEqualresponse.status_code, 400
        self.assertIn('IP address required', response.get_json()['error'])
        
        # Test whitelist
        response = self.app.post('/security/whitelist',
                               data=json.dumps{},
                               content_type='application/json',
                               headers={'X-Admin-API-Key': 'test-admin-key-123'})
        self.assertEqualresponse.status_code, 400
        self.assertIn('IP address required', response.get_json()['error'])

if __name__ == '__main__':
    unittest.main() 
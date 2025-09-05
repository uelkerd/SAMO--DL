#!/usr/bin/env python3
"""
🧪 API Routing Tests
====================
Tests for Flask-RESTX routing fixes and endpoint functionality.
"""

import sys
import os
import unittest
import json
from unittest.mock import patch

# Add the deployment/cloud-run directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deployment', 'cloud-run'))

class TestAPIRouting(unittest.TestCase):
    """Test API routing and endpoint functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading functions to avoid dependency issues
        with patch('secure_api_server.ensure_model_loaded', return_value=True), \
             patch('secure_api_server.predict_emotions', return_value={
                 'text': 'test text',
                 'emotions': [{'emotion': 'happy', 'confidence': 0.9}],
                 'confidence': 0.9,
                 'request_id': 'test-123',
                 'timestamp': 1234567890
             }), \
             patch('secure_api_server.get_model_status', return_value={
                 'model_loaded': True,
                 'model_path': '/test/path',
                 'model_size': '100MB'
             }):
            try:
                from secure_api_server import app
                self.app = app.test_client()
                self.app.testing = True
                self.api_available = True
            except (ImportError, OSError) as e:
                print(f"Warning: Could not import secure_api_server: {e}")
                self.api_available = False
                self.app = None

        # Set required environment variables
        os.environ.setdefault('ADMIN_API_KEY', 'test-admin-key-123')
        os.environ.setdefault('MAX_INPUT_LENGTH', '512')
        os.environ.setdefault('RATE_LIMIT_PER_MINUTE', '100')

    @classmethod
    def is_api_available(cls):
        """Check if API is available for testing."""
        # This is a simplified check - in practice, we'd need to check the actual instance
        # For now, we'll assume API is available if the import succeeded
        try:
            from secure_api_server import app
            return True
        except (ImportError, OSError):
            return False

    def tearDown(self):
        """Clean up after tests."""
        # Clean up environment variables
        for key in ['ADMIN_API_KEY', 'MAX_INPUT_LENGTH', 'RATE_LIMIT_PER_MINUTE']:
            if key in os.environ:
                del os.environ[key]

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_root_endpoint(self):
        """Test that root endpoint is accessible and returns correct response."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn('service', data)
        self.assertIn('status', data)
        self.assertIn('version', data)
        self.assertEqual(data['service'], 'SAMO Emotion Detection API')
        self.assertEqual(data['status'], 'operational')

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_health_endpoint(self):
        """Test health endpoint returns correct status."""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertIn('timestamp', data)

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_endpoint_no_auth(self):
        """Test predict endpoint requires API key."""
        response = self.app.post('/api/predict',
                               data=json.dumps({'text': 'I am happy'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 401)

        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Unauthorized', data['error'])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_endpoint_with_auth(self):
        """Test predict endpoint works with valid API key."""
        response = self.app.post('/api/predict',
                                data=json.dumps({'text': 'I am happy'}),
                                content_type='application/json',
                                headers={'X-API-Key': 'test-admin-key-123'})

        # Should succeed (200) or be rate limited (429), but not auth error (401)
        self.assertIn(response.status_code, [200, 429])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_batch_endpoint_no_auth(self):
        """Test predict_batch endpoint requires API key."""
        response = self.app.post('/api/predict_batch',
                               data=json.dumps({'texts': ['I am happy', 'I am sad']}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 401)

        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Unauthorized', data['error'])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_batch_endpoint_with_auth(self):
        """Test predict_batch endpoint works with valid API key."""
        response = self.app.post('/api/predict_batch',
                                data=json.dumps({'texts': ['I am happy', 'I am sad']}),
                                content_type='application/json',
                                headers={'X-API-Key': 'test-admin-key-123'})

        # Should succeed (200) or be rate limited (429), but not auth error (401)
        self.assertIn(response.status_code, [200, 429])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_emotions_endpoint(self):
        """Test emotions endpoint returns supported emotions."""
        response = self.app.get('/api/emotions')
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn('emotions', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['emotions'], list)
        self.assertGreater(data['count'], 0)

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_admin_model_status_no_auth(self):
        """Test admin model status endpoint requires API key."""
        response = self.app.get('/admin/model_status')
        self.assertEqual(response.status_code, 401)

        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Unauthorized', data['error'])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_admin_model_status_with_auth(self):
        """Test admin model status endpoint works with valid API key."""
        response = self.app.get('/admin/model_status',
                              headers={'X-API-Key': 'test-admin-key-123'})

        # Should succeed (200) or be rate limited (429), but not auth error (401)
        self.assertIn(response.status_code, [200, 429])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_endpoint_missing_text(self):
        """Test predict endpoint handles missing text field."""
        response = self.app.post('/api/predict',
                                data=json.dumps({}),
                                content_type='application/json',
                                headers={'X-API-Key': 'test-admin-key-123'})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('Missing text field', data['error'])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_predict_endpoint_invalid_text(self):
        """Test predict endpoint handles invalid text input."""
        response = self.app.post('/api/predict',
                               data=json.dumps({'text': ''}),
                               content_type='application/json',
                               headers={'X-API-Key': 'test-admin-key-123'})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertIn('error', data)
        self.assertIn('non-empty string', data['error'])

    @unittest.skipUnless(TestAPIRouting.is_api_available, "API not available")
    def test_namespace_routing_no_double_slashes(self):
        """Test that namespace routes don't have double slashes."""
        # Test that /api/health works (not //api/health)
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)

        # Test that /admin/model_status works (not //admin/model_status)
        response = self.app.get('/admin/model_status',
                              headers={'X-API-Key': 'test-admin-key-123'})
        self.assertIn(response.status_code, [200, 401, 429])  # 401 is expected without auth

if __name__ == '__main__':
    unittest.main()
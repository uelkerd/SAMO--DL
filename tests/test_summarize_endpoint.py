import unittest
import json
from unittest.mock import patch
from flask import Flask
from src.summarize_endpoint import summarize_bp, SummarizeEndpoint

class TestSummarizeEndpoint(unittest.TestCase):
    """Test cases for text summarization endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(summarize_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
    def test_summarize_endpoint_health(self):
        """Test summarize endpoint health check."""
        response = self.client.get('/api/summarize/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['endpoint'], 'summarize')
    
    def test_summarize_valid_request(self):
        """Test summarization with valid request."""
        with patch.object(SummarizeEndpoint, 'load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/summarize/', 
                                      json={'text': 'This is a long text that needs to be summarized for testing purposes.', 
                                            'max_length': 50, 'min_length': 20})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('summary', data)
            self.assertIn('compression_ratio', data)
    
    def test_summarize_missing_text(self):
        """Test summarization with missing text."""
        response = self.client.post('/api/summarize/', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_summarize_short_text(self):
        """Test summarization with text too short."""
        response = self.client.post('/api/summarize/', 
                                  json={'text': 'Hi', 'max_length': 50})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_summarize_invalid_parameters(self):
        """Test summarization with invalid parameters."""
        response = self.client.post('/api/summarize/', 
                                  json={'text': 'This is a long text for testing.', 
                                        'max_length': 10, 'min_length': 20})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_summarize_invalid_temperature(self):
        """Test summarization with invalid temperature."""
        response = self.client.post('/api/summarize/', 
                                  json={'text': 'This is a long text for testing.', 
                                        'temperature': 3.0})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()

import unittest
import json
from unittest.mock import patch
from flask import Flask
from src.emotion_endpoint import emotion_bp, EmotionEndpoint

class TestEmotionEndpoint(unittest.TestCase):
    """Test cases for emotion analysis endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(emotion_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
    def test_emotion_endpoint_health(self):
        """Test emotion endpoint health check."""
        response = self.client.get('/api/analyze/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['endpoint'], 'emotion')
    
    def test_emotion_analysis_valid_request(self):
        """Test emotion analysis with valid request."""
        with patch.object(EmotionEndpoint, 'load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/analyze/journal', 
                                      json={'text': 'I feel happy today', 'generate_summary': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('confidence_scores', data)
    
    def test_emotion_analysis_missing_text(self):
        """Test emotion analysis with missing text."""
        response = self.client.post('/api/analyze/journal', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_emotion_analysis_short_text(self):
        """Test emotion analysis with text too short."""
        response = self.client.post('/api/analyze/journal', 
                                  json={'text': 'Hi', 'generate_summary': True})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_emotion_analysis_invalid_json(self):
        """Test emotion analysis with invalid JSON."""
        response = self.client.post('/api/analyze/journal', 
                                  data='invalid json',
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()

import unittest
import json
import base64
from unittest.mock import patch, MagicMock
from flask import Flask
from src.emotion_endpoint import emotion_bp
from src.summarize_endpoint import summarize_bp
from src.transcribe_endpoint import transcribe_bp
from src.complete_analysis_endpoint import complete_analysis_bp
from src.health_endpoints import health_bp

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(emotion_bp)
        self.app.register_blueprint(summarize_bp)
        self.app.register_blueprint(transcribe_bp)
        self.app.register_blueprint(complete_analysis_bp)
        self.app.register_blueprint(health_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
        # Create mock audio data
        self.mock_audio_data = base64.b64encode(b"mock audio data").decode('utf-8')
        
    def test_emotion_analysis_integration(self):
        """Test emotion analysis endpoint integration."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/analyze/journal', 
                                      json={'text': 'I feel happy and content today.', 'generate_summary': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('confidence_scores', data)
            self.assertIn('summary', data)
    
    def test_summarize_integration(self):
        """Test text summarization endpoint integration."""
        with patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/summarize/', 
                                      json={'text': 'This is a long text that needs to be summarized for testing purposes.', 
                                            'max_length': 50})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('summary', data)
            self.assertIn('compression_ratio', data)
    
    def test_transcribe_integration(self):
        """Test audio transcription endpoint integration."""
        with patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/transcribe/', 
                                      json={'audio_data': self.mock_audio_data, 
                                            'audio_format': 'wav', 'language': 'en'})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('text', data)
            self.assertIn('confidence', data)
    
    def test_complete_analysis_integration(self):
        """Test complete analysis endpoint integration."""
        with patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy today.', 
                                            'include_summary': True, 'include_emotion': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('summary', data)
            self.assertIn('processing_time', data)
    
    def test_health_check_integration(self):
        """Test health check endpoint integration."""
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('uptime', data)
    
    def test_endpoint_consistency(self):
        """Test that all endpoints return consistent response formats."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            mock_emotion.return_value = None
            mock_summarize.return_value = None
            mock_transcribe.return_value = None
            mock_complete.return_value = None
            
            # Test all endpoints
            endpoints = [
                ('/api/analyze/journal', {'text': 'I feel happy today.', 'generate_summary': True}),
                ('/api/summarize/', {'text': 'This is a long text for testing.', 'max_length': 50}),
                ('/api/transcribe/', {'audio_data': self.mock_audio_data, 'language': 'en'}),
                ('/api/complete-analysis/', {'text': 'I feel happy today.', 'include_summary': True, 'include_emotion': True})
            ]
            
            for endpoint, data in endpoints:
                response = self.client.post(endpoint, json=data)
                self.assertEqual(response.status_code, 200)
                response_data = json.loads(response.data)
                self.assertIsInstance(response_data, dict)
    
    def test_error_handling_consistency(self):
        """Test that all endpoints handle errors consistently."""
        # Test missing data
        endpoints = [
            '/api/analyze/journal',
            '/api/summarize/',
            '/api/transcribe/',
            '/api/complete-analysis/'
        ]
        
        for endpoint in endpoints:
            response = self.client.post(endpoint, json={})
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)
    
    def test_health_endpoints_consistency(self):
        """Test that all health endpoints return consistent formats."""
        health_endpoints = [
            '/api/analyze/health',
            '/api/summarize/health',
            '/api/transcribe/health',
            '/api/complete-analysis/health',
            '/api/health/'
        ]
        
        for endpoint in health_endpoints:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('status', data)

if __name__ == '__main__':
    unittest.main()

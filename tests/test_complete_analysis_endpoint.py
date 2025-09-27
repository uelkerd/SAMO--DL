import unittest
import json
import base64
from unittest.mock import patch, MagicMock
from flask import Flask
from src.complete_analysis_endpoint import complete_analysis_bp, CompleteAnalysisEndpoint

class TestCompleteAnalysisEndpoint(unittest.TestCase):
    """Test cases for complete analysis endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(complete_analysis_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
        # Create mock audio data
        self.mock_audio_data = base64.b64encode(b"mock audio data").decode('utf-8')
        
    def test_complete_analysis_endpoint_health(self):
        """Test complete analysis endpoint health check."""
        response = self.client.get('/api/complete-analysis/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['endpoint'], 'complete_analysis')
    
    def test_complete_analysis_text_only(self):
        """Test complete analysis with text only."""
        with patch.object(CompleteAnalysisEndpoint, 'load_models') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy and content today.', 
                                            'include_summary': True, 'include_emotion': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('summary', data)
    
    def test_complete_analysis_audio_only(self):
        """Test complete analysis with audio only."""
        with patch.object(CompleteAnalysisEndpoint, 'load_models') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/complete-analysis/', 
                                      json={'audio_data': self.mock_audio_data, 
                                            'include_transcription': True, 'include_emotion': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('transcription', data)
            self.assertIn('emotions', data)
    
    def test_complete_analysis_text_and_audio(self):
        """Test complete analysis with both text and audio."""
        with patch.object(CompleteAnalysisEndpoint, 'load_models') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy today.', 
                                            'audio_data': self.mock_audio_data,
                                            'include_summary': True, 'include_emotion': True, 'include_transcription': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('summary', data)
            self.assertIn('transcription', data)
    
    def test_complete_analysis_missing_inputs(self):
        """Test complete analysis with no text or audio."""
        response = self.client.post('/api/complete-analysis/', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_complete_analysis_short_text(self):
        """Test complete analysis with text too short."""
        response = self.client.post('/api/complete-analysis/', 
                                  json={'text': 'Hi', 'include_emotion': True})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_complete_analysis_invalid_audio(self):
        """Test complete analysis with invalid audio data."""
        response = self.client.post('/api/complete-analysis/', 
                                  json={'audio_data': 'invalid base64', 'include_transcription': True})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_complete_analysis_large_audio(self):
        """Test complete analysis with audio file too large."""
        large_audio_data = base64.b64encode(b"x" * (26 * 1024 * 1024)).decode('utf-8')  # 26MB
        response = self.client.post('/api/complete-analysis/', 
                                  json={'audio_data': large_audio_data, 'include_transcription': True})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()

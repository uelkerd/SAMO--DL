import unittest
import json
import base64
from unittest.mock import patch
from flask import Flask
from src.transcribe_endpoint import transcribe_bp, TranscribeEndpoint

class TestTranscribeEndpoint(unittest.TestCase):
    """Test cases for audio transcription endpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(transcribe_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
        # Create mock audio data
        self.mock_audio_data = base64.b64encode(b"mock audio data").decode('utf-8')
        
    def test_transcribe_endpoint_health(self):
        """Test transcribe endpoint health check."""
        response = self.client.get('/api/transcribe/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['endpoint'], 'transcribe')
    
    def test_transcribe_valid_request(self):
        """Test transcription with valid request."""
        with patch.object(TranscribeEndpoint, 'load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/transcribe/', 
                                      json={'audio_data': self.mock_audio_data, 
                                            'audio_format': 'wav', 'language': 'en'})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('text', data)
            self.assertIn('confidence', data)
    
    def test_transcribe_missing_audio_data(self):
        """Test transcription with missing audio data."""
        response = self.client.post('/api/transcribe/', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_transcribe_invalid_audio_data(self):
        """Test transcription with invalid audio data."""
        response = self.client.post('/api/transcribe/', 
                                  json={'audio_data': 'invalid base64', 'language': 'en'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_transcribe_invalid_language(self):
        """Test transcription with invalid language."""
        response = self.client.post('/api/transcribe/', 
                                  json={'audio_data': self.mock_audio_data, 'language': 'invalid'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_transcribe_invalid_task(self):
        """Test transcription with invalid task."""
        response = self.client.post('/api/transcribe/', 
                                  json={'audio_data': self.mock_audio_data, 'task': 'invalid'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_transcribe_large_audio_file(self):
        """Test transcription with audio file too large."""
        large_audio_data = base64.b64encode(b"x" * (26 * 1024 * 1024)).decode('utf-8')  # 26MB
        response = self.client.post('/api/transcribe/', 
                                  json={'audio_data': large_audio_data, 'language': 'en'})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()

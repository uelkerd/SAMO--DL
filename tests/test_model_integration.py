import unittest
import json
import base64
from unittest.mock import patch, MagicMock
from flask import Flask
from src.emotion_endpoint import emotion_bp
from src.summarize_endpoint import summarize_bp
from src.transcribe_endpoint import transcribe_bp
from src.complete_analysis_endpoint import complete_analysis_bp

class TestModelIntegration(unittest.TestCase):
    """Integration tests for model interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.register_blueprint(emotion_bp)
        self.app.register_blueprint(summarize_bp)
        self.app.register_blueprint(transcribe_bp)
        self.app.register_blueprint(complete_analysis_bp)
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        
        # Create mock audio data
        self.mock_audio_data = base64.b64encode(b"mock audio data").decode('utf-8')
        
    def test_emotion_model_integration(self):
        """Test emotion model integration with endpoint."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/analyze/journal', 
                                      json={'text': 'I feel happy and content today.', 'generate_summary': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('confidence_scores', data)
            self.assertIn('summary', data)
            self.assertIn('processing_time', data)
            self.assertIn('model_used', data)
    
    def test_summarize_model_integration(self):
        """Test summarization model integration with endpoint."""
        with patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/summarize/', 
                                      json={'text': 'This is a long text that needs to be summarized for testing purposes.', 
                                            'max_length': 50, 'min_length': 20})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('summary', data)
            self.assertIn('compression_ratio', data)
            self.assertIn('processing_time', data)
            self.assertIn('model_used', data)
    
    def test_transcribe_model_integration(self):
        """Test transcription model integration with endpoint."""
        with patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_load:
            mock_load.return_value = None
            response = self.client.post('/api/transcribe/', 
                                      json={'audio_data': self.mock_audio_data, 
                                            'audio_format': 'wav', 'language': 'en'})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('text', data)
            self.assertIn('confidence', data)
            self.assertIn('processing_time', data)
            self.assertIn('model_used', data)
    
    def test_complete_analysis_model_integration(self):
        """Test complete analysis model integration with endpoint."""
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
            self.assertIn('models_used', data)
    
    def test_model_loading_consistency(self):
        """Test that all models load consistently."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            mock_emotion.return_value = None
            mock_summarize.return_value = None
            mock_transcribe.return_value = None
            mock_complete.return_value = None
            
            # Test all endpoints to trigger model loading
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
                self.assertIn('model_used', response_data)
    
    def test_model_error_handling(self):
        """Test model error handling across endpoints."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            # Mock model loading failures
            mock_emotion.side_effect = Exception("Model loading failed")
            mock_summarize.side_effect = Exception("Model loading failed")
            mock_transcribe.side_effect = Exception("Model loading failed")
            mock_complete.side_effect = Exception("Model loading failed")
            
            # Test that endpoints handle model loading failures gracefully
            endpoints = [
                ('/api/analyze/journal', {'text': 'I feel happy today.', 'generate_summary': True}),
                ('/api/summarize/', {'text': 'This is a long text for testing.', 'max_length': 50}),
                ('/api/transcribe/', {'audio_data': self.mock_audio_data, 'language': 'en'}),
                ('/api/complete-analysis/', {'text': 'I feel happy today.', 'include_summary': True, 'include_emotion': True})
            ]
            
            for endpoint, data in endpoints:
                response = self.client.post(endpoint, json=data)
                self.assertEqual(response.status_code, 200)  # Should still work with fallback
                response_data = json.loads(response.data)
                self.assertIn('model_used', response_data)

if __name__ == '__main__':
    unittest.main()

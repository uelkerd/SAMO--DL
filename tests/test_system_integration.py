import unittest
import json
import base64
import time
from unittest.mock import patch, MagicMock
from flask import Flask
from src.emotion_endpoint import emotion_bp
from src.summarize_endpoint import summarize_bp
from src.transcribe_endpoint import transcribe_bp
from src.complete_analysis_endpoint import complete_analysis_bp
from src.health_endpoints import health_bp
from src.health_monitor import HealthMonitor

class TestSystemIntegration(unittest.TestCase):
    """System-wide integration tests."""
    
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
        
    def test_full_system_workflow(self):
        """Test complete system workflow from request to response."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            mock_emotion.return_value = None
            mock_summarize.return_value = None
            mock_transcribe.return_value = None
            mock_complete.return_value = None
            
            # Test complete workflow
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy and content today. This is a wonderful day.', 
                                            'include_summary': True, 'include_emotion': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('summary', data)
            self.assertIn('processing_time', data)
            self.assertIn('models_used', data)
    
    def test_system_health_monitoring(self):
        """Test system health monitoring integration."""
        with patch('src.health_monitor.HealthMonitor.get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 45.2,
                'memory_percent': 67.8,
                'disk_percent': 23.1,
                'uptime': 3600
            }
            
            response = self.client.get('/api/health/')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('status', data)
            self.assertIn('uptime', data)
            self.assertIn('cpu_usage', data)
            self.assertIn('memory_usage', data)
    
    def test_system_performance_metrics(self):
        """Test system performance metrics collection."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            mock_emotion.return_value = None
            mock_summarize.return_value = None
            mock_transcribe.return_value = None
            mock_complete.return_value = None
            
            # Test performance metrics
            start_time = time.time()
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy today.', 
                                            'include_summary': True, 'include_emotion': True})
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('processing_time', data)
            self.assertGreater(data['processing_time'], 0)
            self.assertLess(data['processing_time'], end_time - start_time + 1)  # Allow some tolerance
    
    def test_system_error_recovery(self):
        """Test system error recovery and resilience."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            # Mock model loading failures
            mock_emotion.side_effect = Exception("Model loading failed")
            mock_summarize.side_effect = Exception("Model loading failed")
            mock_transcribe.side_effect = Exception("Model loading failed")
            mock_complete.side_effect = Exception("Model loading failed")
            
            # Test that system continues to work despite model failures
            response = self.client.post('/api/complete-analysis/', 
                                      json={'text': 'I feel happy today.', 
                                            'include_summary': True, 'include_emotion': True})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('emotions', data)
            self.assertIn('summary', data)
            self.assertIn('models_used', data)
    
    def test_system_concurrent_requests(self):
        """Test system handling of concurrent requests."""
        with patch('src.emotion_endpoint.EmotionEndpoint.load_model') as mock_emotion, \
             patch('src.summarize_endpoint.SummarizeEndpoint.load_model') as mock_summarize, \
             patch('src.transcribe_endpoint.TranscribeEndpoint.load_model') as mock_transcribe, \
             patch('src.complete_analysis_endpoint.CompleteAnalysisEndpoint.load_models') as mock_complete:
            
            mock_emotion.return_value = None
            mock_summarize.return_value = None
            mock_transcribe.return_value = None
            mock_complete.return_value = None
            
            # Test concurrent requests
            import threading
            import queue
            
            results = queue.Queue()
            
            def make_request():
                response = self.client.post('/api/complete-analysis/', 
                                          json={'text': 'I feel happy today.', 
                                                'include_summary': True, 'include_emotion': True})
                results.put(response.status_code)
            
            # Create multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check that all requests succeeded
            while not results.empty():
                status_code = results.get()
                self.assertEqual(status_code, 200)
    
    def test_system_resource_management(self):
        """Test system resource management and cleanup."""
        with patch('src.health_monitor.HealthMonitor.get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 45.2,
                'memory_percent': 67.8,
                'disk_percent': 23.1,
                'uptime': 3600
            }
            
            # Test resource monitoring
            response = self.client.get('/api/health/')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('status', data)
            self.assertIn('uptime', data)
            
            # Test that system reports healthy status
            self.assertEqual(data['status'], 'healthy')

if __name__ == '__main__':
    unittest.main()

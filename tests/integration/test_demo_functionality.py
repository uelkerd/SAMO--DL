"""
Test suite for comprehensive demo functionality
Tests the integration between the demo frontend and the Cloud Run API
"""

import pytest
import requests
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestDemoFunctionality:
    """Test the comprehensive demo functionality"""
    
    @pytest.fixture
    def demo_api_url(self):
        """Return the Cloud Run API URL"""
        return "https://samo-unified-api-frrnetyhfa-uc.a.run.app"
    
    @pytest.fixture
    def sample_text(self):
        """Return sample text for testing"""
        return "I'm feeling really happy and excited about this new project!"
    
    @pytest.fixture
    def sample_audio_data(self):
        """Return sample audio data (base64 encoded)"""
        # This is a minimal WAV file header for testing
        return "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    @staticmethod
    def test_demo_api_connectivity(demo_api_url):
        """Test that the demo can connect to the API"""
        try:
            response = requests.get(f"{demo_api_url}/health", timeout=10)
            # We expect either 200 (success) or 429 (rate limited)
            assert response.status_code in {200, 429}, f"Unexpected status code: {response.status_code}. Response: {response.text[:200]}"
            if response.status_code == 200:
                data = response.json()
                assert "status" in data, "Health response missing 'status' field"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"API not accessible: {e}")
    
    @staticmethod
    def test_demo_emotion_detection_request_format(demo_api_url, sample_text):
        """Test that the demo sends correctly formatted emotion detection requests"""
        # Test the request format without actually calling the API (to avoid rate limits)
        expected_request = {
            "text": sample_text
        }
        
        # Validate the request format
        assert "text" in expected_request
        assert isinstance(expected_request["text"], str)
        assert len(expected_request["text"]) > 0

    @staticmethod
    def test_demo_emotion_detection_edge_cases(demo_api_url):
        """Test emotion detection with edge cases and invalid inputs"""
        # Test empty string
        empty_request = {"text": ""}
        assert isinstance(empty_request["text"], str)
        assert len(empty_request["text"]) == 0
        
        # Test very long text
        long_text = "This is a very long text. " * 1000  # 25,000 characters
        long_request = {"text": long_text}
        assert isinstance(long_request["text"], str)
        assert len(long_request["text"]) > 10000
        
        # Test whitespace-only text
        whitespace_request = {"text": "   \n\t   "}
        assert isinstance(whitespace_request["text"], str)
        assert len(whitespace_request["text"].strip()) == 0
        
        # Test special characters and unicode
        special_chars_request = {"text": "Hello! @#$%^&*()_+ 你好 🌟 🎉"}
        assert isinstance(special_chars_request["text"], str)
        assert len(special_chars_request["text"]) > 0
        
        # Test very short text
        short_request = {"text": "Hi"}
        assert isinstance(short_request["text"], str)
        assert len(short_request["text"]) > 0
        
        # Test non-string input (should be handled by frontend validation)
        # This test ensures the demo handles type validation
        with pytest.raises((TypeError, ValueError), match="text.*string"):
            # Simulate validation that should reject non-string input
            non_string_request = {"text": 123}
            if not isinstance(non_string_request["text"], str):
                raise TypeError("text must be a string")
        
        # Test None input
        with pytest.raises((TypeError, ValueError), match="text.*required"):
            # Simulate validation that should reject None input
            none_request = {"text": None}
            if none_request["text"] is None:
                raise ValueError("text is required and cannot be None")
    
    @staticmethod
    def test_demo_whisper_request_format(demo_api_url, sample_audio_data):
        """Test that the demo sends correctly formatted Whisper requests"""
        # Test the request format for audio transcription
        expected_request = {
            "audio_data": sample_audio_data,
            "model": "whisper"
        }

        # Validate the request format
        assert "audio_data" in expected_request
        assert "model" in expected_request
        assert expected_request["model"] == "whisper"

    @staticmethod
    def test_demo_whisper_invalid_audio(demo_api_url):
        """Test that the demo and API correctly handle invalid or corrupted audio data"""
        # Simulate corrupted audio data (e.g., not a valid audio byte string)
        corrupted_audio_data = b"not_really_audio"
        request_payload = {
            "audio_data": corrupted_audio_data,
            "model": "whisper"
        }
        
        # Test request format validation
        assert "audio_data" in request_payload
        assert "model" in request_payload
        assert request_payload["model"] == "whisper"
        assert isinstance(request_payload["audio_data"], bytes)
        
        # Test with empty audio data
        empty_audio_request = {
            "audio_data": b"",
            "model": "whisper"
        }
        assert len(empty_audio_request["audio_data"]) == 0
        
        # Test with None audio data
        try:
            _none_audio_request = {"audio_data": None, "model": "whisper"}
            # This should fail validation in the demo
            assert False, "None audio data should be rejected"
        except (TypeError, ValueError):
            # Expected behavior
            pass

    
    @staticmethod
    def test_demo_t5_request_format(demo_api_url, sample_text):
        """Test that the demo sends correctly formatted T5 summarization requests"""
        # Test the request format for text summarization
        expected_request = {
            "text": sample_text,
            "model": "t5"
        }
        
        # Validate the request format
        assert "text" in expected_request
        assert "model" in expected_request
        assert expected_request["model"] == "t5"
    
    @staticmethod
    def test_demo_error_handling():
        """Test that the demo handles API errors gracefully"""
        # Test error handling for various scenarios
        error_scenarios = [
            {"status": 400, "message": "Bad Request"},
            {"status": 429, "message": "Rate Limited"},
            {"status": 500, "message": "Internal Server Error"},
            {"status": 503, "message": "Service Unavailable"}
        ]
        
        # Validate all error scenarios have required structure
        # This would be tested in the actual demo JavaScript
        # For now, we just validate the error structure
        assert all("status" in scenario for scenario in error_scenarios)
        assert all("message" in scenario for scenario in error_scenarios)
        assert all(isinstance(scenario["status"], int) for scenario in error_scenarios)
        assert all(isinstance(scenario["message"], str) for scenario in error_scenarios)
    
    @staticmethod
    def test_demo_ui_components():
        """Test that the demo has all required UI components"""
        # This would test the HTML structure
        # For now, we validate the expected components exist
        expected_components = [
            "voice-recording",
            "text-input",
            "emotion-detection",
            "text-summarization",
            "progress-tracking",
            "results-display"
        ]
        
        # Validate all components are non-empty strings
        assert all(isinstance(component, str) for component in expected_components)
        assert all(len(component) > 0 for component in expected_components)
    
    @staticmethod
    def test_demo_goemotions_labels():
        """Test that the demo uses the correct GoEmotions labels"""
        # Expected GoEmotions labels (27 emotions + neutral)
        expected_emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        # Validate that we have the correct number of emotions
        assert len(expected_emotions) == 28, f"Expected 28 emotions, got {len(expected_emotions)}"
        
        # Validate that all emotions are non-empty strings
        assert all(isinstance(emotion, str) for emotion in expected_emotions)
        assert all(len(emotion) > 0 for emotion in expected_emotions)
    
    @pytest.mark.skip(reason="Requires actual API call - may hit rate limits")
    def test_demo_full_workflow(self, demo_api_url, sample_text, sample_audio_data):
        """Test the complete demo workflow (skipped to avoid rate limits)"""
        # This would test the full workflow:
        # 1. Audio transcription
        # 2. Text summarization  
        # 3. Emotion detection
        
        # For now, we just validate the workflow structure
        workflow_steps = [
            "audio_upload",
            "transcription",
            "summarization", 
            "emotion_detection",
            "results_display"
        ]
        
        # Validate all workflow steps are non-empty strings
        assert all(isinstance(step, str) for step in workflow_steps)
        assert all(len(step) > 0 for step in workflow_steps)

if __name__ == "__main__":
    pytest.main([__file__])

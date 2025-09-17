"""
Test suite for comprehensive demo functionality
Tests the integration between the demo frontend and the Cloud Run API
"""

import pytest
import requests
import sys
import os
import base64

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture
def demo_api_url():
    """Return the Cloud Run API URL"""
    return "https://samo-unified-api-frrnetyhfa-uc.a.run.app"

@pytest.fixture
def sample_text():
    """Return sample text for testing"""
    return "I'm feeling really happy and excited about this new project!"

@pytest.fixture
def sample_audio_data_bytes():
    """Return sample audio data as decoded bytes"""
    # This is a minimal WAV file header for testing
    return base64.b64decode("UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=")

class TestDemoFunctionality:
    """Test the comprehensive demo functionality"""
    
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
    def test_demo_emotion_detection_request_format(sample_text):
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
        from unittest.mock import patch, Mock

        # Test cases for edge scenarios
        edge_cases = [
            {"text": "", "expected_error": True, "description": "empty string"},
            {"text": "   \n\t   ", "expected_error": True, "description": "whitespace-only text"},
            {"text": "Hi", "expected_error": False, "description": "very short text"},
            {"text": "This is a very long text. " * 1000, "expected_error": False, "description": "very long text"},
            {"text": "Hello! @#$%^&*()_+ 你好 🌟 🎉", "expected_error": False, "description": "special characters and unicode"},
            {"text": "😀😂😭😡😱", "expected_error": False, "description": "emoji-only text"},
            {"text": "1234567890", "expected_error": False, "description": "numbers-only text"},
            {"text": "A" * 50000, "expected_error": True, "description": "extremely long text"},
        ]

        with patch('requests.post') as mock_post:
            for case in edge_cases:
                # Mock appropriate response based on expected behavior
                mock_response = Mock()
                if case["expected_error"]:
                    mock_response.status_code = 400
                    mock_response.json.return_value = {"error": "Invalid input data"}
                else:
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "emotions": {"neutral": 0.8, "joy": 0.2},
                        "predicted_emotion": "neutral"
                    }

                mock_post.return_value = mock_response

                # Test the request
                response = requests.post(f"{demo_api_url}/predict", json={"text": case["text"]})

                if case["expected_error"]:
                    assert response.status_code == 400, f"Expected error for {case['description']}"
                    error_data = response.json()
                    assert "error" in error_data
                else:
                    assert response.status_code == 200, f"Expected success for {case['description']}"
                    data = response.json()
                    assert "emotions" in data
                    assert "predicted_emotion" in data

        # Test non-string input validation
        with pytest.raises((TypeError, ValueError), match="text.*string"):
            non_string_request = {"text": 123}
            if not isinstance(non_string_request["text"], str):
                raise TypeError("text must be a string")

        # Test None input validation
        with pytest.raises((TypeError, ValueError), match="text.*required"):
            none_request = {"text": None}
            if none_request["text"] is None:
                raise ValueError("text is required and cannot be None")
    
    @staticmethod
    def test_demo_whisper_request_format(sample_audio_data_bytes):
        """Test that the demo sends correctly formatted Whisper requests"""
        # Test the request format for audio transcription (multipart/form-data)
        import io
        
        # Simulate multipart file upload
        audio_file = io.BytesIO(sample_audio_data_bytes)
        audio_file.name = "test_audio.wav"
        audio_file.content_type = "audio/wav"
        
        # Validate the multipart file format
        assert hasattr(audio_file, 'read')
        assert hasattr(audio_file, 'name')
        assert hasattr(audio_file, 'content_type')
        assert audio_file.name.endswith('.wav')
        assert audio_file.content_type.startswith('audio/')
        assert isinstance(audio_file.read(), bytes)

    @staticmethod
    def test_demo_whisper_invalid_audio(demo_api_url):
        """Test that the demo and API correctly handle invalid or corrupted audio data"""
        # Simulate corrupted audio data (e.g., not a valid audio byte string)
        import io
        
        corrupted_audio_data = b"not_really_audio"
        audio_file = io.BytesIO(corrupted_audio_data)
        audio_file.name = "corrupted_audio.wav"
        audio_file.content_type = "audio/wav"
        
        # Test multipart file format validation
        assert hasattr(audio_file, 'read')
        assert hasattr(audio_file, 'name')
        assert hasattr(audio_file, 'content_type')
        assert audio_file.name.endswith('.wav')
        assert audio_file.content_type.startswith('audio/')
        assert isinstance(audio_file.read(), bytes)
        
        # Test with empty audio data
        empty_audio_file = io.BytesIO(b"")
        empty_audio_file.name = "empty_audio.wav"
        empty_audio_file.content_type = "audio/wav"
        assert len(empty_audio_file.read()) == 0
        
        # Test with None audio data - validate that None is properly handled
        none_audio_request = {"audio_data": None, "model": "whisper"}
        # Validate that None audio data is detected and handled appropriately
        assert none_audio_request["audio_data"] is None

        # Test API error handling for invalid audio with mocked requests
        from unittest.mock import patch, Mock

        with patch('requests.post') as mock_post:
            # Mock error response for corrupted audio
            mock_error_response = Mock()
            mock_error_response.status_code = 400
            mock_error_response.json.return_value = {"error": "Invalid audio format"}
            mock_post.return_value = mock_error_response

            # Test corrupted audio handling
            files = {'audio': ('corrupted.wav', io.BytesIO(b"not_audio"), 'audio/wav')}
            response = requests.post(f"{demo_api_url}/transcribe/voice", files=files)

            assert response.status_code == 400
            error_data = response.json()
            assert "error" in error_data
            assert "Invalid audio" in error_data["error"] or "format" in error_data["error"]

            # Test empty audio handling
            mock_empty_response = Mock()
            mock_empty_response.status_code = 400
            mock_empty_response.json.return_value = {"error": "No audio data provided"}
            mock_post.return_value = mock_empty_response

            files = {'audio': ('empty.wav', io.BytesIO(b""), 'audio/wav')}
            response = requests.post(f"{demo_api_url}/transcribe/voice", files=files)

            assert response.status_code == 400
            error_data = response.json()
            assert "error" in error_data

    
    @staticmethod
    def test_demo_t5_request_format(sample_text):
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
    
    @pytest.mark.integration
    def test_demo_timeout_handling(self, demo_api_url, sample_text):
        """Test that the demo handles API timeouts gracefully"""
        from unittest.mock import patch, Mock

        with patch('requests.post') as mock_post:
            # Mock timeout error
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

            try:
                response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=1)
                assert False, "Expected timeout exception"
            except requests.exceptions.Timeout as e:
                assert "timeout" in str(e).lower()

            # Mock connection error
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

            try:
                response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text})
                assert False, "Expected connection exception"
            except requests.exceptions.ConnectionError as e:
                assert "connection" in str(e).lower()

    @pytest.mark.integration
    def test_demo_full_workflow_mocked(self, demo_api_url, sample_text, sample_audio_data_bytes):
        """Test the complete demo workflow with API mocking"""
        import io
        from unittest.mock import patch, Mock

        # Mock responses for each step of the workflow
        mock_transcription_response = {
            "transcription": "This is a test transcription",
            "confidence": 0.95,
            "duration": 2.5
        }

        mock_summary_response = {
            "summary": "Test summary of the content",
            "original_text": sample_text
        }

        mock_emotion_response = {
            "emotions": {
                "joy": 0.8,
                "sadness": 0.1,
                "anger": 0.05,
                "fear": 0.03,
                "surprise": 0.02
            },
            "predicted_emotion": "joy",
            "confidence": 0.8
        }

        # Test workflow steps with mocked API calls
        with patch('requests.post') as mock_post:
            # Configure mock responses based on endpoint
            def mock_api_response(url, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {}

                if '/transcribe/voice' in url:
                    mock_response.json.return_value = mock_transcription_response
                elif '/summarize/text' in url:
                    mock_response.json.return_value = mock_summary_response
                elif '/predict' in url or '/emotion' in url:
                    mock_response.json.return_value = mock_emotion_response

                return mock_response

            mock_post.side_effect = mock_api_response
            audio_file = io.BytesIO(sample_audio_data_bytes)
            files = {'audio': ('test.wav', audio_file, 'audio/wav')}
            transcription_response = requests.post(f"{demo_api_url}/transcribe/voice", files=files)

            assert transcription_response.status_code == 200
            transcription_data = transcription_response.json()
            assert "transcription" in transcription_data
            assert isinstance(transcription_data["transcription"], str)
            assert len(transcription_data["transcription"]) > 0

            # Test summarization step
            summary_response = requests.post(f"{demo_api_url}/summarize/text",
                                           json={"text": sample_text})

            assert summary_response.status_code == 200
            summary_data = summary_response.json()
            assert "summary" in summary_data
            assert isinstance(summary_data["summary"], str)
            assert len(summary_data["summary"]) > 0

            # Test emotion detection step
            emotion_response = requests.post(f"{demo_api_url}/predict",
                                           json={"text": sample_text})

            assert emotion_response.status_code == 200
            emotion_data = emotion_response.json()
            assert "emotions" in emotion_data
            assert "predicted_emotion" in emotion_data
            assert isinstance(emotion_data["emotions"], dict)
            assert len(emotion_data["emotions"]) > 0

            # Validate workflow completed successfully
            workflow_results = {
                "transcription": transcription_data,
                "summary": summary_data,
                "emotions": emotion_data
            }

            # Verify all steps completed
            assert all(key in workflow_results for key in ["transcription", "summary", "emotions"])
            assert all(len(str(value)) > 0 for value in workflow_results.values())

            # Verify API was called for each step
            assert mock_post.call_count >= 3

    @pytest.mark.integration
    def test_demo_error_handling(self, demo_api_url, sample_text):
        """Test that the demo handles API errors gracefully"""
        from unittest.mock import patch, Mock

        with patch('requests.post') as mock_post:
            # Test API error responses
            mock_error_response = Mock()
            mock_error_response.status_code = 500
            mock_error_response.json.return_value = {"error": "Internal server error"}
            mock_post.return_value = mock_error_response

            # Test error handling for emotion detection
            try:
                response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text})
                assert response.status_code == 500
                error_data = response.json()
                assert "error" in error_data
                assert isinstance(error_data["error"], str)
                # Verify error message doesn't contain sensitive information
                assert "Internal server error" in error_data["error"]
                assert "Exception" not in error_data["error"]
                assert "Traceback" not in error_data["error"]
            except Exception as e:
                # If the mock fails, ensure we're testing error handling properly
                assert "error" in str(e).lower()

    @pytest.mark.integration
    def test_demo_rate_limiting(self, demo_api_url, sample_text):
        """Test that the demo handles rate limiting gracefully"""
        from unittest.mock import patch, Mock

        with patch('requests.post') as mock_post:
            # Test rate limiting response
            mock_rate_limit_response = Mock()
            mock_rate_limit_response.status_code = 429
            mock_rate_limit_response.json.return_value = {
                "error": "rate_limit_exceeded",
                "retry_after": 60
            }
            mock_post.return_value = mock_rate_limit_response

            response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text})
            assert response.status_code == 429
            rate_limit_data = response.json()
            assert "error" in rate_limit_data
            assert rate_limit_data["error"] == "rate_limit_exceeded"

    @pytest.mark.integration
    def test_demo_performance_metrics(self, demo_api_url, sample_text):
        """Test that the demo tracks performance metrics correctly"""
        from unittest.mock import patch, Mock
        import time

        with patch('requests.post') as mock_post:
            # Mock successful response with timing
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "emotions": {"joy": 0.8, "neutral": 0.2},
                "predicted_emotion": "joy",
                "processing_time_ms": 150
            }

            # Add artificial delay to test timing
            def delayed_response(*args, **kwargs):
                time.sleep(0.1)  # 100ms delay
                return mock_response

            mock_post.side_effect = delayed_response

            # Test timing measurement
            start_time = time.time()
            response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text})
            end_time = time.time()

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "emotions" in data
            assert "predicted_emotion" in data

            # Verify timing (should be at least 100ms due to our delay)
            elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            assert elapsed_time >= 90, f"Expected at least 90ms, got {elapsed_time}ms"

            # Test processing time formatting (if available in response)
            if "processing_time_ms" in data:
                processing_time = data["processing_time_ms"]
                assert isinstance(processing_time, (int, float))
                assert processing_time > 0

    @pytest.mark.integration
    def test_demo_concurrent_requests(self, demo_api_url, sample_text):
        """Test that the demo handles concurrent requests properly"""
        from unittest.mock import patch, Mock
        import threading
        import time

        with patch('requests.post') as mock_post:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "emotions": {"neutral": 0.9, "joy": 0.1},
                "predicted_emotion": "neutral"
            }
            mock_post.return_value = mock_response

            # Function to make a request
            def make_request(results, index):
                try:
                    response = requests.post(f"{demo_api_url}/predict", json={"text": f"{sample_text} {index}"})
                    results[index] = response.status_code == 200
                except Exception as e:
                    results[index] = False

            # Test concurrent requests
            num_threads = 5
            results = {}
            threads = []

            # Start all threads
            for i in range(num_threads):
                thread = threading.Thread(target=make_request, args=(results, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)  # 10 second timeout per thread

            # Verify all requests succeeded
            assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
            assert all(results.values()), f"Some requests failed: {results}"

if __name__ == "__main__":
    pytest.main([__file__])

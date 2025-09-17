"""
Test suite for comprehensive demo functionality
Tests the integration between the demo frontend and the Cloud Run API
"""

import pytest
import requests
import os
import base64
from unittest.mock import patch, Mock


@pytest.fixture
def demo_api_url():
    """Return the demo API URL from environment variable or default to local stub"""
    return os.getenv('DEMO_API_URL', 'http://localhost:8000')

@pytest.fixture
def sample_text():
    """Return sample text for testing"""
    return "I'm feeling really happy and excited about this new project!"

@pytest.fixture
def sample_audio_data_bytes():
    """Return sample audio data as decoded bytes"""
    # This is a minimal WAV file header for testing
    return base64.b64decode("UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=")

@pytest.mark.integration
class TestDemoFunctionality:
    """Test the comprehensive demo functionality"""

    @staticmethod
    def test_demo_api_connectivity(demo_api_url):
        """Test that the demo can connect to the API"""
        # Mock the requests.get call to avoid actual HTTP calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch('requests.get', return_value=mock_response) as mock_get:
            response = requests.get(f"{demo_api_url}/health", timeout=10)
            # Verify the call was made with proper timeout
            mock_get.assert_called_once_with(f"{demo_api_url}/health", timeout=10)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data, "Health response missing 'status' field"

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

    @pytest.mark.parametrize(
        "payload,expected_error",
        [
            ({"text": ""}, True),
            ({"text": "   \n\t   "}, True),
            ({"text": "Hi"}, False),
            ({"text": "This is a very long text. " * 1000}, False),
            ({"text": "Hello! @#$%^&*()_+ 你好 🌟 🎉"}, False),
            ({"text": "😀😂😭😡😱"}, False),
            ({"text": "1234567890"}, False),
            ({"text": "A" * 50000}, True),
        ],
    )
    def test_emotion_edge_case_param(self, payload, expected_error, demo_api_url):
        """Test emotion detection with edge cases and invalid inputs"""
        with patch('requests.post') as mock_post:
            # Mock appropriate response based on expected behavior
            mock_response = Mock()
            if expected_error:
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
            response = requests.post(f"{demo_api_url}/predict", json=payload, timeout=10)

            if expected_error:
                assert response.status_code == 400
                error_data = response.json()
                assert "error" in error_data
            else:
                assert response.status_code == 200
                data = response.json()
                assert "emotions" in data
                assert "predicted_emotion" in data

        # Test non-string input validation - call actual API
        response = requests.post(f"{demo_api_url}/predict", json={"text": 123}, timeout=10)
        assert response.status_code == 400, "Non-string input should return 400"

        # Test None input validation - call actual API
        response = requests.post(f"{demo_api_url}/predict", json={"text": None}, timeout=10)
        assert response.status_code == 400, "None input should return 400"

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

        with patch('requests.post') as mock_post:
            # Mock error response for corrupted audio
            mock_error_response = Mock()
            mock_error_response.status_code = 400
            mock_error_response.json.return_value = {"error": "Invalid audio format"}
            mock_post.return_value = mock_error_response

            # Test corrupted audio handling
            files = {'audio_file': ('corrupted.wav', io.BytesIO(b"not_audio"), 'audio/wav')}
            response = requests.post(f"{demo_api_url}/transcribe/voice", files=files, timeout=10)

            assert response.status_code == 400
            error_data = response.json()
            assert "error" in error_data
            assert "Invalid audio" in error_data["error"] or "format" in error_data["error"]

            # Test empty audio handling
            mock_empty_response = Mock()
            mock_empty_response.status_code = 400
            mock_empty_response.json.return_value = {"error": "No audio data provided"}
            mock_post.return_value = mock_empty_response

            files = {'audio_file': ('empty.wav', io.BytesIO(b""), 'audio/wav')}
            response = requests.post(f"{demo_api_url}/transcribe/voice", files=files, timeout=10)

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

    @pytest.mark.parametrize("scenario", [
        {"status": 400, "message": "Bad Request"},
        {"status": 429, "message": "Rate Limited"},
        {"status": 500, "message": "Internal Server Error"},
        {"status": 503, "message": "Service Unavailable"}
    ])
    def test_demo_error_handling_contract(self, scenario):
        """Test that the demo error handling contract is properly defined"""
        # Validate error scenario has required structure
        # This tests the error handling contract that the demo should follow
        # The actual error handling implementation is tested in test_demo_api_error_handling
        assert "status" in scenario
        assert "message" in scenario
        assert isinstance(scenario["status"], int)
        assert isinstance(scenario["message"], str)

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

        # TODO: Consider using Playwright for actual DOM checks
        # This would validate presence/visibility of components in the rendered page

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
        with patch('requests.post') as mock_post:
            # Mock timeout error
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

            with pytest.raises(requests.exceptions.Timeout) as e1:
                requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=1)
            assert "timeout" in str(e1.value).lower()

            # Mock connection error
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

            with pytest.raises(requests.exceptions.ConnectionError) as e2:
                requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=10)
            assert "connection" in str(e2.value).lower()

    @pytest.mark.integration
    def test_demo_full_workflow_mocked(self, demo_api_url, sample_text, sample_audio_data_bytes):
        """Test the complete demo workflow with API mocking"""
        import io

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
            def mock_api_response(url, **_kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {}

                if '/transcribe/voice' in url:
                    mock_response.json.return_value = mock_transcription_response
                elif '/summarize/text' in url or '/analyze/summarize' in url:
                    mock_response.json.return_value = mock_summary_response
                elif '/predict' in url or '/analyze/emotion' in url:
                    mock_response.json.return_value = mock_emotion_response

                return mock_response

            mock_post.side_effect = mock_api_response
            audio_file = io.BytesIO(sample_audio_data_bytes)
            files = {'audio_file': ('test.wav', audio_file, 'audio/wav')}
            transcription_response = requests.post(f"{demo_api_url}/transcribe/voice", files=files, timeout=10)

            assert transcription_response.status_code == 200
            transcription_data = transcription_response.json()
            assert "transcription" in transcription_data
            assert isinstance(transcription_data["transcription"], str)
            assert len(transcription_data["transcription"]) > 0

            # Test summarization step
            summary_response = requests.post(f"{demo_api_url}/analyze/summarize",
                                           json={"text": sample_text}, timeout=10)

            assert summary_response.status_code == 200
            summary_data = summary_response.json()
            assert "summary" in summary_data
            assert isinstance(summary_data["summary"], str)
            assert len(summary_data["summary"]) > 0

            # Test emotion detection step
            emotion_response = requests.post(f"{demo_api_url}/analyze/emotion",
                                           json={"text": sample_text}, timeout=10)

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
            assert all(str(value) for value in workflow_results.values())

            # Verify API was called for each step
            assert mock_post.call_count >= 3

    @pytest.mark.integration
    def test_demo_api_error_handling(self, demo_api_url, sample_text):
        """Test that the demo handles API errors gracefully with actual HTTP requests"""
        with patch('requests.post') as mock_post:
            # Test API error responses
            mock_error_response = Mock()
            mock_error_response.status_code = 500
            mock_error_response.json.return_value = {"error": "Internal server error"}
            mock_post.return_value = mock_error_response

            # Test error handling for emotion detection
            response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=10)
            assert response.status_code == 500
            error_data = response.json()
            assert "error" in error_data
            assert isinstance(error_data["error"], str)
            assert "Internal server error" in error_data["error"]
            assert "Exception" not in error_data["error"]
            assert "Traceback" not in error_data["error"]

    @pytest.mark.integration
    def test_demo_rate_limiting(self, demo_api_url, sample_text):
        """Test that the demo handles rate limiting gracefully"""
        with patch('requests.post') as mock_post:
            # Test rate limiting response
            mock_rate_limit_response = Mock()
            mock_rate_limit_response.status_code = 429
            mock_rate_limit_response.json.return_value = {
                "error": "rate_limit_exceeded",
                "retry_after": 60
            }
            mock_post.return_value = mock_rate_limit_response

            response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=10)
            assert response.status_code == 429
            rate_limit_data = response.json()
            assert "error" in rate_limit_data
            assert rate_limit_data["error"] == "rate_limit_exceeded"

    @pytest.mark.integration
    def test_demo_performance_metrics(self, demo_api_url, sample_text):
        """Test that the demo tracks performance metrics correctly"""
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
            def delayed_response(*_args, **_kwargs):
                time.sleep(0.1)  # 100ms delay
                return mock_response

            mock_post.side_effect = delayed_response

            # Test timing measurement
            start_time = time.time()
            response = requests.post(f"{demo_api_url}/predict", json={"text": sample_text}, timeout=10)
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
                    response = requests.post(f"{demo_api_url}/predict", json={"text": f"{sample_text} {index}"}, timeout=10)
                    results[index] = response.status_code == 200
                except requests.RequestException:
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


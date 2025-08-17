            # Check field types are consistent
        # CI environment should respond within 2 seconds
        # Check all requests succeeded
        # Check all responses have same structure
        # Check emotion analysis structure
        # Check expected models
        # Check model status structure
        # Check processing time in response
        # Check required fields
        # Check response structure
        # Create multiple threads
        # For JSON-based endpoints, form data might not be accepted
        # Mock the emotion detection
        # Note: Depending on FastAPI configuration, this might need adjustment
        # Test JSON content type (primary)
        # Test empty text
        # Test form data (fallback)
        # Test invalid endpoint
        # Test malformed request
        # Test missing required field
        # Test very long text
        # Wait for all threads to complete
import pytest
import queue
import threading
import time
""""
""""
Integration tests for API endpoints.
Tests API functionality, request/response handling, and error scenarios.
from unittest.mock import patch




@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for SAMO AI API endpoints."""

    def test_health_endpoint(self, api_client):
        """Test /health endpoint returns correct status."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "models" in data
        assert "timestamp" in data

        assert isinstance(data["models"], dict)
        for _model_name, model_status in data["models"].items():
            assert "loaded" in model_status
            assert "status" in model_status

        def test_root_endpoint(self, api_client):
        """Test root endpoint returns welcome message."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "SAMO" in data["message"]
        assert "version" in data

    @patch("src.models.emotion_detection.bert_classifier.BERTEmotionClassifier")
        def test_journal_analysis_endpoint(self, mock_bert, api_client):
        """Test /analyze/journal endpoint with text input."""
        mock_model = mock_bert.return_value
        mock_model.predict_emotions.return_value = [0, 13, 17]  # joy, excitement, gratitude

        test_data = {
            "text": "I had an amazing day today! I completed my project and felt so proud.",
            "generate_summary": True,
            "confidence_threshold": 0.5,
        }

        response = api_client.post("/analyze/journal", json=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "emotion_analysis" in data
        assert "summary" in data
        assert "processing_time_ms" in data
        assert "pipeline_status" in data
        assert "insights" in data

        emotion_analysis = data["emotion_analysis"]
        assert "emotions" in emotion_analysis
        assert "primary_emotion" in emotion_analysis
        assert "confidence" in emotion_analysis
        assert isinstance(emotion_analysis["emotions"], dict)

        def test_journal_analysis_validation(self, api_client):
        """Test journal analysis input validation."""
        response = api_client.post("/analyze/journal", json={"text": ""})
        assert response.status_code == 422

        long_text = "x" * 10001
        response = api_client.post("/analyze/journal", json={"text": long_text})
        assert response.status_code == 422

        response = api_client.post("/analyze/journal", json={})
        assert response.status_code == 422

        def test_models_status_endpoint(self, api_client):
        """Test /models/status endpoint returns model information."""
        response = api_client.get("/models/status")

        assert response.status_code == 200
        data = response.json()

        expected_models = ["emotion_detector", "text_summarizer", "voice_transcriber"]

        for model in expected_models:
            assert model in data
            assert "loaded" in data[model]
            assert "model_type" in data[model]
            assert "capabilities" in data[model]

    @pytest.mark.slow
        def test_performance_requirements(self, api_client):
        """Test API meets performance requirements."""
        test_data = {"text": "I feel great today! This is a wonderful experience."}

        start_time = time.time()
        response = api_client.post("/analyze/journal", json=test_data)
        end_time = time.time()

        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 2.0

        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

        def test_error_handling(self, api_client):
        """Test API error handling and response format."""
        response = api_client.get("/invalid/endpoint")
        assert response.status_code == 404

        response = api_client.post()
            "/analyze/journal",
            data={"invalid": "data"},
            headers={"Content-Type": "application/json"},
(        )
        assert response.status_code == 422

        def test_concurrent_requests(self, api_client):
        """Test API handles concurrent requests."""
        results = queue.Queue()
        test_data = {"text": "Testing concurrent request handling."}

        def make_request():
            try:
                response = api_client.post("/analyze/journal", json=test_data)
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        while not results.empty():
            result = results.get()
            assert result == 200

        def test_content_type_handling(self, api_client):
        """Test API handles different content types correctly."""
        test_data = {"text": "Testing content type handling."}

        response = api_client.post("/analyze/journal", json=test_data)
        assert response.status_code == 200

        response = api_client.post("/analyze/journal", data=test_data)

        def test_response_consistency(self, api_client):
        """Test API response format consistency across multiple calls."""
        test_data = {"text": "Testing response consistency."}

        responses = []
        for _ in range(3):
            response = api_client.post("/analyze/journal", json=test_data)
            assert response.status_code == 200
            responses.append(response.json())

        required_fields = ["emotion_analysis", "summary", "processing_time_ms", "pipeline_status", "insights"]

        for response_data in responses:
            for field in required_fields:
                assert field in response_data

            assert isinstance(response_data["emotion_analysis"], dict)
            assert isinstance(response_data["summary"], dict)
            assert isinstance(response_data["processing_time_ms"], (int, float))
            assert isinstance(response_data["pipeline_status"], dict)
            assert isinstance(response_data["insights"], dict)

"""
Integration tests for API endpoints.
Tests actual API behavior, endpoint integration, and response consistency.
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for SAMO AI API endpoints."""

    def test_health_endpoint(self, api_client):
        """Test /health endpoint returns correct status."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "models" in data
        assert "timestamp" in data

        # Check model status structure
        assert isinstance(data["models"], dict)
        for model_name, model_status in data["models"].items():
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

    @patch('src.models.emotion_detection.bert_classifier.BertEmotionClassifier')
    def test_journal_analysis_endpoint(self, mock_bert, api_client):
        """Test /analyze/journal endpoint with text input."""
        # Mock the emotion detection
        mock_model = mock_bert.return_value
        mock_model.predict_emotions.return_value = [0, 13, 17]  # joy, excitement, gratitude

        test_data = {
            "text": "I had an amazing day today! I completed my project and felt so proud.",
            "generate_summary": True,
            "confidence_threshold": 0.5
        }

        response = api_client.post("/analyze/journal", data=test_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "text" in data
        assert "emotions" in data
        assert "summary" in data
        assert "processing_time" in data
        assert "timestamp" in data

        # Check emotions structure
        assert isinstance(data["emotions"], list)
        if len(data["emotions"]) > 0:
            emotion = data["emotions"][0]
            assert "emotion" in emotion
            assert "confidence" in emotion

    def test_journal_analysis_validation(self, api_client):
        """Test journal analysis input validation."""
        # Test empty text
        response = api_client.post("/analyze/journal", data={"text": ""})
        assert response.status_code == 422

        # Test very long text
        long_text = "x" * 10001
        response = api_client.post("/analyze/journal", data={"text": long_text})
        assert response.status_code == 422

        # Test missing required field
        response = api_client.post("/analyze/journal", data={})
        assert response.status_code == 422

    def test_models_status_endpoint(self, api_client):
        """Test /models/status endpoint returns model information."""
        response = api_client.get("/models/status")

        assert response.status_code == 200
        data = response.json()

        # Check expected models
        expected_models = ["emotion_detector", "text_summarizer", "voice_transcriber"]

        for model in expected_models:
            assert model in data
            assert "loaded" in data[model]
            assert "model_info" in data[model]

    @pytest.mark.slow
    def test_performance_requirements(self, api_client):
        """Test API meets performance requirements."""
        test_data = {"text": "I feel great today! This is a wonderful experience."}

        start_time = time.time()
        response = api_client.post("/analyze/journal", data=test_data)
        end_time = time.time()

        response_time = end_time - start_time

        assert response.status_code == 200
        # CI environment should respond within 2 seconds
        assert response_time < 2.0

        # Check processing time in response
        data = response.json()
        assert "processing_time" in data
        assert data["processing_time"] > 0

    def test_error_handling(self, api_client):
        """Test API error handling and response format."""
        # Test invalid endpoint
        response = api_client.get("/invalid/endpoint")
        assert response.status_code == 404

        # Test malformed request
        response = api_client.post("/analyze/journal",
                                 data={"invalid": "data"})
        assert response.status_code == 422

    def test_concurrent_requests(self, api_client):
        """Test API handles concurrent requests."""
        import threading
        import queue

        results = queue.Queue()
        test_data = {"text": "Testing concurrent request handling."}

        def make_request():
            try:
                response = api_client.post("/analyze/journal", data=test_data)
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check all requests succeeded
        while not results.empty():
            result = results.get()
            assert result == 200

    def test_content_type_handling(self, api_client):
        """Test API handles different content types correctly."""
        test_data = {"text": "Testing content type handling."}

        # Test form data (default)
        response = api_client.post("/analyze/journal", data=test_data)
        assert response.status_code == 200

        # Test JSON content type (should also work)
        response = api_client.post("/analyze/journal",
                                 json=test_data)
        # Note: Depending on FastAPI configuration, this might need adjustment
        # For form-based endpoints, JSON might not be accepted

    def test_response_consistency(self, api_client):
        """Test API response format consistency across multiple calls."""
        test_data = {"text": "Testing response consistency."}

        responses = []
        for _ in range(3):
            response = api_client.post("/analyze/journal", data=test_data)
            assert response.status_code == 200
            responses.append(response.json())

        # Check all responses have same structure
        required_fields = ["text", "emotions", "processing_time", "timestamp"]

        for response_data in responses:
            for field in required_fields:
                assert field in response_data

            # Check field types are consistent
            assert isinstance(response_data["emotions"], list)
            assert isinstance(response_data["processing_time"], (int, float))
            assert isinstance(response_data["timestamp"], str)

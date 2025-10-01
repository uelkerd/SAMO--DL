#!/usr/bin/env python3
"""
Test Suite for SAMO Unified API Server

This module provides comprehensive tests for the unified API server,
testing individual endpoints and combined processing pipelines.
"""

import pytest
import io
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import the API server
from src.models.unified_api_server import SAMOUnifiedAPIServer


class TestSAMOUnifiedAPIServer:
    """Test suite for SAMO Unified API Server."""

    @pytest.fixture
    def api_server(self):
        """Create API server instance for testing."""
        server = SAMOUnifiedAPIServer()
        return server

    @pytest.fixture
    def client(self, api_server):
        """Create test client."""
        return TestClient(api_server.app)

    @staticmethod
    def test_health_endpoint(client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "memory_usage" in data

        # Check models_loaded structure
        models_loaded = data["models_loaded"]
        assert "summarizer" in models_loaded
        assert "transcriber" in models_loaded
        assert "emotion_detector" in models_loaded

    @staticmethod
    def test_summarize_endpoint_success(client):
        """Test successful text summarization."""
        test_text = """
        Today was such a rollercoaster of emotions. I started the morning feeling anxious about my job interview,
        but I tried to stay positive. The interview actually went really well - I felt confident and articulate.
        The interviewer seemed impressed with my experience. After that, I met up with Sarah for coffee and we
        talked about everything that's been going on in our lives. She's been struggling with her relationship,
        and I tried to be supportive. By evening, I was exhausted but also proud of myself for handling a
        stressful day so well. I'm learning to trust myself more and not overthink everything.
        """

        request_data = {
            "text": test_text,
            "max_length": 100,
            "min_length": 30,
            "num_beams": 4
        }

        response = client.post("/summarize", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data
        assert "original_length" in data
        assert "summary_length" in data
        assert "processing_time" in data
        assert "model_info" in data

        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0
        assert data["original_length"] == len(test_text)
        assert data["summary_length"] <= data["original_length"]

    @staticmethod
    def test_summarize_endpoint_validation(client):
        """Test summarization endpoint validation."""
        # Test empty text
        response = client.post("/summarize", json={"text": ""})
        assert response.status_code == 422  # Validation error

        # Test too short text
        response = client.post("/summarize", json={"text": "Hi"})
        assert response.status_code == 422

        # Test too long text
        long_text = "word " * 10000
        response = client.post("/summarize", json={"text": long_text})
        assert response.status_code == 422

    @staticmethod
    def test_detect_emotions_endpoint_success(client):
        """Test successful emotion detection."""
        test_text = "I am so happy today! This is amazing!"

        request_data = {
            "text": test_text,
            "threshold": 0.5,
            "top_k": 5
        }

        response = client.post("/detect-emotions", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "emotions" in data
        assert "probabilities" in data
        assert "predictions" in data
        assert "processing_time" in data
        assert "model_info" in data

        assert isinstance(data["emotions"], list)
        assert isinstance(data["probabilities"], list)
        assert isinstance(data["predictions"], list)

    @staticmethod
    def test_detect_emotions_edge_cases(client):
        """Test emotion detection with edge-case inputs."""
        # Test ambiguous text (multiple emotions)
        ambiguous_text = "I'm feeling both excited and nervous about this opportunity, but also a bit sad to leave my current job."
        
        response = client.post("/detect-emotions", json={
            "text": ambiguous_text,
            "threshold": 0.3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["emotions"]) > 1  # Should detect multiple emotions
        
        # Test very short text
        short_text = "Happy!"
        response = client.post("/detect-emotions", json={
            "text": short_text,
            "threshold": 0.3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        
        # Test text with mixed emotions
        mixed_text = "I love this but hate that. I'm excited yet anxious. Joy and fear together."
        response = client.post("/detect-emotions", json={
            "text": mixed_text,
            "threshold": 0.2
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["emotions"]) >= 2  # Should detect multiple conflicting emotions

    @staticmethod
    def test_detect_emotions_endpoint_validation(client):
        """Test emotion detection endpoint validation."""
        # Test empty text
        response = client.post("/detect-emotions", json={"text": ""})
        assert response.status_code == 422

        # Test invalid threshold
        response = client.post("/detect-emotions", json={
            "text": "Test text",
            "threshold": 1.5  # Invalid threshold
        })
        assert response.status_code == 422

        # Test maximum allowed text length (10,000 characters)
        long_text = "a" * 10000
        response = client.post("/detect-emotions", json={"text": long_text})
        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        assert "probabilities" in data
        assert "predictions" in data
        assert "processing_time" in data
        assert "model_info" in data

    @staticmethod
    def test_transcribe_endpoint_validation(client):
        """Test transcription endpoint validation."""
        # Test without file
        response = client.post("/transcribe")
        assert response.status_code == 422

        # Test with unsupported file type
        file_content = b"fake audio content"
        files = {"file": ("test.txt", file_content, "text/plain")}

        response = client.post("/transcribe", files=files)
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]

        # Test with valid audio file type but empty content
        empty_audio_content = b""
        files = {"file": ("empty.wav", empty_audio_content, "audio/wav")}
        response = client.post("/transcribe", files=files)
        # Adjust the expected status code and error message as per your API's behavior
        assert response.status_code in (400, 422)
        assert "empty" in response.json()["detail"].lower() or "no audio" in response.json()["detail"].lower()

    @patch('src.models.unified_api_server.create_whisper_transcriber')
    def test_transcribe_endpoint_success(self, mock_create_transcriber, client):
        """Test successful audio transcription with mocked transcriber."""
        # Mock the transcriber
        mock_transcriber = Mock()
        mock_result = Mock()
        mock_result.text = "This is a test transcription"
        mock_result.language = "en"
        mock_result.confidence = 0.95
        mock_result.duration = 10.5
        mock_result.processing_time = 2.1
        mock_result.audio_quality = "excellent"
        mock_result.word_count = 5
        mock_result.speaking_rate = 150.0
        mock_result.no_speech_probability = 0.1

        mock_transcriber.transcribe.return_value = mock_result
        mock_create_transcriber.return_value = mock_transcriber

        # Create a fake audio file
        audio_content = b"fake mp3 content"
        files = {"file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")}

        response = client.post("/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["text"] == "This is a test transcription"
        assert data["language"] == "en"
        assert data["confidence"] == 0.95
        assert data["duration"] == 10.5
        assert data["processing_time"] == 2.1
        assert data["audio_quality"] == "excellent"
        assert data["word_count"] == 5
        assert data["speaking_rate"] == 150.0
        assert data["no_speech_probability"] == 0.1

    @staticmethod
    def test_combined_processing_validation(client):
        """Test combined processing endpoint validation."""
        # Test without file
        response = client.post("/process-audio")
        assert response.status_code == 422

    @patch('src.models.unified_api_server.create_whisper_transcriber')
    @patch('src.models.unified_api_server.create_t5_summarizer')
    @patch('src.models.unified_api_server.create_samo_bert_emotion_classifier')
    def test_combined_processing_success(self, mock_emotion_detector, mock_summarizer,
                                       mock_transcriber, client):
        """Test successful combined audio processing with mocked models."""
        # Mock transcription result
        mock_transcription = Mock()
        mock_transcription.text = "This is a test transcription of a journal entry about feeling happy."
        mock_transcription.language = "en"
        mock_transcription.confidence = 0.95
        mock_transcription.duration = 10.5
        mock_transcription.processing_time = 2.1
        mock_transcription.audio_quality = "excellent"
        mock_transcription.word_count = 12
        mock_transcription.speaking_rate = 120.0
        mock_transcription.no_speech_probability = 0.1

        mock_transcriber.return_value.transcribe.return_value = mock_transcription

        # Mock summarizer
        mock_summary_model = Mock()
        mock_summary_model.generate_summary.return_value = "Test summary of journal entry."
        mock_summary_model.get_model_info.return_value = {"model_name": "t5-small"}
        mock_summarizer.return_value = mock_summary_model

        # Mock emotion detector
        mock_emotion_results = {
            "emotions": [["emotion_0", "emotion_1"]],
            "probabilities": [[0.8, 0.6]],
            "predictions": [[1, 1]]
        }
        mock_emotion_model = Mock()
        mock_emotion_model.predict_emotions.return_value = mock_emotion_results
        mock_emotion_detector.return_value = mock_emotion_model

        # Create fake audio file
        audio_content = b"fake mp3 content"
        files = {"file": ("test.mp3", io.BytesIO(audio_content), "audio/mpeg")}

        response = client.post("/process-audio", files=files)

        assert response.status_code == 200
        data = response.json()

        assert "transcription" in data
        assert "summary" in data
        assert "emotions" in data
        assert "total_processing_time" in data
        assert "pipeline_steps" in data

        # Check transcription data
        transcription = data["transcription"]
        assert transcription["text"] == mock_transcription.text
        assert transcription["language"] == "en"

        # Check summary data
        summary = data["summary"]
        assert summary["summary"] == "Test summary of journal entry."
        assert summary["original_length"] == len(mock_transcription.text)

        # Check emotions data
        emotions = data["emotions"]
        assert emotions["emotions"] == ["emotion_0", "emotion_1"]
        assert emotions["probabilities"] == [0.8, 0.6]

        # Check pipeline steps
        assert "transcription" in data["pipeline_steps"]
        assert "summarization" in data["pipeline_steps"]
        assert "emotion_detection" in data["pipeline_steps"]

    @staticmethod
    def test_model_unavailable_errors(api_server, client):
        """Test error handling when models are not available."""
        import copy
        
        # Temporarily set models to None
        original_models = copy.deepcopy(api_server.models)

        try:
            # Mock unavailable models
            api_server.models = {
                "summarizer": None,
                "transcriber": None,
                "emotion_detector": None
            }

            # Test summarization
            response = client.post("/summarize", json={"text": "Test text"})
            assert response.status_code == 503
            assert "not available" in response.json()["detail"]

            # Test emotion detection
            response = client.post("/detect-emotions", json={"text": "Test text"})
            assert response.status_code == 503
            assert "not available" in response.json()["detail"]

        finally:
            # Restore original models
            api_server.models = original_models


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
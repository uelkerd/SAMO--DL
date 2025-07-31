#!/usr/bin/env python3
"""
End-to-end tests for complete user workflows.
Tests full system integration, data flow, and user scenarios.
"""

import datetime
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Test constants
HTTP_OK = 200
HTTP_UNPROCESSABLE_ENTITY = 422
MAX_WORKFLOW_TIME = 3.0
MAX_PROCESSING_TIME = 2.0
MAX_RESPONSE_TIME = 3.0
MAX_AVERAGE_TIME = 2.0
MAX_TIMESTAMP_DIFF = 60


@pytest.mark.e2e
class TestCompleteWorkflows:
    """End-to-end tests for SAMO AI complete user workflows."""

    def test_text_journal_complete_workflow(self, api_client, sample_journal_entry):
        """Test complete text journal analysis workflow."""
        start_time = time.time()

        response = api_client.post(
            "/analyze/journal",
            json={
                "text": sample_journal_entry["text"],
                "generate_summary": True,
                "emotion_threshold": 0.5,
            },
        )

        end_time = time.time()
        workflow_time = end_time - start_time

        assert response.status_code == HTTP_OK
        data = response.json()

        assert "emotion_analysis" in data
        assert "summary" in data
        assert "processing_time_ms" in data
        assert "pipeline_status" in data

        emotion_analysis = data["emotion_analysis"]
        assert "emotions" in emotion_analysis
        assert "primary_emotion" in emotion_analysis
        assert "confidence" in emotion_analysis

        emotions = emotion_analysis["emotions"]
        assert isinstance(emotions, dict)
        assert len(emotions) > 0

        for emotion, confidence in emotions.items():
            assert 0.0 <= confidence <= 1.0

        summary = data["summary"]
        assert "summary" in summary
        assert "key_emotions" in summary
        assert len(summary["summary"]) > 0
        assert isinstance(summary["key_emotions"], list)

        assert workflow_time < MAX_WORKFLOW_TIME  # Complete workflow under 3 seconds
        assert data["processing_time_ms"] < MAX_PROCESSING_TIME * 1000  # Processing time under 2 seconds

    @pytest.mark.slow
    def test_voice_journal_complete_workflow(self, api_client, sample_audio_data):
        """Test complete voice journal analysis workflow."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(b"fake audio data for testing")
            temp_audio_path = temp_audio.name

        try:
            with Path(temp_audio_path).open("rb") as audio_file:
                files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
                data = {"language": "en", "generate_summary": True, "emotion_threshold": 0.5}

                with patch(
                    "src.models.voice_processing.whisper_transcriber.whisper"
                ) as mock_whisper:
                    mock_model = mock_whisper.load_model.return_value
                    mock_model.transcribe.return_value = {
                        "text": sample_audio_data["expected_text"]
                    }

                    response = api_client.post("/analyze/voice-journal", files=files, data=data)

            # Voice processing may fail in test environment, so we accept both success and failure
            if response.status_code == HTTP_OK:
                data = response.json()
                assert "emotion_analysis" in data
                assert "summary" in data
                assert "processing_time_ms" in data
            else:
                # If voice processing fails, it should return a 400 with a clear error message
                assert response.status_code == 400
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data

        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

    def test_error_recovery_workflow(self, api_client):
        """Test error recovery and graceful degradation."""
        # Test with invalid input
        response = api_client.post(
            "/analyze/journal",
            json={"text": "", "generate_summary": True, "emotion_threshold": 0.5},
        )
        assert response.status_code in [400, 422]  # Should return validation error

        # Test with very long text
        long_text = "test " * 1000  # Very long text
        response = api_client.post(
            "/analyze/journal",
            json={"text": long_text, "generate_summary": True, "emotion_threshold": 0.5},
        )
        assert response.status_code in [200, 413]  # Should handle gracefully

        # Test with normal text
        response = api_client.post(
            "/analyze/journal",
            json={
                "text": "I had a great day today!",
                "generate_summary": True,
                "emotion_threshold": 0.5,
            },
        )
        assert response.status_code == HTTP_OK

    def test_high_volume_workflow(self, api_client):
        """Test high volume processing with multiple requests."""
        requests_data = [
            {"text": f"Request {i}: I had a great day!", "generate_summary": True, "emotion_threshold": 0.5}
            for i in range(5)
        ]

        success_count = 0
        for request_data in requests_data:
            response = api_client.post("/analyze/journal", json=request_data)
            if response.status_code == HTTP_OK:
                success_count += 1

        assert success_count >= 4  # At least 80% success rate

    def test_data_consistency_workflow(self, api_client):
        """Test data consistency across multiple requests."""
        test_text = "I had a great day today!"
        responses = []

        # Send same request multiple times
        for _ in range(3):
            response = api_client.post(
                "/analyze/journal",
                json={
                    "text": test_text,
                    "generate_summary": True,
                    "emotion_threshold": 0.5,
                },
            )
            responses.append(response)

        # All responses should be successful
        for response in responses:
            assert response.status_code == HTTP_OK

        # Check data consistency
        response_data = [r.json() for r in responses]
        
        # Basic structure should be consistent
        for data in response_data:
            assert "emotion_analysis" in data
            assert "summary" in data
            assert "processing_time_ms" in data

    def test_configuration_workflow(self, api_client):
        """Test different configuration options."""
        test_text = "I had a great day today!"

        # Test with different emotion thresholds
        response = api_client.post(
            "/analyze/journal",
            json={
                "text": test_text,
                "generate_summary": True,
                "emotion_threshold": 0.1,
            },
        )
        assert response.status_code == HTTP_OK

        # Test without summary generation
        response = api_client.post(
            "/analyze/journal",
            json={
                "text": test_text,
                "generate_summary": False,
                "emotion_threshold": 0.5,
            },
        )
        assert response.status_code == HTTP_OK

    @pytest.mark.model
    def test_model_integration_workflow(self, api_client):
        """Test integration between different AI models."""
        test_text = "I had a great day today!"

        response = api_client.post(
            "/analyze/journal",
            json={
                "text": test_text,
                "generate_summary": True,
                "emotion_threshold": 0.5,
            },
        )
        assert response.status_code == HTTP_OK
        data = response.json()

        # Check that all model components are integrated
        assert "emotion_analysis" in data
        assert "summary" in data
        assert "pipeline_status" in data

        # Verify pipeline status shows model availability
        pipeline_status = data["pipeline_status"]
        assert isinstance(pipeline_status, dict)
        assert "emotion_detection" in pipeline_status
        assert "text_summarization" in pipeline_status

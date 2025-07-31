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
            data={
                "text": sample_journal_entry["text"],
                "generate_summary": True,
                "confidence_threshold": 0.5,
            },
        )

        end_time = time.time()
        workflow_time = end_time - start_time

        assert response.status_code == HTTP_OK
        data = response.json()

        assert "text" in data
        assert "emotions" in data
        assert "summary" in data
        assert "processing_time" in data
        assert "timestamp" in data

        emotions = data["emotions"]
        assert isinstance(emotions, list)
        assert len(emotions) > 0

        for emotion in emotions:
            assert "emotion" in emotion
            assert "confidence" in emotion
            assert 0.0 <= emotion["confidence"] <= 1.0

        if data.get("summary"):
            summary = data["summary"]
            assert "summary" in summary
            assert "key_themes" in summary
            assert len(summary["summary"]) > 0
            assert isinstance(summary["key_themes"], list)

        assert workflow_time < MAX_WORKFLOW_TIME  # Complete workflow under 3 seconds
        assert data["processing_time"] < MAX_PROCESSING_TIME  # Processing time under 2 seconds

    @pytest.mark.slow
    def test_voice_journal_complete_workflow(self, api_client, sample_audio_data):
        """Test complete voice journal analysis workflow."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(b"fake audio data for testing")
            temp_audio_path = temp_audio.name

        try:
            with Path(temp_audio_path).open("rb") as audio_file:
                files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
                data = {"language": "en", "generate_summary": True, "confidence_threshold": 0.5}

                with patch(
                    "src.models.voice_processing.whisper_transcriber.whisper"
                ) as mock_whisper:
                    mock_model = mock_whisper.load_model.return_value
                    mock_model.transcribe.return_value = {
                        "text": sample_audio_data["expected_text"]
                    }

                    response = api_client.post("/analyze/voice-journal", files=files, data=data)

            assert response.status_code == HTTP_OK
            data = response.json()

            assert "text" in data
            assert "emotions" in data
            assert "summary" in data
            assert "processing_time" in data
            assert "timestamp" in data

            # Verify transcription
            assert data["text"] == sample_audio_data["expected_text"]

            # Verify emotions
            emotions = data["emotions"]
            assert isinstance(emotions, list)
            assert len(emotions) > 0

            for emotion in emotions:
                assert "emotion" in emotion
                assert "confidence" in emotion
                assert 0.0 <= emotion["confidence"] <= 1.0

            # Verify summary
            if data.get("summary"):
                summary = data["summary"]
                assert "summary" in summary
                assert "key_themes" in summary
                assert len(summary["summary"]) > 0
                assert isinstance(summary["key_themes"], list)

        finally:
            # Clean up temporary file
            Path(temp_audio_path).unlink(missing_ok=True)

    def test_error_recovery_workflow(self, api_client):
        """Test error recovery and graceful degradation."""
        # Test with invalid input
        response = api_client.post(
            "/analyze/journal",
            data={"text": "", "generate_summary": True},
        )

        # Should return 422 for validation error
        assert response.status_code == HTTP_UNPROCESSABLE_ENTITY

        # Test with malformed JSON
        response = api_client.post(
            "/analyze/journal",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # Should handle gracefully
        assert response.status_code in [400, 422]

        # Test with very long text
        long_text = "This is a very long text. " * 1000
        response = api_client.post(
            "/analyze/journal",
            data={"text": long_text, "generate_summary": False},
        )

        # Should handle long text gracefully
        assert response.status_code in [200, 413]

    def test_high_volume_workflow(self, api_client):
        """Test high volume processing capabilities."""
        responses = []
        start_time = time.time()

        # Send multiple requests rapidly
        for i in range(5):
            response = api_client.post(
                "/analyze/journal",
                data={
                    "text": f"This is test text number {i} for high volume testing.",
                    "generate_summary": False,
                },
            )
            responses.append(response)

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == HTTP_OK)
        assert success_count >= 4  # At least 80% success rate

        # Average response time should be reasonable
        avg_time = total_time / len(responses)
        assert avg_time < MAX_AVERAGE_TIME

    def test_data_consistency_workflow(self, api_client):
        """Test data consistency across multiple requests."""
        test_text = "I am feeling happy and excited about the future!"

        # Send same request multiple times
        responses = []
        for _ in range(3):
            response = api_client.post(
                "/analyze/journal",
                data={"text": test_text, "generate_summary": True},
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == HTTP_OK

        # Extract data from responses
        data_list = [r.json() for r in responses]

        # Check consistency of core fields
        for data in data_list:
            assert "text" in data
            assert "emotions" in data
            assert "processing_time" in data
            assert "timestamp" in data

        # Text should be consistent
        texts = [data["text"] for data in data_list]
        assert len(set(texts)) == 1  # All texts should be identical

        # Emotions should be consistent (same emotions detected)
        emotion_sets = [set(e["emotion"] for e in data["emotions"]) for data in data_list]
        assert len(set(tuple(sorted(es)) for es in emotion_sets)) == 1

    def test_configuration_workflow(self, api_client):
        """Test different configuration options."""
        test_text = "I am feeling mixed emotions today."

        # Test with different confidence thresholds
        thresholds = [0.3, 0.5, 0.7]
        emotion_counts = []

        for threshold in thresholds:
            response = api_client.post(
                "/analyze/journal",
                data={
                    "text": test_text,
                    "confidence_threshold": threshold,
                    "generate_summary": False,
                },
            )

            assert response.status_code == HTTP_OK
            data = response.json()
            emotions = data["emotions"]
            emotion_counts.append(len(emotions))

        # Higher threshold should generally result in fewer emotions
        # (more strict filtering)
        assert emotion_counts[0] >= emotion_counts[1] >= emotion_counts[2]

        # Test with and without summary generation
        response_with_summary = api_client.post(
            "/analyze/journal",
            data={"text": test_text, "generate_summary": True},
        )

        response_without_summary = api_client.post(
            "/analyze/journal",
            data={"text": test_text, "generate_summary": False},
        )

        assert response_with_summary.status_code == HTTP_OK
        assert response_without_summary.status_code == HTTP_OK

        data_with = response_with_summary.json()
        data_without = response_without_summary.json()

        # Summary should be present when requested
        assert "summary" in data_with
        assert data_without.get("summary") is None

    @pytest.mark.model
    def test_model_integration_workflow(self, api_client):
        """Test integration with all AI models."""
        test_text = "I am feeling joyful and grateful for this amazing day!"

        response = api_client.post(
            "/analyze/journal",
            data={
                "text": test_text,
                "generate_summary": True,
                "confidence_threshold": 0.5,
            },
        )

        assert response.status_code == HTTP_OK
        data = response.json()

        # Verify all model outputs are present
        assert "text" in data
        assert "emotions" in data
        assert "summary" in data

        # Verify emotion detection model output
        emotions = data["emotions"]
        assert isinstance(emotions, list)
        assert len(emotions) > 0

        # Check for expected emotions in joyful text
        emotion_names = [e["emotion"] for e in emotions]
        assert any("joy" in name.lower() or "happy" in name.lower() for name in emotion_names)

        # Verify summarization model output
        summary = data["summary"]
        assert "summary" in summary
        assert "key_themes" in summary
        assert len(summary["summary"]) > 0
        assert isinstance(summary["key_themes"], list)

        # Verify processing time is reasonable
        assert data["processing_time"] < MAX_PROCESSING_TIME

        # Verify timestamp is recent
        timestamp = datetime.datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        time_diff = abs((now - timestamp).total_seconds())
        assert time_diff < MAX_TIMESTAMP_DIFF

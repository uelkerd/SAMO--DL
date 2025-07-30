"""
End-to-end tests for complete user workflows.
Tests full system integration, data flow, and user scenarios.
"""

import tempfile
from pathlib import Path


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
        # Step 1: Submit journal entry for analysis
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

        # Verify successful response
        assert response.status_code == HTTP_OK
        data = response.json()

        # Step 2: Validate complete analysis structure
        assert "text" in data
        assert "emotions" in data
        assert "summary" in data
        assert "processing_time" in data
        assert "timestamp" in data

        # Step 3: Verify emotion detection results
        emotions = data["emotions"]
        assert isinstance(emotions, list)
        assert len(emotions) > 0

        for emotion in emotions:
            assert "emotion" in emotion
            assert "confidence" in emotion
            assert 0.0 <= emotion["confidence"] <= 1.0

        # Step 4: Verify summarization results
        if data.get("summary"):
            summary = data["summary"]
            assert "summary" in summary
            assert "key_themes" in summary
            assert len(summary["summary"]) > 0
            assert isinstance(summary["key_themes"], list)

        # Step 5: Verify performance requirements
        assert workflow_time < MAX_WORKFLOW_TIME  # Complete workflow under 3 seconds
        assert data["processing_time"] < MAX_PROCESSING_TIME  # Processing time under 2 seconds

    @pytest.mark.slow
    def test_voice_journal_complete_workflow(self, api_client, sample_audio_data):
        """Test complete voice journal analysis workflow."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            # Note: In real implementation, we'd write actual audio data
            temp_audio.write(b"fake audio data for testing")
            temp_audio_path = temp_audio.name

        try:
            # Step 1: Submit audio file for analysis
            with Path(temp_audio_path).open("rb") as audio_file:
                files = {"audio_file": ("test_audio.wav", audio_file, "audio/wav")}
                data = {"language": "en", "generate_summary": True, "confidence_threshold": 0.5}

                # Mock the transcription for testing
                with patch(
                    "src.models.voice_processing.whisper_transcriber.whisper"
                ) as mock_whisper:
                    mock_model = mock_whisper.load_model.return_value
                    mock_model.transcribe.return_value = {
                        "text": sample_audio_data["expected_text"]
                    }

                    response = api_client.post("/analyze/voice-journal", files=files, data=data)

            # Step 2: Verify successful transcription and analysis
            assert response.status_code == HTTP_OK
            result = response.json()

            # Step 3: Validate complete voice analysis structure
            assert "transcription" in result
            assert "text" in result
            assert "emotions" in result
            assert "summary" in result
            assert "processing_time" in result

            # Step 4: Verify transcription quality
            transcription = result["transcription"]
            assert "text" in transcription
            assert "language" in transcription
            assert "confidence" in transcription

        finally:
            # Cleanup
            Path(temp_audio_path).unlink(missing_ok=True)

    def test_error_recovery_workflow(self, api_client):
        """Test system error recovery and graceful degradation."""
        # Step 1: Test with invalid input
        response = api_client.post("/analyze/journal", data={"text": ""})
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

        # Step 2: Test with valid input after error
        response = api_client.post(
            "/analyze/journal",
            data={"text": "This is a valid journal entry for testing error recovery."},
        )
        assert response.status_code == 200

        # Step 3: Verify system continues working normally
        data = response.json()
        assert "emotions" in data
        assert len(data["emotions"]) >= 0

    def test_high_volume_workflow(self, api_client):
        """Test system behavior under high volume of requests."""
        test_texts = [
            "I feel incredibly happy today!",
            "Work was stressful and overwhelming.",
            "Had a peaceful walk in the park.",
            "Excited about my upcoming vacation.",
            "Feeling anxious about the presentation tomorrow.",
        ]

        results = []
        total_start_time = time.time()

        # Step 1: Submit multiple requests
        for text in test_texts:
            start_time = time.time()
            response = api_client.post("/analyze/journal", data={"text": text})
            end_time = time.time()

            assert response.status_code == 200
            data = response.json()

            results.append({"response": data, "response_time": end_time - start_time})

        total_time = time.time() - total_start_time

        # Step 2: Verify all requests succeeded
        assert len(results) == len(test_texts)

        # Step 3: Verify performance consistency
        for result in results:
            assert result["response_time"] < 3.0  # Each request under 3 seconds
            assert "emotions" in result["response"]

        # Step 4: Verify total throughput
        average_time = total_time / len(test_texts)
        assert average_time < 2.0  # Average processing under 2 seconds

    def test_data_consistency_workflow(self, api_client):
        """Test data consistency across multiple processing steps."""
        original_text = "I had an amazing day today! I completed my project and felt incredibly proud of my accomplishment."

        # Step 1: Submit for analysis
        response = api_client.post(
            "/analyze/journal", data={"text": original_text, "generate_summary": True}
        )

        assert response.status_code == 200
        data = response.json()

        # Step 2: Verify original text preserved
        assert data["text"] == original_text

        # Step 3: Verify data relationships
        if data.get("summary") and data["summary"].get("original_length"):
            assert data["summary"]["original_length"] == len(original_text)

        # Step 4: Verify timestamp consistency
        assert "timestamp" in data
        # Timestamp should be recent (within last minute)
        import datetime

        # Use more robust timestamp parsing
        timestamp_str = data["timestamp"]
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"

        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        now = datetime.datetime.now(datetime.UTC)
        time_diff = (now - timestamp).total_seconds()
        assert time_diff < 60  # Within last minute

    def test_configuration_workflow(self, api_client):
        """Test different configuration options work correctly."""
        test_text = "I'm feeling mixed emotions about this situation."

        # Test different confidence thresholds
        thresholds = [0.1, 0.5, 0.8]

        for threshold in thresholds:
            response = api_client.post(
                "/analyze/journal", data={"text": test_text, "confidence_threshold": threshold}
            )

            assert response.status_code == 200
            data = response.json()

            # Higher thresholds should generally result in fewer emotions
            emotions = data["emotions"]
            assert isinstance(emotions, list)

            # Verify all returned emotions meet threshold
            for emotion in emotions:
                assert emotion["confidence"] >= threshold

    @pytest.mark.model
    def test_model_integration_workflow(self, api_client):
        """Test integration between different AI models."""
        test_text = (
            "Today was fantastic! I achieved my goals and felt genuinely happy and grateful."
        )

        # Step 1: Request full analysis
        response = api_client.post(
            "/analyze/journal", data={"text": test_text, "generate_summary": True}
        )

        assert response.status_code == 200
        data = response.json()

        # Step 2: Verify emotion detection worked
        emotions = data.get("emotions", [])
        assert len(emotions) > 0

        # Step 3: Verify summarization worked
        summary = data.get("summary")
        if summary:
            assert "summary" in summary
            assert len(summary["summary"]) > 0
            # Summary should be shorter than original
            assert len(summary["summary"]) < len(test_text)

        # Step 4: Verify models worked together coherently
        # If emotions detected positive feelings, summary should reflect that
        emotion_names = [e["emotion"] for e in emotions]
        positive_emotions = {"joy", "happiness", "gratitude", "excitement", "pride"}

        has_positive_emotion = any(emotion in positive_emotions for emotion in emotion_names)

        if has_positive_emotion and summary:
            summary_text = summary["summary"].lower()
            # Summary should contain some positive language
            positive_words = ["positive", "happy", "great", "good", "wonderful", "fantastic"]
            assert any(word in summary_text for word in positive_words)

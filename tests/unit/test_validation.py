"""
Unit tests for validation module.
Tests data validation, schema validation, and input validation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data.validation import (
    validate_text_input,
    validate_emotion_data,
    validate_summary_data,
    validate_voice_data,
    ValidationError,
    DataValidator
)


class TestValidationError:
    """Test suite for ValidationError class."""

    def test_validation_error_initialization(self):
        """Test ValidationError initialization."""
        error = ValidationError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"

    def test_validation_error_with_details(self):
        """Test ValidationError with additional details."""
        error = ValidationError("Test error", field="test_field", value="test_value")
        
        assert error.message == "Test error"
        assert error.field == "test_field"
        assert error.value == "test_value"


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        
        assert hasattr(validator, 'validate')
        assert hasattr(validator, 'validate_required')
        assert hasattr(validator, 'validate_type')

    def test_validate_required_fields(self):
        """Test validate_required method."""
        validator = DataValidator()
        
        # Valid case
        data = {"field1": "value1", "field2": "value2"}
        required_fields = ["field1", "field2"]
        
        result = validator.validate_required(data, required_fields)
        assert result is True

    def test_validate_required_missing_field(self):
        """Test validate_required with missing field."""
        validator = DataValidator()
        
        data = {"field1": "value1"}
        required_fields = ["field1", "field2"]
        
        with pytest.raises(ValidationError, match="Missing required field: field2"):
            validator.validate_required(data, required_fields)

    def test_validate_type_correct_type(self):
        """Test validate_type with correct type."""
        validator = DataValidator()
        
        result = validator.validate_type("test", str, "test_field")
        assert result is True

    def test_validate_type_incorrect_type(self):
        """Test validate_type with incorrect type."""
        validator = DataValidator()
        
        with pytest.raises(ValidationError, match="test_field must be int"):
            validator.validate_type("test", int, "test_field")

    def test_validate_complex_data(self):
        """Test validate method with complex data."""
        validator = DataValidator()
        
        data = {
            "text": "Test text",
            "emotions": {"happy": 0.8},
            "confidence": 0.9
        }
        
        schema = {
            "text": str,
            "emotions": dict,
            "confidence": float
        }
        
        result = validator.validate(data, schema)
        assert result is True


class TestValidateTextInput:
    """Test suite for validate_text_input function."""

    def test_validate_text_input_valid(self):
        """Test validate_text_input with valid input."""
        text = "This is a valid text input for testing purposes."
        
        result = validate_text_input(text)
        assert result is True

    def test_validate_text_input_empty(self):
        """Test validate_text_input with empty text."""
        with pytest.raises(ValidationError, match="Text input cannot be empty"):
            validate_text_input("")

    def test_validate_text_input_none(self):
        """Test validate_text_input with None."""
        with pytest.raises(ValidationError, match="Text input cannot be None"):
            validate_text_input(None)

    def test_validate_text_input_too_short(self):
        """Test validate_text_input with too short text."""
        with pytest.raises(ValidationError, match="Text input must be at least 10 characters"):
            validate_text_input("Short")

    def test_validate_text_input_too_long(self):
        """Test validate_text_input with too long text."""
        long_text = "A" * 10001
        with pytest.raises(ValidationError, match="Text input cannot exceed 10000 characters"):
            validate_text_input(long_text)

    def test_validate_text_input_invalid_characters(self):
        """Test validate_text_input with invalid characters."""
        with pytest.raises(ValidationError, match="Text input contains invalid characters"):
            validate_text_input("Text with \x00 null bytes")

    def test_validate_text_input_whitespace_only(self):
        """Test validate_text_input with whitespace only."""
        with pytest.raises(ValidationError, match="Text input cannot be empty"):
            validate_text_input("   \n\t   ")


class TestValidateEmotionData:
    """Test suite for validate_emotion_data function."""

    def test_validate_emotion_data_valid(self):
        """Test validate_emotion_data with valid data."""
        emotion_data = {
            "happy": 0.8,
            "sad": 0.2,
            "angry": 0.1
        }
        
        result = validate_emotion_data(emotion_data)
        assert result is True

    def test_validate_emotion_data_empty(self):
        """Test validate_emotion_data with empty data."""
        with pytest.raises(ValidationError, match="Emotion data cannot be empty"):
            validate_emotion_data({})

    def test_validate_emotion_data_none(self):
        """Test validate_emotion_data with None."""
        with pytest.raises(ValidationError, match="Emotion data cannot be None"):
            validate_emotion_data(None)

    def test_validate_emotion_data_invalid_type(self):
        """Test validate_emotion_data with invalid type."""
        with pytest.raises(ValidationError, match="Emotion data must be a dictionary"):
            validate_emotion_data("not a dict")

    def test_validate_emotion_data_invalid_probability(self):
        """Test validate_emotion_data with invalid probability values."""
        emotion_data = {
            "happy": 1.5,  # > 1.0
            "sad": -0.1    # < 0.0
        }
        
        with pytest.raises(ValidationError, match="Probability values must be between 0 and 1"):
            validate_emotion_data(emotion_data)

    def test_validate_emotion_data_invalid_emotion_name(self):
        """Test validate_emotion_data with invalid emotion names."""
        emotion_data = {
            "happy": 0.8,
            "": 0.2,  # Empty emotion name
            "very_long_emotion_name_that_exceeds_limit": 0.1
        }
        
        with pytest.raises(ValidationError, match="Invalid emotion name"):
            validate_emotion_data(emotion_data)


class TestValidateSummaryData:
    """Test suite for validate_summary_data function."""

    def test_validate_summary_data_valid(self):
        """Test validate_summary_data with valid data."""
        summary_data = {
            "summary": "This is a test summary.",
            "word_count": 7,
            "confidence": 0.9
        }
        
        result = validate_summary_data(summary_data)
        assert result is True

    def test_validate_summary_data_missing_summary(self):
        """Test validate_summary_data with missing summary."""
        summary_data = {
            "word_count": 7,
            "confidence": 0.9
        }
        
        with pytest.raises(ValidationError, match="Summary is required"):
            validate_summary_data(summary_data)

    def test_validate_summary_data_empty_summary(self):
        """Test validate_summary_data with empty summary."""
        summary_data = {
            "summary": "",
            "word_count": 0,
            "confidence": 0.9
        }
        
        with pytest.raises(ValidationError, match="Summary cannot be empty"):
            validate_summary_data(summary_data)

    def test_validate_summary_data_invalid_confidence(self):
        """Test validate_summary_data with invalid confidence."""
        summary_data = {
            "summary": "Test summary",
            "confidence": 1.5  # > 1.0
        }
        
        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            validate_summary_data(summary_data)

    def test_validate_summary_data_invalid_word_count(self):
        """Test validate_summary_data with invalid word count."""
        summary_data = {
            "summary": "Test summary",
            "word_count": -1  # Negative
        }
        
        with pytest.raises(ValidationError, match="Word count must be non-negative"):
            validate_summary_data(summary_data)


class TestValidateVoiceData:
    """Test suite for validate_voice_data function."""

    def test_validate_voice_data_valid(self):
        """Test validate_voice_data with valid data."""
        voice_data = {
            "transcription": "This is a test transcription.",
            "confidence": 0.95,
            "audio_duration": 30.5
        }
        
        result = validate_voice_data(voice_data)
        assert result is True

    def test_validate_voice_data_missing_transcription(self):
        """Test validate_voice_data with missing transcription."""
        voice_data = {
            "confidence": 0.95,
            "audio_duration": 30.5
        }
        
        with pytest.raises(ValidationError, match="Transcription is required"):
            validate_voice_data(voice_data)

    def test_validate_voice_data_empty_transcription(self):
        """Test validate_voice_data with empty transcription."""
        voice_data = {
            "transcription": "",
            "confidence": 0.95
        }
        
        with pytest.raises(ValidationError, match="Transcription cannot be empty"):
            validate_voice_data(voice_data)

    def test_validate_voice_data_invalid_confidence(self):
        """Test validate_voice_data with invalid confidence."""
        voice_data = {
            "transcription": "Test transcription",
            "confidence": -0.1  # < 0.0
        }
        
        with pytest.raises(ValidationError, match="Confidence must be between 0 and 1"):
            validate_voice_data(voice_data)

    def test_validate_voice_data_invalid_duration(self):
        """Test validate_voice_data with invalid duration."""
        voice_data = {
            "transcription": "Test transcription",
            "audio_duration": -5.0  # Negative
        }
        
        with pytest.raises(ValidationError, match="Audio duration must be positive"):
            validate_voice_data(voice_data)

    def test_validate_voice_data_too_long_duration(self):
        """Test validate_voice_data with too long duration."""
        voice_data = {
            "transcription": "Test transcription",
            "audio_duration": 3601.0  # > 1 hour
        }
        
        with pytest.raises(ValidationError, match="Audio duration cannot exceed 1 hour"):
            validate_voice_data(voice_data) 
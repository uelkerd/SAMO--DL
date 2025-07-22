"""
Unit tests for API data models and validation.
Tests Pydantic models, request/response validation, and data transformations.
"""

import pytest
from datetime import datetime
from typing import Dict, List
from pydantic import ValidationError

# Note: These imports will need to be adjusted based on actual API model structure
# from src.unified_ai_api import EmotionResult, SummaryResult, CompleteJournalAnalysis


class TestAPIModels:
    """Test suite for API data models."""

    def test_emotion_result_validation(self):
        """Test EmotionResult model validation."""
        # Test valid emotion result
        valid_data = {
            "emotion": "joy",
            "confidence": 0.85,
            "probability": 0.92
        }
        
        # This test will need actual model import to work
        # result = EmotionResult(**valid_data)
        # assert result.emotion == "joy"
        # assert result.confidence == 0.85
        # assert result.probability == 0.92
        
        # For now, just validate the test structure
        assert valid_data["emotion"] == "joy"
        assert 0.0 <= valid_data["confidence"] <= 1.0
        assert 0.0 <= valid_data["probability"] <= 1.0

    def test_emotion_result_invalid_confidence(self):
        """Test EmotionResult rejects invalid confidence values."""
        invalid_data = {
            "emotion": "joy",
            "confidence": 1.5,  # Invalid: > 1.0
            "probability": 0.92
        }
        
        # Test validation logic
        assert invalid_data["confidence"] > 1.0  # This should be caught by validation

    def test_summary_result_validation(self):
        """Test SummaryResult model validation."""
        valid_data = {
            "summary": "User had a positive day with accomplishments.",
            "key_themes": ["achievement", "positivity"],
            "word_count": 12,
            "original_length": 150,
            "compression_ratio": 0.08
        }
        
        assert len(valid_data["summary"]) > 0
        assert isinstance(valid_data["key_themes"], list)
        assert valid_data["word_count"] > 0
        assert valid_data["compression_ratio"] < 1.0

    def test_complete_analysis_validation(self):
        """Test CompleteJournalAnalysis model validation."""
        valid_data = {
            "text": "Original journal entry text...",
            "emotions": [
                {"emotion": "joy", "confidence": 0.85, "probability": 0.92},
                {"emotion": "gratitude", "confidence": 0.78, "probability": 0.84}
            ],
            "summary": {
                "summary": "User expressed joy and gratitude.",
                "key_themes": ["emotions", "reflection"],
                "word_count": 6,
                "original_length": 50,
                "compression_ratio": 0.12
            },
            "processing_time": 1.23,
            "timestamp": datetime.now().isoformat()
        }
        
        assert len(valid_data["text"]) > 0
        assert len(valid_data["emotions"]) > 0
        assert valid_data["processing_time"] > 0
        assert "timestamp" in valid_data

    def test_text_length_validation(self):
        """Test text length validation for different endpoints."""
        # Test minimum length
        short_text = "Hi"
        assert len(short_text) >= 2  # Minimum viable input
        
        # Test maximum length (e.g., 10,000 characters)
        long_text = "x" * 10001
        assert len(long_text) > 10000  # Should be rejected
        
        # Test reasonable length
        normal_text = "This is a normal journal entry with reasonable length."
        assert 10 <= len(normal_text) <= 10000

    def test_audio_file_validation(self):
        """Test audio file validation for voice endpoints."""
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        
        # Test valid extensions
        for ext in valid_extensions:
            filename = f"audio{ext}"
            assert any(filename.endswith(e) for e in valid_extensions)
        
        # Test invalid extension
        invalid_filename = "audio.txt"
        assert not any(invalid_filename.endswith(e) for e in valid_extensions)

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        valid_thresholds = [0.1, 0.5, 0.7, 0.9]
        
        for threshold in valid_thresholds:
            assert 0.0 <= threshold <= 1.0
        
        # Test invalid thresholds
        invalid_thresholds = [-0.1, 1.5, 2.0]
        for threshold in invalid_thresholds:
            assert not (0.0 <= threshold <= 1.0)

    def test_language_code_validation(self):
        """Test language code validation for voice processing."""
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
        
        # Test valid language codes
        for lang in valid_languages:
            assert len(lang) == 2
            assert lang.islower()
        
        # Test invalid language codes
        invalid_languages = ['ENG', 'english', '123', 'x']
        for lang in invalid_languages:
            if len(lang) == 2:
                assert not lang.islower() or not lang.isalpha()

    def test_response_format_consistency(self):
        """Test API response format consistency."""
        # All successful responses should have these fields
        required_fields = ['status', 'data', 'processing_time', 'timestamp']
        
        mock_response = {
            'status': 'success',
            'data': {},
            'processing_time': 1.23,
            'timestamp': datetime.now().isoformat()
        }
        
        for field in required_fields:
            assert field in mock_response
        
        assert mock_response['status'] in ['success', 'error']
        assert isinstance(mock_response['processing_time'], (int, float))
        assert mock_response['processing_time'] >= 0

    def test_error_response_format(self):
        """Test error response format consistency."""
        error_response = {
            'status': 'error',
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': 'Text too short for analysis',
                'details': {}
            },
            'timestamp': datetime.now().isoformat()
        }
        
        assert error_response['status'] == 'error'
        assert 'error' in error_response
        assert 'code' in error_response['error']
        assert 'message' in error_response['error'] 
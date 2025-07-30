"""
Unit tests for data models module.
Tests data models, schemas, and validation.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.data.models import (
    BaseModel,
    User,
    JournalEntry,
    EmotionAnalysis,
    SummaryAnalysis,
    VoiceAnalysis,
    AnalysisResult
)


class TestBaseModel:
    """Test suite for BaseModel class."""

    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        model = BaseModel()

        assert hasattr(model, 'id')
        assert hasattr(model, 'created_at')
        assert hasattr(model, 'updated_at')

    def test_base_model_with_custom_values(self):
        """Test BaseModel with custom values."""
        custom_time = datetime.now()
        model = BaseModel(
            id=1,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert model.id == 1
        assert model.created_at == custom_time
        assert model.updated_at == custom_time


class TestUser:
    """Test suite for User model."""

    def test_user_initialization(self):
        """Test User initialization."""
        user = User(
            username="testuser",
            email="test@example.com"
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True

    def test_user_with_all_fields(self):
        """Test User with all fields."""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is False


class TestJournalEntry:
    """Test suite for JournalEntry model."""

    def test_journal_entry_initialization(self):
        """Test JournalEntry initialization."""
        entry = JournalEntry(
            user_id=1,
            content="Test journal entry"
        )

        assert entry.user_id == 1
        assert entry.content == "Test journal entry"
        assert entry.is_analyzed is False

    def test_journal_entry_with_all_fields(self):
        """Test JournalEntry with all fields."""
        custom_time = datetime.now()
        entry = JournalEntry(
            id=1,
            user_id=1,
            content="Test journal entry",
            is_analyzed=True,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert entry.id == 1
        assert entry.user_id == 1
        assert entry.content == "Test journal entry"
        assert entry.is_analyzed is True
        assert entry.created_at == custom_time
        assert entry.updated_at == custom_time


class TestEmotionAnalysis:
    """Test suite for EmotionAnalysis model."""

    def test_emotion_analysis_initialization(self):
        """Test EmotionAnalysis initialization."""
        analysis = EmotionAnalysis(
            journal_entry_id=1,
            emotions={"happy": 0.8, "sad": 0.2}
        )

        assert analysis.journal_entry_id == 1
        assert analysis.emotions == {"happy": 0.8, "sad": 0.2}
        assert analysis.confidence == 0.0

    def test_emotion_analysis_with_all_fields(self):
        """Test EmotionAnalysis with all fields."""
        custom_time = datetime.now()
        analysis = EmotionAnalysis(
            id=1,
            journal_entry_id=1,
            emotions={"happy": 0.8, "sad": 0.2},
            confidence=0.85,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert analysis.id == 1
        assert analysis.journal_entry_id == 1
        assert analysis.emotions == {"happy": 0.8, "sad": 0.2}
        assert analysis.confidence == 0.85
        assert analysis.created_at == custom_time
        assert analysis.updated_at == custom_time


class TestSummaryAnalysis:
    """Test suite for SummaryAnalysis model."""

    def test_summary_analysis_initialization(self):
        """Test SummaryAnalysis initialization."""
        analysis = SummaryAnalysis(
            journal_entry_id=1,
            summary="This is a test summary"
        )

        assert analysis.journal_entry_id == 1
        assert analysis.summary == "This is a test summary"
        assert analysis.word_count == 0

    def test_summary_analysis_with_all_fields(self):
        """Test SummaryAnalysis with all fields."""
        custom_time = datetime.now()
        analysis = SummaryAnalysis(
            id=1,
            journal_entry_id=1,
            summary="This is a test summary",
            word_count=5,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert analysis.id == 1
        assert analysis.journal_entry_id == 1
        assert analysis.summary == "This is a test summary"
        assert analysis.word_count == 5
        assert analysis.created_at == custom_time
        assert analysis.updated_at == custom_time


class TestVoiceAnalysis:
    """Test suite for VoiceAnalysis model."""

    def test_voice_analysis_initialization(self):
        """Test VoiceAnalysis initialization."""
        analysis = VoiceAnalysis(
            journal_entry_id=1,
            transcription="This is a test transcription"
        )

        assert analysis.journal_entry_id == 1
        assert analysis.transcription == "This is a test transcription"
        assert analysis.confidence == 0.0

    def test_voice_analysis_with_all_fields(self):
        """Test VoiceAnalysis with all fields."""
        custom_time = datetime.now()
        analysis = VoiceAnalysis(
            id=1,
            journal_entry_id=1,
            transcription="This is a test transcription",
            confidence=0.92,
            audio_duration=30.5,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert analysis.id == 1
        assert analysis.journal_entry_id == 1
        assert analysis.transcription == "This is a test transcription"
        assert analysis.confidence == 0.92
        assert analysis.audio_duration == 30.5
        assert analysis.created_at == custom_time
        assert analysis.updated_at == custom_time


class TestAnalysisResult:
    """Test suite for AnalysisResult model."""

    def test_analysis_result_initialization(self):
        """Test AnalysisResult initialization."""
        result = AnalysisResult(
            journal_entry_id=1,
            analysis_type="emotion"
        )

        assert result.journal_entry_id == 1
        assert result.analysis_type == "emotion"
        assert result.status == "pending"

    def test_analysis_result_with_all_fields(self):
        """Test AnalysisResult with all fields."""
        custom_time = datetime.now()
        result = AnalysisResult(
            id=1,
            journal_entry_id=1,
            analysis_type="emotion",
            status="completed",
            result_data={"emotions": {"happy": 0.8}},
            created_at=custom_time,
            updated_at=custom_time
        )

        assert result.id == 1
        assert result.journal_entry_id == 1
        assert result.analysis_type == "emotion"
        assert result.status == "completed"
        assert result.result_data == {"emotions": {"happy": 0.8}}
        assert result.created_at == custom_time
        assert result.updated_at == custom_time

    def test_analysis_result_status_validation(self):
        """Test AnalysisResult status validation."""
        result = AnalysisResult(
            journal_entry_id=1,
            analysis_type="emotion",
            status="invalid_status"
        )

        # Should accept any string for status
        assert result.status == "invalid_status"

#!/usr/bin/env python3
"""Unit tests for data models module.

Tests data models, schemas, and validation.
"""

from datetime import datetime, timezone

from src.data.models import (
    Base,
    Embedding,
    JournalEntry,
    Prediction,
    Tag,
    User,
    VoiceTranscription,
)

TEST_USER_PASSWORD_HASH = "test_hashed_password_123"  # noqa: S105


class TestBase:
    """Test suite for Base model."""

    def test_base_class_exists(self):
        """Test that Base class exists."""
        assert Base is not None


class TestUser:
    """Test suite for User model."""

    def test_user_initialization(self):
        """Test User initialization."""
        user = User(
            email="test@example.com",
            password_hash=TEST_USER_PASSWORD_HASH
        )

        assert user.email == "test@example.com"
        assert user.password_hash == TEST_USER_PASSWORD_HASH

    def test_user_with_all_fields(self):
        """Test User with all fields."""
        custom_time = datetime.now(timezone.utc)
        user = User(
            email="test@example.com",
            password_hash=TEST_USER_PASSWORD_HASH,
            consent_version="1.0",
            consent_given_at=custom_time,
            data_retention_policy="standard"
        )

        assert user.email == "test@example.com"
        assert user.password_hash == TEST_USER_PASSWORD_HASH
        assert user.consent_version == "1.0"
        assert user.consent_given_at == custom_time
        assert user.data_retention_policy == "standard"


class TestJournalEntry:
    """Test suite for JournalEntry model."""

    def test_journal_entry_initialization(self):
        """Test JournalEntry initialization."""
        entry = JournalEntry(
            user_id="test-user-id",
            content="Test journal entry"
        )

        assert entry.user_id == "test-user-id"
        assert entry.content == "Test journal entry"
        assert JournalEntry.__table__.columns['is_private'].default.arg is True

    def test_journal_entry_with_all_fields(self):
        """Test JournalEntry with all fields."""
        custom_time = datetime.now(timezone.utc)
        entry = JournalEntry(
            user_id="test-user-id",
            title="Test Title",
            content="Test journal entry",
            sentiment_score=0.8,
            mood_category="happy",
            is_private=False,
            created_at=custom_time,
            updated_at=custom_time
        )

        assert entry.user_id == "test-user-id"
        assert entry.title == "Test Title"
        assert entry.content == "Test journal entry"
        assert entry.sentiment_score == 0.8
        assert entry.mood_category == "happy"
        assert entry.is_private is False
        assert entry.created_at == custom_time
        assert entry.updated_at == custom_time


class TestEmbedding:
    """Test suite for Embedding model."""

    def test_embedding_initialization(self):
        """Test Embedding initialization."""
        embedding = Embedding(
            journal_entry_id="test-entry-id",
            embedding_vector=[0.1, 0.2, 0.3]
        )

        assert embedding.journal_entry_id == "test-entry-id"
        assert embedding.embedding_vector == [0.1, 0.2, 0.3]

    def test_embedding_with_all_fields(self):
        """Test Embedding with all fields."""
        custom_time = datetime.now(timezone.utc)
        embedding = Embedding(
            journal_entry_id="test-entry-id",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name="test-model",
            created_at=custom_time
        )

        assert embedding.journal_entry_id == "test-entry-id"
        assert embedding.embedding_vector == [0.1, 0.2, 0.3]
        assert embedding.model_name == "test-model"
        assert embedding.created_at == custom_time


class TestPrediction:
    """Test suite for Prediction model."""

    def test_prediction_initialization(self):
        """Test Prediction initialization."""
        prediction = Prediction(
            journal_entry_id="test-entry-id",
            prediction_type="emotion",
            prediction_value={"happy": 0.8, "sad": 0.2}
        )

        assert prediction.journal_entry_id == "test-entry-id"
        assert prediction.prediction_type == "emotion"
        assert prediction.prediction_value == {"happy": 0.8, "sad": 0.2}

    def test_prediction_with_all_fields(self):
        """Test Prediction with all fields."""
        custom_time = datetime.now(timezone.utc)
        prediction = Prediction(
            journal_entry_id="test-entry-id",
            prediction_type="emotion",
            prediction_value={"happy": 0.8, "sad": 0.2},
            confidence_score=0.95,
            model_name="test-model",
            created_at=custom_time
        )

        assert prediction.journal_entry_id == "test-entry-id"
        assert prediction.prediction_type == "emotion"
        assert prediction.prediction_value == {"happy": 0.8, "sad": 0.2}
        assert prediction.confidence_score == 0.95
        assert prediction.model_name == "test-model"
        assert prediction.created_at == custom_time


class TestVoiceTranscription:
    """Test suite for VoiceTranscription model."""

    def test_voice_transcription_initialization(self):
        """Test VoiceTranscription initialization."""
        transcription = VoiceTranscription(
            journal_entry_id="test-entry-id",
            transcription_text="Test transcription"
        )

        assert transcription.journal_entry_id == "test-entry-id"
        assert transcription.transcription_text == "Test transcription"

    def test_voice_transcription_with_all_fields(self):
        """Test VoiceTranscription with all fields."""
        custom_time = datetime.now(timezone.utc)
        transcription = VoiceTranscription(
            journal_entry_id="test-entry-id",
            transcription_text="Test transcription",
            audio_file_path="/path/to/audio.wav",
            confidence_score=0.95,
            model_name="whisper-large",
            processing_time=2.5,
            created_at=custom_time
        )

        assert transcription.journal_entry_id == "test-entry-id"
        assert transcription.transcription_text == "Test transcription"
        assert transcription.audio_file_path == "/path/to/audio.wav"
        assert transcription.confidence_score == 0.95
        assert transcription.model_name == "whisper-large"
        assert transcription.processing_time == 2.5
        assert transcription.created_at == custom_time


class TestTag:
    """Test suite for Tag model."""

    def test_tag_initialization(self):
        """Test Tag initialization."""
        tag = Tag(name="test-tag")

        assert tag.name == "test-tag"

    def test_tag_with_all_fields(self):
        """Test Tag with all fields."""
        custom_time = datetime.now(timezone.utc)
        tag = Tag(
            name="test-tag",
            description="Test tag description",
            color="#FF0000",
            created_at=custom_time
        )

        assert tag.name == "test-tag"
        assert tag.description == "Test tag description"
        assert tag.color == "#FF0000"
        assert tag.created_at == custom_time

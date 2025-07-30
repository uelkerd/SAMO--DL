
"""
Unit tests for data models module.
Tests data models, schemas, and validation.
"""

from datetime import datetime, timezone

from src.data.models import (
    Base,
    User,
    JournalEntry,
    Embedding,
    Prediction,
    VoiceTranscription,
    Tag
)


class TestBase:
    """Test suite for Base class."""

    def test_base_class_exists(self):
        """Test that Base class exists and is properly configured."""
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, '__tablename__') is False  # Base class shouldn't have tablename


class TestUser:
    """Test suite for User model."""

    def test_user_initialization(self):
        """Test User initialization."""
        user = User(
            email="test@example.com",
            password_hash="test_hashed_password"
        )

        assert user.email == "test@example.com"
        assert user.password_hash == "test_hashed_password"

    def test_user_with_all_fields(self):
        """Test User with all fields."""
        custom_time = datetime.now(timezone.utc)
        user = User(
            email="test@example.com",
            password_hash="test_hashed_password",
            consent_version="1.0",
            consent_given_at=custom_time,
            data_retention_policy="standard"
        )

        assert user.email == "test@example.com"
        assert user.password_hash == "test_hashed_password"
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
        # Check that is_private column has the correct default value
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
            entry_id="test-entry-id",
            model_version="bert-base-uncased"
        )

        assert embedding.entry_id == "test-entry-id"
        assert embedding.model_version == "bert-base-uncased"

    def test_embedding_with_all_fields(self):
        """Test Embedding with all fields."""
        custom_time = datetime.now(timezone.utc)
        embedding = Embedding(
            entry_id="test-entry-id",
            model_version="bert-base-uncased",
            created_at=custom_time
        )

        assert embedding.entry_id == "test-entry-id"
        assert embedding.model_version == "bert-base-uncased"
        assert embedding.created_at == custom_time


class TestPrediction:
    """Test suite for Prediction model."""

    def test_prediction_initialization(self):
        """Test Prediction initialization."""
        prediction = Prediction(
            user_id="test-user-id",
            prediction_type="emotion",
            prediction_content={"happy": 0.8, "sad": 0.2}
        )

        assert prediction.user_id == "test-user-id"
        assert prediction.prediction_type == "emotion"
        assert prediction.prediction_content == {"happy": 0.8, "sad": 0.2}
        # Check that is_feedback_given column has the correct default value
        assert Prediction.__table__.columns['is_feedback_given'].default.arg is False

    def test_prediction_with_all_fields(self):
        """Test Prediction with all fields."""
        custom_time = datetime.now(timezone.utc)
        prediction = Prediction(
            user_id="test-user-id",
            prediction_type="emotion",
            prediction_content={"happy": 0.8, "sad": 0.2},
            confidence_score=0.85,
            is_feedback_given=True,
            feedback_rating=5,
            created_at=custom_time
        )

        assert prediction.user_id == "test-user-id"
        assert prediction.prediction_type == "emotion"
        assert prediction.prediction_content == {"happy": 0.8, "sad": 0.2}
        assert prediction.confidence_score == 0.85
        assert prediction.is_feedback_given is True
        assert prediction.feedback_rating == 5
        assert prediction.created_at == custom_time


class TestVoiceTranscription:
    """Test suite for VoiceTranscription model."""

    def test_voice_transcription_initialization(self):
        """Test VoiceTranscription initialization."""
        transcription = VoiceTranscription(
            user_id="test-user-id",
            transcript_text="This is a test transcription"
        )

        assert transcription.user_id == "test-user-id"
        assert transcription.transcript_text == "This is a test transcription"

    def test_voice_transcription_with_all_fields(self):
        """Test VoiceTranscription with all fields."""
        custom_time = datetime.now(timezone.utc)
        transcription = VoiceTranscription(
            user_id="test-user-id",
            audio_file_path="/path/to/audio.wav",
            transcript_text="This is a test transcription",
            duration_seconds=30,
            whisper_model_version="whisper-1",
            confidence_score=0.92,
            created_at=custom_time
        )

        assert transcription.user_id == "test-user-id"
        assert transcription.audio_file_path == "/path/to/audio.wav"
        assert transcription.transcript_text == "This is a test transcription"
        assert transcription.duration_seconds == 30
        assert transcription.whisper_model_version == "whisper-1"
        assert transcription.confidence_score == 0.92
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
            created_at=custom_time
        )

        assert tag.name == "test-tag"
        assert tag.created_at == custom_time

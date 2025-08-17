#!/usr/bin/env python3
"""Database models for the SAMO-DL application.

These models correspond to the tables in the PostgreSQL schema.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class BaseDeclarativeBase:
    """Base class for all SQLAlchemy models."""

    pass


# Junction table for many-to-many relationship between journal entries and tags
journal_entry_tags = Table(
    "journal_entry_tags",
    Base.metadata,
    Column(
        "entry_id",
        UUIDas_uuid=True,
        ForeignKey"journal_entries.id", ondelete="CASCADE",
        primary_key=True,
    ),
    Column(
        "tag_id",
        UUIDas_uuid=True,
        ForeignKey"tags.id", ondelete="CASCADE",
        primary_key=True,
    ),
)


class UserBase:
    """User model representing a system user."""

    __tablename__ = "users"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    email = Column(String255, unique=True, nullable=False)
    password_hash = Column(String255, nullable=False)
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)
    updated_at = Column(DateTimetimezone=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    consent_version = Column(String50)
    consent_given_at = Column(DateTimetimezone=True)
    data_retention_policy = Column(String50, default="standard")

    # Relationships
    journal_entries = relationship(
        "JournalEntry", back_populates="user", cascade="all, delete-orphan"
    )
    predictions = relationship"Prediction", back_populates="user", cascade="all, delete-orphan"
    voice_transcriptions = relationship(
        "VoiceTranscription", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__self -> str:
        return f"<Userid='{self.id}', email='{self.email}'>"


class JournalEntryBase:
    """Journal entry model representing user's journal entries."""

    __tablename__ = "journal_entries"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDas_uuid=True, ForeignKey"users.id", ondelete="CASCADE", nullable=False)
    title = Column(String255)
    content = ColumnText, nullable=False
    encrypted_content = ColumnLargeBinary
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)
    updated_at = Column(DateTimetimezone=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    sentiment_score = ColumnFloat
    mood_category = Column(String50)
    is_private = ColumnBoolean, default=True

    # Relationships
    user = relationship"User", back_populates="journal_entries"
    embeddings = relationship(
        "Embedding", back_populates="journal_entry", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="journal_entry", cascade="all, delete-orphan"
    )
    voice_transcriptions = relationship(
        "VoiceTranscription", back_populates="journal_entry", cascade="all, delete-orphan"
    )
    tags = relationship"Tag", secondary=journal_entry_tags, back_populates="entries"

    def __repr__self -> str:
        return f"<JournalEntryid='{self.id}', title='{self.title}'>"


class EmbeddingBase:
    """Embedding model storing vector embeddings for journal entries."""

    __tablename__ = "embeddings"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column(
        UUIDas_uuid=True,
        ForeignKey"journal_entries.id", ondelete="CASCADE",
        nullable=False,
    )
    embedding_vector = Column(Vector768)  # 768 dimensions for BERT-base
    model_name = Column(String100, nullable=False)
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)

    # Relationships
    journal_entry = relationship"JournalEntry", back_populates="embeddings"

    def __repr__self -> str:
        return f"<Embeddingid='{self.id}', journal_entry_id='{self.journal_entry_id}', model_name='{self.model_name}'>"


class PredictionBase:
    """Prediction model storing AI-generated predictions about user mood, topics, etc."""

    __tablename__ = "predictions"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column(UUIDas_uuid=True, ForeignKey"journal_entries.id", ondelete="CASCADE", nullable=False)
    user_id = Column(UUIDas_uuid=True, ForeignKey"users.id", ondelete="CASCADE", nullable=False)
    prediction_type = Column(String100, nullable=False)
    prediction_value = ColumnJSONB, nullable=False
    confidence_score = ColumnFloat
    model_name = Column(String100)
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)
    is_feedback_given = ColumnBoolean, default=False
    feedback_rating = ColumnInteger

    # Relationships
    user = relationship"User", back_populates="predictions"
    journal_entry = relationship"JournalEntry", back_populates="predictions"

    def __repr__self -> str:
        return f"<Predictionid='{self.id}', prediction_type='{self.prediction_type}', confidence_score={self.confidence_score}>"


class VoiceTranscriptionBase:
    """Voice transcription model storing transcribed audio from users."""

    __tablename__ = "voice_transcriptions"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column(UUIDas_uuid=True, ForeignKey"journal_entries.id", ondelete="CASCADE", nullable=False)
    user_id = Column(UUIDas_uuid=True, ForeignKey"users.id", ondelete="CASCADE", nullable=False)
    audio_file_path = Column(String255)
    transcription_text = ColumnText, nullable=False
    duration_seconds = ColumnInteger
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)
    model_name = Column(String100)
    confidence_score = ColumnFloat
    processing_time = ColumnFloat

    # Relationships
    user = relationship"User", back_populates="voice_transcriptions"
    journal_entry = relationship"JournalEntry", back_populates="voice_transcriptions"

    def __repr__self -> str:
        return f"<VoiceTranscriptionid='{self.id}', transcription_text='{self.transcription_text[:50]}...'>"


class TagBase:
    """Tag model for categorizing journal entries."""

    __tablename__ = "tags"

    id = Column(UUIDas_uuid=True, primary_key=True, default=uuid.uuid4)
    name = Column(String100, unique=True, nullable=False)
    description = ColumnText
    color = Column(String7)  # Hex color code
    created_at = Column(DateTimetimezone=True, default=datetime.utcnow)

    # Relationships
    entries = relationship"JournalEntry", secondary=journal_entry_tags, back_populates="tags"

    def __repr__self -> str:
        return f"<Tagid='{self.id}', name='{self.name}'>"

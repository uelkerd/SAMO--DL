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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Junction table for many-to-many relationship between journal entries and tags
journal_entry_tags = Table(
    "journal_entry_tags",
    Base.metadata,
    Column(
        "entry_id",
        UUID(as_uuid=True),
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "tag_id",
        UUID(as_uuid=True),
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(Base):
    """User model representing a system user."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    consent_version = Column(String(50))
    consent_given_at = Column(DateTime(timezone=True))
    data_retention_policy = Column(String(50), default="standard")

    # Relationships
    journal_entries = relationship(
        "JournalEntry", back_populates="user", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )
    voice_transcriptions = relationship(
        "VoiceTranscription", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id='{self.id}', email='{self.email}')>"


class JournalEntry(Base):
    """Journal entry model representing user's journal entries."""

    __tablename__ = "journal_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title = Column(String(255))
    content = Column(Text, nullable=False)
    encrypted_content = Column(LargeBinary)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    sentiment_score = Column(Float)
    mood_category = Column(String(50))
    is_private = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="journal_entries")
    embeddings = relationship(
        "Embedding", back_populates="journal_entry", cascade="all, delete-orphan"
    )
    tags = relationship("Tag", secondary=journal_entry_tags, back_populates="entries")

    def __repr__(self) -> str:
        return f"<JournalEntry(id='{self.id}', title='{self.title}', created_at='{self.created_at}')>"


class Embedding(Base):
    """Embedding model storing vector embeddings for journal entries."""

    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entry_id = Column(
        UUID(as_uuid=True),
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_version = Column(String(100), nullable=False)
    embedding = Column(Vector(768))  # 768 dimensions for BERT-base
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    journal_entry = relationship("JournalEntry", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<Embedding(id='{self.id}', entry_id='{self.entry_id}', model_version='{self.model_version}')>"


class Prediction(Base):
    """Prediction model storing AI-generated predictions about user mood, topics, etc."""

    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    prediction_type = Column(String(100), nullable=False)
    prediction_content = Column(JSONB, nullable=False)
    confidence_score = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_feedback_given = Column(Boolean, default=False)
    feedback_rating = Column(Integer)

    # Relationships
    user = relationship("User", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction(id='{self.id}', type='{self.prediction_type}', user_id='{self.user_id}')>"


class VoiceTranscription(Base):
    """Voice transcription model storing transcribed audio from users."""

    __tablename__ = "voice_transcriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    audio_file_path = Column(String(255))
    transcript_text = Column(Text, nullable=False)
    duration_seconds = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    whisper_model_version = Column(String(100))
    confidence_score = Column(Float)

    # Relationships
    user = relationship("User", back_populates="voice_transcriptions")

    def __repr__(self) -> str:
        return f"<VoiceTranscription(id='{self.id}', user_id='{self.user_id}', duration={self.duration_seconds}s)>"


class Tag(Base):
    """Tag model for categorizing journal entries."""

    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    entries = relationship(
        "JournalEntry", secondary=journal_entry_tags, back_populates="tags"
    )

    def __repr__(self) -> str:
        return f"<Tag(id='{self.id}', name='{self.name}')>"

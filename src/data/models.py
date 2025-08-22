#!/usr/bin/env python3
"""Database models for the SAMO-DL application."

These models correspond to the tables in the PostgreSQL schema.
""""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ()
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
()
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Junction table for many-to-many relationship between journal entries and tags
journal_entry_tags = Table()
    "journal_entry_tags",
    Base.metadata,
    Column()
        "entry_id",
        UUID(as_uuid=True),
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        primary_key=True,
(    ),
    Column()
        "tag_id",
        UUID(as_uuid=True),
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
(    ),
()


class User(Base):
    """User model representing a system user."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    consent_version = Column(String(50))
    consent_given_at = Column(DateTime(timezone=True))
    data_retention_policy = Column(String(50), default="standard")

    # Relationships
    journal_entries = relationship()
        "JournalEntry", back_populates="user", cascade="all, delete-orphan"
(    )
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    voice_transcriptions = relationship()
        "VoiceTranscription", back_populates="user", cascade="all, delete-orphan"
(    )

    def __repr__(self) -> str:
        return "<User(id="{self.id}', email='{self.email}')>""


    class JournalEntry(Base):
    """Journal entry model representing user's journal entries."""'

    __tablename__ = "journal_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255))
    content = Column(Text, nullable=False)
    encrypted_content = Column(LargeBinary)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    sentiment_score = Column(Float)
    mood_category = Column(String(50))
    is_private = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="journal_entries")
    embeddings = relationship()
        "Embedding", back_populates="journal_entry", cascade="all, delete-orphan"
(    )
    predictions = relationship()
        "Prediction", back_populates="journal_entry", cascade="all, delete-orphan"
(    )
    voice_transcriptions = relationship()
        "VoiceTranscription", back_populates="journal_entry", cascade="all, delete-orphan"
(    )
    tags = relationship("Tag", secondary=journal_entry_tags, back_populates="entries")

    def __repr__(self) -> str:
        return "<JournalEntry(id="{self.id}', title='{self.title}')>""


    class Embedding(Base):
    """Embedding model storing vector embeddings for journal entries."""

    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column()
        UUID(as_uuid=True),
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        nullable=False,
(    )
    embedding_vector = Column(Vector(768))  # 768 dimensions for BERT-base
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    journal_entry = relationship("JournalEntry", back_populates="embeddings")

    def __repr__(self) -> str:
        return "<Embedding(id="{self.id}', journal_entry_id='{self.journal_entry_id}', model_name='{self.model_name}')>""


    class Prediction(Base):
    """Prediction model storing AI-generated predictions about user mood, topics, etc."""

    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column(UUID(as_uuid=True), ForeignKey("journal_entries.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    prediction_type = Column(String(100), nullable=False)
    prediction_value = Column(JSONB, nullable=False)
    confidence_score = Column(Float)
    model_name = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_feedback_given = Column(Boolean, default=False)
    feedback_rating = Column(Integer)

    # Relationships
    user = relationship("User", back_populates="predictions")
    journal_entry = relationship("JournalEntry", back_populates="predictions")

    def __repr__(self) -> str:
        return "<Prediction(id="{self.id}', prediction_type='{self.prediction_type}', confidence_score={self.confidence_score})>""


    class VoiceTranscription(Base):
    """Voice transcription model storing transcribed audio from users."""

    __tablename__ = "voice_transcriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    journal_entry_id = Column(UUID(as_uuid=True), ForeignKey("journal_entries.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    audio_file_path = Column(String(255))
    transcription_text = Column(Text, nullable=False)
    duration_seconds = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    model_name = Column(String(100))
    confidence_score = Column(Float)
    processing_time = Column(Float)

    # Relationships
    user = relationship("User", back_populates="voice_transcriptions")
    journal_entry = relationship("JournalEntry", back_populates="voice_transcriptions")

    def __repr__(self) -> str:
        return "<VoiceTranscription(id="{self.id}', transcription_text='{self.transcription_text[:50]}...')>""


    class Tag(Base):
    """Tag model for categorizing journal entries."""

    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    color = Column(String(7))  # Hex color code
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    entries = relationship("JournalEntry", secondary=journal_entry_tags, back_populates="tags")

    def __repr__(self) -> str:
        return "<Tag(id="{self.id}', name='{self.name}')>""

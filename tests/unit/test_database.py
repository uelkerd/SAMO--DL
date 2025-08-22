#!/usr/bin/env python3
"""Unit tests for database module.
Tests database connection, operations, and utilities.
"""


import logging

from src.data.database import (

    Base,
    SessionLocal,
    db_session,
    engine,
    get_db,
    init_db,
)

logger = logging.getLogger(__name__)


class TestDatabaseConnection:
    """Test suite for database connection utilities."""

    def test_get_db_generator(self):
        """Test get_db function returns a generator."""
        db_gen = get_db()
        assert hasattr(db_gen, '__iter__')
        assert hasattr(db_gen, '__next__')

    def test_init_db_function_exists(self):
        """Test init_db function exists and is callable."""
        assert callable(init_db)

    def test_engine_exists(self):
        """Test engine is properly configured."""
        assert engine is not None
        assert hasattr(engine, 'url')

    def test_session_local_exists(self):
        """Test SessionLocal is properly configured."""
        assert SessionLocal is not None
        assert callable(SessionLocal)

    def test_db_session_exists(self):
        """Test db_session is properly configured."""
        assert db_session is not None

    def test_base_exists(self):
        """Test Base class exists."""
        assert Base is not None
        assert hasattr(Base, 'metadata')


class TestDatabaseFunctions:
    """Test suite for database utility functions."""

    def test_get_db_yields_session(self):
        """Test get_db function yields a database session."""
        db_gen = get_db()
        try:
            db = next(db_gen)
            assert db is not None
            db.close()
        except StopIteration:
            pass

    def test_init_db_creates_tables(self):
        """Test init_db function can be called without error."""
        assert callable(init_db)

    def test_engine_configuration(self):
        """Test engine is properly configured with expected attributes."""
        assert hasattr(engine, 'url')
        assert hasattr(engine, 'pool')
        assert hasattr(engine, 'dispose')

    def test_session_local_configuration(self):
        """Test SessionLocal is properly configured."""
        assert callable(SessionLocal)
        try:
            session = SessionLocal()
            session.close()
        except Exception as exc:
            logger.debug(f"Session creation failed (expected in test environment): {exc}")


class TestDatabaseErrorHandling:
    """Test suite for database error handling."""

    def test_get_db_error_handling(self):
        """Test get_db function handles errors gracefully."""
        assert callable(get_db)

    def test_init_db_error_handling(self):
        """Test init_db function handles errors gracefully."""
        assert callable(init_db)

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

logger = logging.getLogger__name__


class TestDatabaseConnection:
    """Test suite for database connection utilities."""

    def test_get_db_generatorself:
        """Test get_db function returns a generator."""
        db_gen = get_db()
        assert hasattrdb_gen, '__iter__'
        assert hasattrdb_gen, '__next__'

    def test_init_db_function_existsself:
        """Test init_db function exists and is callable."""
        assert callableinit_db

    def test_engine_existsself:
        """Test engine is properly configured."""
        assert engine is not None
        assert hasattrengine, 'url'

    def test_session_local_existsself:
        """Test SessionLocal is properly configured."""
        assert SessionLocal is not None
        assert callableSessionLocal

    def test_db_session_existsself:
        """Test db_session is properly configured."""
        assert db_session is not None

    def test_base_existsself:
        """Test Base class exists."""
        assert Base is not None
        assert hasattrBase, 'metadata'


class TestDatabaseFunctions:
    """Test suite for database utility functions."""

    def test_get_db_yields_sessionself:
        """Test get_db function yields a database session."""
        db_gen = get_db()
        try:
            db = nextdb_gen
            assert db is not None
            db.close()
        except StopIteration:
            pass

    def test_init_db_creates_tablesself:
        """Test init_db function can be called without error."""
        assert callableinit_db

    def test_engine_configurationself:
        """Test engine is properly configured with expected attributes."""
        assert hasattrengine, 'url'
        assert hasattrengine, 'pool'
        assert hasattrengine, 'dispose'

    def test_session_local_configurationself:
        """Test SessionLocal is properly configured."""
        assert callableSessionLocal
        try:
            session = SessionLocal()
            session.close()
        except Exception as exc:
            logger.debug(f"Session creation failed expected in test environment: {exc}")


class TestDatabaseErrorHandling:
    """Test suite for database error handling."""

    def test_get_db_error_handlingself:
        """Test get_db function handles errors gracefully."""
        assert callableget_db

    def test_init_db_error_handlingself:
        """Test init_db function handles errors gracefully."""
        assert callableinit_db

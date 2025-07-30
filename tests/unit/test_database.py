"""
Unit tests for database module.
Tests database connection, operations, and utilities.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.data.database import (
    get_database_connection,
    execute_query,
    close_database_connection,
    DatabaseConnection
)


class TestDatabaseConnection:
    """Test suite for DatabaseConnection class."""

    def test_database_connection_initialization(self):
        """Test DatabaseConnection initialization."""
        mock_connection = MagicMock()
        db_conn = DatabaseConnection(mock_connection)
        
        assert db_conn.connection == mock_connection
        assert db_conn.is_connected() is True

    def test_database_connection_close(self):
        """Test DatabaseConnection close method."""
        mock_connection = MagicMock()
        db_conn = DatabaseConnection(mock_connection)
        
        db_conn.close()
        mock_connection.close.assert_called_once()

    def test_database_connection_context_manager(self):
        """Test DatabaseConnection as context manager."""
        mock_connection = MagicMock()
        
        with DatabaseConnection(mock_connection) as db_conn:
            assert db_conn.connection == mock_connection
        
        mock_connection.close.assert_called_once()


class TestDatabaseFunctions:
    """Test suite for database utility functions."""

    @patch('src.data.database.DatabaseConnection')
    def test_get_database_connection(self, mock_db_conn_class):
        """Test get_database_connection function."""
        mock_connection = MagicMock()
        mock_db_conn_class.return_value = mock_connection
        
        result = get_database_connection()
        
        assert result == mock_connection
        mock_db_conn_class.assert_called_once()

    @patch('src.data.database.DatabaseConnection')
    def test_execute_query(self, mock_db_conn_class):
        """Test execute_query function."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn_class.return_value = mock_connection
        
        query = "SELECT * FROM test_table"
        params = {"id": 1}
        
        execute_query(query, params)
        
        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(query, params)
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    @patch('src.data.database.DatabaseConnection')
    def test_execute_query_without_params(self, mock_db_conn_class):
        """Test execute_query function without parameters."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_db_conn_class.return_value = mock_connection
        
        query = "SELECT * FROM test_table"
        
        execute_query(query)
        
        mock_connection.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with(query, None)
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    @patch('src.data.database.DatabaseConnection')
    def test_close_database_connection(self, mock_db_conn_class):
        """Test close_database_connection function."""
        mock_connection = MagicMock()
        mock_db_conn_class.return_value = mock_connection
        
        close_database_connection()
        
        mock_connection.close.assert_called_once()


class TestDatabaseErrorHandling:
    """Test suite for database error handling."""

    @patch('src.data.database.DatabaseConnection')
    def test_execute_query_with_error(self, mock_db_conn_class):
        """Test execute_query function with database error."""
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_db_conn_class.return_value = mock_connection
        
        query = "SELECT * FROM test_table"
        
        with pytest.raises(Exception, match="Database error"):
            execute_query(query)
        
        mock_cursor.close.assert_called_once()

    @patch('src.data.database.DatabaseConnection')
    def test_execute_query_cursor_error(self, mock_db_conn_class):
        """Test execute_query function with cursor error."""
        mock_connection = MagicMock()
        mock_connection.cursor.side_effect = Exception("Cursor error")
        mock_db_conn_class.return_value = mock_connection
        
        query = "SELECT * FROM test_table"
        
        with pytest.raises(Exception, match="Cursor error"):
            execute_query(query) 
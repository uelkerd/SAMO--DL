#!/usr/bin/env python3
"""
Unit tests for data validation functionality.
"""

import pandas as pd

from src.data.validation import DataValidator, validate_text_input


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()

        assert hasattr(validator, 'check_missing_values')
        assert hasattr(validator, 'check_data_types')
        assert hasattr(validator, 'check_text_quality')
        assert hasattr(validator, 'validate_journal_entries')

    def test_check_missing_values(self):
        """Test check_missing_values method."""
        validator = DataValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            'user_id': [1, 2, None, 4],
            'content': ['text1', 'text2', 'text3', None],
            'optional_field': ['a', 'b', 'c', 'd']
        })

        result = validator.check_missing_values(df, required_columns=['user_id', 'content'])

        assert 'user_id' in result
        assert 'content' in result
        assert result['user_id'] == 25.0  # 1 out of 4 is missing
        assert result['content'] == 25.0  # 1 out of 4 is missing

    def test_check_data_types(self):
        """Test check_data_types method."""
        validator = DataValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'content': ['text1', 'text2', 'text3', 'text4'],
            'is_private': [True, False, True, False]
        })

        expected_types = {
            'user_id': int,
            'content': str,
            'is_private': bool
        }

        result = validator.check_data_types(df, expected_types)

        assert result['user_id'] is True
        assert result['content'] is True
        assert result['is_private'] is True

    def test_check_text_quality(self):
        """Test check_text_quality method."""
        validator = DataValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            'content': ['This is a test', '', '   ', 'Another test with more words']
        })

        result = validator.check_text_quality(df, text_column='content')

        assert 'text_length' in result.columns
        assert 'word_count' in result.columns
        assert 'is_empty' in result.columns
        assert 'is_very_short' in result.columns

    def test_validate_journal_entries(self):
        """Test validate_journal_entries method."""
        validator = DataValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4],
            'content': ['text1', 'text2', 'text3', 'text4'],
            'title': ['title1', 'title2', 'title3', 'title4'],
            'created_at': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'is_private': [True, False, True, False]
        })

        required_columns = ['user_id', 'content']
        expected_types = {
            'user_id': int,
            'content': str,
            'title': str,
            'created_at': 'datetime64[ns]',
            'is_private': bool
        }

        result = validator.validate_journal_entries(df, required_columns, expected_types)

        assert 'missing_values' in result
        assert 'data_types' in result
        assert 'text_quality' in result
        assert result['missing_values']['user_id'] == 0.0
        assert result['missing_values']['content'] == 0.0


class TestValidateTextInput:
    """Test suite for validate_text_input function."""

    def test_validate_text_input_valid(self):
        """Test validate_text_input with valid input."""
        text = "This is a valid text input with reasonable length."
        result = validate_text_input(text)
        assert result['is_valid'] is True
        assert result['error'] is None

    def test_validate_text_input_empty(self):
        """Test validate_text_input with empty string."""
        text = ""
        result = validate_text_input(text)
        assert result['is_valid'] is False
        assert "empty" in result['error'].lower()

    def test_validate_text_input_none(self):
        """Test validate_text_input with None."""
        result = validate_text_input(None)
        assert result['is_valid'] is False
        assert "none" in result['error'].lower()

    def test_validate_text_input_too_short(self):
        """Test validate_text_input with too short text."""
        text = "Hi"
        result = validate_text_input(text, min_length=10)
        assert result['is_valid'] is False
        assert "short" in result['error'].lower()

    def test_validate_text_input_too_long(self):
        """Test validate_text_input with too long text."""
        text = "A" * 10001  # 10,001 characters
        result = validate_text_input(text, max_length=10000)
        assert result['is_valid'] is False
        assert "long" in result['error'].lower()

    def test_validate_text_input_invalid_characters(self):
        """Test validate_text_input with invalid characters."""
        text = "Text with invalid chars: \x00\x01\x02"
        result = validate_text_input(text)
        assert result['is_valid'] is False
        assert "invalid" in result['error'].lower()

    def test_validate_text_input_whitespace_only(self):
        """Test validate_text_input with whitespace-only text."""
        text = "   \n\t   "
        result = validate_text_input(text)
        assert result['is_valid'] is False
        assert "whitespace" in result['error'].lower()

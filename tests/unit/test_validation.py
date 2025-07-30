from src.data.validation import (
        import pandas as pd
        import pandas as pd
        import pandas as pd
        import pandas as pd

"""
Unit tests for validation module.
Tests data validation, schema validation, and input validation.
"""


    validate_text_input,
    DataValidator
)


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
            'is_private': bool
        }

        validation_passed, validated_df = validator.validate_journal_entries(
            df, required_columns, expected_types
        )

        assert validation_passed is True
        assert 'text_length' in validated_df.columns
        assert 'word_count' in validated_df.columns


class TestValidateTextInput:
    """Test suite for validate_text_input function."""

    def test_validate_text_input_valid(self):
        """Test validate_text_input with valid input."""
        text = "This is a valid text input for testing purposes."

        result, message = validate_text_input(text)
        assert result is True
        assert message == ""

    def test_validate_text_input_empty(self):
        """Test validate_text_input with empty text."""
        result, message = validate_text_input("")
        assert result is False
        assert "at least 1 characters" in message

    def test_validate_text_input_none(self):
        """Test validate_text_input with None."""
        result, message = validate_text_input(None)
        assert result is False
        assert "must be a string" in message

    def test_validate_text_input_too_short(self):
        """Test validate_text_input with too short text."""
        result, message = validate_text_input("Short", min_length=10)
        assert result is False
        assert "at least 10 characters" in message

    def test_validate_text_input_too_long(self):
        """Test validate_text_input with too long text."""
        long_text = "A" * 10001
        result, message = validate_text_input(long_text)
        assert result is False
        assert "no more than 10000 characters" in message

    def test_validate_text_input_invalid_characters(self):
        """Test validate_text_input with invalid characters."""
        result, message = validate_text_input("Text with <script>alert('xss')</script>")
        assert result is False
        assert "potentially harmful content" in message

    def test_validate_text_input_whitespace_only(self):
        """Test validate_text_input with whitespace only."""
        result, message = validate_text_input("   \n\t   ", min_length=5)
        assert result is False
        assert "at least 5 characters" in message

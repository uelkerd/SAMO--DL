"""
Enhanced tests for data validation module to increase coverage.
"""

import pandas as pd
from src.data.validation import DataValidator, validate_text_input


class TestDataValidatorEnhanced:
    """Enhanced test suite for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()

        # Create test data that matches the expected schema
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'user_id': [1, 2, 3, 4, 5],  # No missing values
            'title': ['Entry 1', 'Entry 2', 'Entry 3', 'Entry 4', 'Entry 5'],
            'content': ['Hello world', 'Test entry', 'Another test', 'Valid content', 'Good content'],
            'created_at': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'is_private': [False, True, False, True, False]
        })

    def test_check_missing_values_basic(self):
        """Test basic missing values check."""
        missing_stats = self.validator.check_missing_values(self.test_df)

        assert isinstance(missing_stats, dict)
        assert 'user_id' in missing_stats
        assert 'content' in missing_stats
        assert missing_stats['user_id'] == 0.0  # No missing values
        assert missing_stats['content'] == 0.0   # No missing content

    def test_check_missing_values_with_required_columns(self):
        """Test missing values check with required columns."""
        missing_stats = self.validator.check_missing_values(
            self.test_df,
            required_columns=['user_id', 'content']
        )

        assert missing_stats['user_id'] == 0.0
        assert missing_stats['content'] == 0.0

    def test_check_data_types_basic(self):
        """Test data type checking."""
        expected_types = {
            'user_id': int,
            'content': str,
            'emotion_score': float
        }

        type_results = self.validator.check_data_types(self.test_df, expected_types)

        assert isinstance(type_results, dict)
        assert 'user_id' in type_results
        assert 'content' in type_results
        assert 'emotion_score' in type_results

    def test_check_data_types_with_missing_column(self):
        """Test data type checking with missing column."""
        expected_types = {
            'user_id': int,
            'nonexistent_column': str
        }

        type_results = self.validator.check_data_types(self.test_df, expected_types)

        assert type_results['nonexistent_column'] is False

    def test_check_text_quality_basic(self):
        """Test text quality checking."""
        result_df = self.validator.check_text_quality(self.test_df, 'content')

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(self.test_df)
        assert 'text_length' in result_df.columns
        assert 'word_count' in result_df.columns

    def test_check_text_quality_with_empty_text(self):
        """Test text quality checking with empty text."""
        empty_df = pd.DataFrame({
            'content': ['', '   ', 'valid text']
        })

        result_df = self.validator.check_text_quality(empty_df, 'content')

        assert result_df.iloc[0]['text_length'] == 0  # Empty string
        assert result_df.iloc[1]['text_length'] == 3  # Three spaces
        assert result_df.iloc[2]['text_length'] > 0

    def test_validate_journal_entries_basic(self):
        """Test journal entries validation."""
        results = self.validator.validate_journal_entries(self.test_df)

        assert isinstance(results, dict)
        assert 'is_valid' in results
        assert 'validated_df' in results
        assert 'missing_values' in results

        # Assert the expected value of 'is_valid'
        assert isinstance(results['is_valid'], bool)
        # For this test data, it should be valid
        assert results['is_valid'] is True

        # Assert the structure/type of missing_values
        assert isinstance(results['missing_values'], dict)

        # Assert the structure/type of validated_df
        import pandas as pd
        assert isinstance(results['validated_df'], pd.DataFrame)
        # Should have the original columns plus text quality columns
        original_columns = list(self.test_df.columns)
        quality_columns = ['text_length', 'word_count', 'is_empty', 'is_very_short']
        expected_columns = original_columns + quality_columns
        assert all(col in results['validated_df'].columns for col in expected_columns)
        # Should have the same number of rows
        assert len(results['validated_df']) == len(self.test_df)

    def test_validate_journal_entries_with_required_columns(self):
        """Test journal entries validation with required columns."""
        results = self.validator.validate_journal_entries(
            self.test_df,
            required_columns=['user_id', 'content']
        )

        assert isinstance(results, dict)
        assert 'is_valid' in results

    def test_validate_journal_entries_with_expected_types(self):
        """Test journal entries validation with expected types."""
        expected_types = {
            'user_id': int,
            'content': str,
            'emotion_score': float
        }

        results = self.validator.validate_journal_entries(
            self.test_df,
            expected_types=expected_types
        )

        assert isinstance(results, dict)
        assert 'is_valid' in results


class TestValidateTextInputEnhanced:
    """Enhanced test suite for validate_text_input function."""

    def test_validate_text_input_valid(self):
        """Test valid text input."""
        result = validate_text_input("This is a valid text input")

        assert isinstance(result, dict)
        assert result['is_valid'] is True
        assert 'error' in result

    def test_validate_text_input_too_short(self):
        """Test text input that's too short."""
        result = validate_text_input("", min_length=5)

        assert isinstance(result, dict)
        assert result['is_valid'] is False
        assert 'error' in result

    def test_validate_text_input_too_long(self):
        """Test text input that's too long."""
        long_text = "x" * 10001
        result = validate_text_input(long_text, max_length=10000)

        assert isinstance(result, dict)
        assert result['is_valid'] is False
        assert 'error' in result

    def test_validate_text_input_custom_lengths(self):
        """Test text input with custom length constraints."""
        result = validate_text_input("Test", min_length=3, max_length=10)

        assert isinstance(result, dict)
        assert result['is_valid'] is True

    def test_validate_text_input_edge_cases(self):
        """Test text input edge cases."""
        # Test with whitespace
        result = validate_text_input("   ", min_length=1)
        assert result['is_valid'] is False

        # Test with single character
        result = validate_text_input("a", min_length=1, max_length=1)
        assert result['is_valid'] is True

        # Test with exact max length
        exact_text = "x" * 100
        result = validate_text_input(exact_text, max_length=100)
        assert result['is_valid'] is True

    def test_validate_text_input_invalid_types(self):
        """Test text input with invalid types."""
        # Test with None
        result = validate_text_input(None)
        assert result['is_valid'] is False

        # Test with non-string
        result = validate_text_input(123)
        assert result['is_valid'] is False

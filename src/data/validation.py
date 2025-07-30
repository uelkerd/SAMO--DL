            # Check if types match (with some flexibility for numeric types)
            # Get actual type
        # Check data types
        # Check for missing values
        # Check for required columns
        # Check text quality
        # Identify potentially problematic entries
        # Log summary of issues
        # Make a copy to avoid modifying the original
        # Text length
        # Word count
    # Check for potentially harmful content (basic check)
# Configure logging
# G004: Logging f-strings temporarily allowed for development
from typing import Optional
import logging
import pandas as pd



logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality checks for journal entries."""

    def __init__(self) -> None:
        """Initialize data validator."""

    def check_missing_values(
        self, df: pd.DataFrame, required_columns: Optional[list[str]] = None
    ) -> dict[str, float]:
        """Check for missing values in DataFrame.

        Args:
            df: DataFrame to check
            required_columns: List of columns that must not have missing values

        Returns:
            Dictionary with column names and percentage of missing values

        """
        if required_columns is None:
            required_columns = ["user_id", "content"]

        missing_stats = {}
        total_rows = len(df)

        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            missing_stats[column] = missing_percent

            if column in required_columns and missing_count > 0:
                logger.warning(
                    "Required column '{column}' has {missing_count} missing values ({missing_percent:.2f}%)"
                )

        return missing_stats

    def check_data_types(
        self, df: pd.DataFrame, expected_types: dict[str, type]
    ) -> dict[str, bool]:
        """Check if columns have expected data types.

        Args:
            df: DataFrame to check
            expected_types: Dictionary mapping column names to expected types

        Returns:
            Dictionary with column names and whether they match expected types

        """
        type_check_results = {}

        for column, expected_type in expected_types.items():
            if column not in df.columns:
                logger.warning(
                    "Column '{column}' not found in DataFrame",
                    extra={"format_args": True},
                )
                type_check_results[column] = False
                continue

            actual_type = df[column].dtype

            if (expected_type in (int, float) and pd.api.types.is_numeric_dtype(actual_type)) or (
                expected_type is str and pd.api.types.is_string_dtype(actual_type)
            ):
                type_check_results[column] = True
            else:
                is_match = actual_type == expected_type
                if not is_match:
                    logger.warning(
                        "Column '{column}' has type {actual_type}, expected {expected_type}"
                    )
                type_check_results[column] = is_match

        return type_check_results

    def check_text_quality(self, df: pd.DataFrame, text_column: str = "content") -> pd.DataFrame:
        """Check text quality metrics.

        Args:
            df: DataFrame to check
            text_column: Name of column containing text data

        Returns:
            DataFrame with text quality metrics

        """
        if text_column not in df.columns:
            logger.error(
                "Text column '{text_column}' not found in DataFrame",
                extra={"format_args": True},
            )
            return df

        result_df = df.copy()

        result_df["text_length"] = result_df[text_column].astype(str).apply(len)

        result_df["word_count"] = result_df[text_column].astype(str).apply(lambda x: len(x.split()))

        result_df["is_empty"] = (
            result_df[text_column].astype(str).apply(lambda x: len(x.strip()) == 0)
        )
        result_df["is_very_short"] = result_df["word_count"] < 5

        empty_count = result_df["is_empty"].sum()
        very_short_count = result_df["is_very_short"].sum()

        if empty_count > 0:
            logger.warning("Found {empty_count} empty entries in '{text_column}' column")

        if very_short_count > 0:
            logger.warning(
                "Found {very_short_count} very short entries (< 5 words) in '{text_column}' column"
            )

        return result_df

    def validate_journal_entries(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
        expected_types: Optional[dict[str, type]] = None,
    ) -> tuple[bool, pd.DataFrame]:
        """Perform comprehensive validation on journal entries DataFrame.

        Args:
            df: DataFrame containing journal entries
            required_columns: List of columns that must not have missing values
            expected_types: Dictionary mapping column names to expected types

        Returns:
            Tuple of (validation_passed, validated_df)

        """
        if required_columns is None:
            required_columns = ["user_id", "content"]

        if expected_types is None:
            expected_types = {
                "id": int,
                "user_id": int,
                "title": str,
                "content": str,
                "created_at": pd.Timestamp,
                "is_private": bool,
            }

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                "Required columns missing: {missing_columns}",
                extra={"format_args": True},
            )
            return False, df

        missing_stats = self.check_missing_values(df, required_columns)
        has_missing_required = any(missing_stats.get(col, 0) > 0 for col in required_columns)

        type_check_results = self.check_data_types(df, expected_types)
        has_type_mismatch = not all(type_check_results.values())

        df_with_quality = self.check_text_quality(df)

        validation_passed = not (has_missing_required or has_type_mismatch)

        if validation_passed:
            logger.info("Data validation passed")
        else:
            logger.warning("Data validation failed")

        return validation_passed, df_with_quality


def validate_text_input(input_text: str, min_length: int = 1, max_length: int = 10000) -> tuple[bool, str]:
    """Validate text input for journal entries.

    Args:
        input_text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(input_text, str):
        return False, "Input must be a string"

    if len(input_text.strip()) < min_length:
        return False, f"Text must be at least {min_length} characters long"

    if len(input_text) > max_length:
        return False, f"Text must be no more than {max_length} characters long"

    harmful_patterns = ["<script>", "javascript:", "data:text/html"]
    for pattern in harmful_patterns:
        if pattern.lower() in input_text.lower():
            return False, f"Text contains potentially harmful content: {pattern}"

    return True, ""

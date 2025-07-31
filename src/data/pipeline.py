#!/usr/bin/env python3
"""Data Pipeline for SAMO Deep Learning.

This module orchestrates the complete data processing pipeline for journal entries,
including loading, validation, preprocessing, feature engineering, and embedding generation.

Key Features:
- Multi-source data loading (database, JSON, CSV)
- Comprehensive data validation
- Text preprocessing and feature engineering
- Multiple embedding methods (TF-IDF, Word2Vec, FastText)
- Topic extraction and analysis
- Configurable output and intermediate data saving
"""

import datetime
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .embeddings import (
    EmbeddingPipeline,
    FastTextEmbedder,
    TfidfEmbedder,
    Word2VecEmbedder,
)
from .feature_engineering import FeatureEngineer
from .loaders import (
    load_entries_from_csv,
    load_entries_from_db,
    load_entries_from_json,
)
from .preprocessing import JournalEntryPreprocessor
from .validation import DataValidator

# Configure logging
# G004: Logging f-strings temporarily allowed for development
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrator for the journal entry data processing pipeline."""

    def __init__(
        self,
        preprocessor: JournalEntryPreprocessor | None = None,
        validator: DataValidator | None = None,
        feature_engineer: FeatureEngineer | None = None,
        embedding_method: str = "tfid",
    ) -> None:
        """Initialize data pipeline.

        Args:
            preprocessor: Journal entry preprocessor
            validator: Data validator
            feature_engineer: Feature engineer
            embedding_method: Method for generating embeddings ('tfid', 'word2vec', or 'fasttext')

        """
        self.preprocessor = preprocessor or JournalEntryPreprocessor()
        self.validator = validator or DataValidator()
        self.feature_engineer = feature_engineer or FeatureEngineer()

        if embedding_method == "tfid":
            embedder = TfidfEmbedder(max_features=1000)
        elif embedding_method == "word2vec":
            embedder = Word2VecEmbedder(vector_size=100)
        elif embedding_method == "fasttext":
            embedder = FastTextEmbedder(vector_size=100)
        else:
            logger.warning(f"Unknown embedding method '{embedding_method}'. Defaulting to TF-IDF.")
            embedder = TfidfEmbedder(max_features=1000)

        self.embedding_pipeline = EmbeddingPipeline(embedder)
        self.embedding_method = embedding_method

    def run(
        self,
        data_source: str | pd.DataFrame,
        source_type: str = "db",
        output_dir: str | None = None,
        user_id: int | None = None,
        limit: int | None = None,
        extract_topics: bool = True,
        save_intermediates: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Run the complete data processing pipeline.

        Args:
            data_source: Source of journal entries (DataFrame or path to file/DB identifier)
            source_type: Type of data source ('db', 'json', 'csv', or 'dataframe')
            output_dir: Directory to save output files
            user_id: Filter entries by user_id
            limit: Maximum number of entries to process
            extract_topics: Whether to extract topic features
            save_intermediates: Whether to save intermediate DataFrames

        Returns:
            Dictionary of DataFrames with raw, processed, featured and embeddings data

        """
        raw_df = self._load_data(data_source, source_type, user_id, limit)

        if raw_df.empty:
            logger.warning("No data loaded. Exiting pipeline.")
            return {"raw": raw_df}

        logger.info(
            "Pipeline processing {len(raw_df)} journal entries",
            extra={"format_args": True},
        )

        validation_passed, validated_df = self.validator.validate_journal_entries(raw_df)

        if not validation_passed:
            logger.warning(
                "Data validation failed. Continuing with validated data, but results may be unreliable."
            )

        processed_df = self.preprocessor.preprocess(validated_df)
        logger.info("Preprocessing completed")

        if extract_topics:
            featured_df, topics_df = self.feature_engineer.extract_all_features(
                processed_df, extract_topics=True
            )
            logger.info("Feature extraction completed (including topics)")
        else:
            featured_df = self.feature_engineer.extract_all_features(
                processed_df, extract_topics=False
            )
            topics_df = None
            logger.info("Feature extraction completed (without topics)")

        embeddings_df = self.embedding_pipeline.generate_embeddings(
            featured_df, text_column="processed_text", id_column="id"
        )
        logger.info("Generated {len(embeddings_df)} embeddings using {self.embedding_method}")

        if output_dir:
            self._save_results(
                output_dir,
                raw_df,
                processed_df,
                featured_df,
                embeddings_df,
                topics_df,
                save_intermediates,
            )

        results = {
            "raw": raw_df,
            "processed": processed_df,
            "featured": featured_df,
            "embeddings": embeddings_df,
        }

        if topics_df is not None:
            results["topics"] = topics_df

        return results

    def _load_data(
        self,
        data_source: str | pd.DataFrame,
        source_type: str,
        user_id: int | None,
        limit: int | None,
    ) -> pd.DataFrame:
        """Load data from specified source.

        Args:
            data_source: Source of journal entries (DataFrame or path to file/DB identifier)
            source_type: Type of data source ('db', 'json', 'csv', or 'dataframe')
            user_id: Filter entries by user_id
            limit: Maximum number of entries to process

        Returns:
            DataFrame containing raw journal entries

        """
        if source_type == "dataframe" and isinstance(data_source, pd.DataFrame):
            logger.info(
                "Using provided DataFrame with {len(data_source)} entries",
                extra={"format_args": True},
            )
            return data_source

        if source_type == "db":
            user_info = " for user {user_id}" if user_id else ""
            limit_info = " (limit: {limit})" if limit else ""
            logger.info("Loading data from database{user_info}{limit_info}")
            return load_entries_from_db(limit=limit, user_id=user_id)

        if source_type == "json" and isinstance(data_source, str):
            logger.info(
                "Loading data from JSON file: {data_source}",
                extra={"format_args": True},
            )
            return load_entries_from_json(data_source)

        if source_type == "csv" and isinstance(data_source, str):
            logger.info("Loading data from CSV file: {data_source}", extra={"format_args": True})
            return load_entries_from_csv(data_source)

        logger.error("Invalid data source type: {source_type}", extra={"format_args": True})
        return pd.DataFrame()

    def _save_results(
        self,
        output_dir: str,
        raw_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        featured_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        topics_df: pd.DataFrame | None = None,
        save_intermediates: bool = False,
    ) -> None:
        """Save pipeline results to output directory.

        Args:
            output_dir: Directory to save output files
            raw_df: DataFrame with raw data
            processed_df: DataFrame with processed data
            featured_df: DataFrame with extracted features
            embeddings_df: DataFrame with embeddings
            topics_df: DataFrame with topic information
            save_intermediates: Whether to save intermediate DataFrames

        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        featured_df.to_csv(
            Path(output_dir, "journal_features_{timestamp}.csv").as_posix(),
            index=False,
        )
        logger.info("Saved featured data to {output_dir}/journal_features_{timestamp}.csv")

        embeddings_path = Path(output_dir, "journal_embeddings_{timestamp}.csv").as_posix()
        self.embedding_pipeline.save_embeddings_to_csv(embeddings_df, embeddings_path)

        if topics_df is not None:
            topics_df.to_csv(
                Path(output_dir, "journal_topics_{timestamp}.csv").as_posix(),
                index=False,
            )
            logger.info("Saved topic data to {output_dir}/journal_topics_{timestamp}.csv")

        if save_intermediates:
            raw_df.to_csv(Path(output_dir, "journal_raw_{timestamp}.csv").as_posix(), index=False)
            logger.info(
                "Saved raw data to {output_dir}/journal_raw_{timestamp}.csv",
                extra={"format_args": True},
            )

            processed_df.to_csv(
                Path(output_dir, "journal_processed_{timestamp}.csv").as_posix(),
                index=False,
            )
            logger.info("Saved processed data to {output_dir}/journal_processed_{timestamp}.csv")

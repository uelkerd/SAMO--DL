import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import os
import logging
from datetime import datetime

from .loaders import (
    load_entries_from_db,
    load_entries_from_json,
    load_entries_from_csv,
    save_entries_to_csv,
    save_entries_to_json
)
from .preprocessing import TextPreprocessor, JournalEntryPreprocessor
from .validation import DataValidator
from .feature_engineering import FeatureEngineer
from .embeddings import TfidfEmbedder, Word2VecEmbedder, FastTextEmbedder, EmbeddingPipeline

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Orchestrator for the journal entry data processing pipeline"""
    
    def __init__(self,
                preprocessor: Optional[JournalEntryPreprocessor] = None,
                validator: Optional[DataValidator] = None,
                feature_engineer: Optional[FeatureEngineer] = None,
                embedding_method: str = 'tfidf'):
        """
        Initialize data pipeline
        
        Args:
            preprocessor: Journal entry preprocessor
            validator: Data validator
            feature_engineer: Feature engineer
            embedding_method: Method for generating embeddings ('tfidf', 'word2vec', or 'fasttext')
        """
        self.preprocessor = preprocessor or JournalEntryPreprocessor()
        self.validator = validator or DataValidator()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # Set up embedding pipeline based on specified method
        if embedding_method == 'tfidf':
            embedder = TfidfEmbedder(max_features=1000)
        elif embedding_method == 'word2vec':
            embedder = Word2VecEmbedder(vector_size=100)
        elif embedding_method == 'fasttext':
            embedder = FastTextEmbedder(vector_size=100)
        else:
            logger.warning(f"Unknown embedding method '{embedding_method}'. Defaulting to TF-IDF.")
            embedder = TfidfEmbedder(max_features=1000)
            
        self.embedding_pipeline = EmbeddingPipeline(embedder)
        self.embedding_method = embedding_method
    
    def run(self,
           data_source: Union[str, pd.DataFrame],
           source_type: str = 'db',
           output_dir: Optional[str] = None,
           user_id: Optional[int] = None,
           limit: Optional[int] = None,
           extract_topics: bool = True,
           save_intermediates: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data processing pipeline
        
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
        # Step 1: Load the data
        raw_df = self._load_data(data_source, source_type, user_id, limit)
        
        if raw_df.empty:
            logger.warning("No data loaded. Exiting pipeline.")
            return {'raw': raw_df}
        
        logger.info(f"Pipeline processing {len(raw_df)} journal entries")
        
        # Step 2: Validate raw data
        validation_passed, validated_df = self.validator.validate_journal_entries(raw_df)
        
        if not validation_passed:
            logger.warning("Data validation failed. Continuing with validated data, but results may be unreliable.")
        
        # Step 3: Preprocess data
        processed_df = self.preprocessor.preprocess(validated_df)
        logger.info("Preprocessing completed")
        
        # Step 4: Feature engineering
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
        
        # Step 5: Generate embeddings
        embeddings_df = self.embedding_pipeline.generate_embeddings(
            featured_df, text_column='processed_text', id_column='id'
        )
        logger.info(f"Generated {len(embeddings_df)} embeddings using {self.embedding_method}")
        
        # Save results if output directory is provided
        if output_dir:
            self._save_results(
                output_dir, raw_df, processed_df, featured_df, 
                embeddings_df, topics_df, save_intermediates
            )
        
        # Return results
        results = {
            'raw': raw_df,
            'processed': processed_df,
            'featured': featured_df,
            'embeddings': embeddings_df
        }
        
        if topics_df is not None:
            results['topics'] = topics_df
            
        return results
    
    def _load_data(self,
                  data_source: Union[str, pd.DataFrame],
                  source_type: str,
                  user_id: Optional[int],
                  limit: Optional[int]) -> pd.DataFrame:
        """
        Load data from specified source
        
        Args:
            data_source: Source of journal entries (DataFrame or path to file/DB identifier)
            source_type: Type of data source ('db', 'json', 'csv', or 'dataframe')
            user_id: Filter entries by user_id
            limit: Maximum number of entries to process
            
        Returns:
            DataFrame containing raw journal entries
        """
        if source_type == 'dataframe' and isinstance(data_source, pd.DataFrame):
            logger.info(f"Using provided DataFrame with {len(data_source)} entries")
            return data_source
        
        elif source_type == 'db':
            logger.info(f"Loading data from database" + 
                       (f" for user {user_id}" if user_id else "") +
                       (f" (limit: {limit})" if limit else ""))
            return load_entries_from_db(limit=limit, user_id=user_id)
        
        elif source_type == 'json' and isinstance(data_source, str):
            logger.info(f"Loading data from JSON file: {data_source}")
            return load_entries_from_json(data_source)
        
        elif source_type == 'csv' and isinstance(data_source, str):
            logger.info(f"Loading data from CSV file: {data_source}")
            return load_entries_from_csv(data_source)
        
        else:
            logger.error(f"Invalid data source type: {source_type}")
            return pd.DataFrame()
    
    def _save_results(self,
                     output_dir: str,
                     raw_df: pd.DataFrame,
                     processed_df: pd.DataFrame,
                     featured_df: pd.DataFrame,
                     embeddings_df: pd.DataFrame,
                     topics_df: Optional[pd.DataFrame] = None,
                     save_intermediates: bool = False) -> None:
        """
        Save pipeline results to output directory
        
        Args:
            output_dir: Directory to save output files
            raw_df: DataFrame with raw data
            processed_df: DataFrame with processed data
            featured_df: DataFrame with extracted features
            embeddings_df: DataFrame with embeddings
            topics_df: DataFrame with topic information
            save_intermediates: Whether to save intermediate DataFrames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save featured data (main output)
        featured_df.to_csv(os.path.join(output_dir, f'journal_features_{timestamp}.csv'), index=False)
        logger.info(f"Saved featured data to {output_dir}/journal_features_{timestamp}.csv")
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, f'journal_embeddings_{timestamp}.csv')
        self.embedding_pipeline.save_embeddings_to_csv(embeddings_df, embeddings_path)
        
        # Save topics if available
        if topics_df is not None:
            topics_df.to_csv(os.path.join(output_dir, f'journal_topics_{timestamp}.csv'), index=False)
            logger.info(f"Saved topic data to {output_dir}/journal_topics_{timestamp}.csv")
        
        # Save intermediate data if requested
        if save_intermediates:
            raw_df.to_csv(os.path.join(output_dir, f'journal_raw_{timestamp}.csv'), index=False)
            logger.info(f"Saved raw data to {output_dir}/journal_raw_{timestamp}.csv")
            
            processed_df.to_csv(os.path.join(output_dir, f'journal_processed_{timestamp}.csv'), index=False)
            logger.info(f"Saved processed data to {output_dir}/journal_processed_{timestamp}.csv") 
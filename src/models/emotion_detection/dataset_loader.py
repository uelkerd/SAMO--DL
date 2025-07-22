"""GoEmotions Dataset Loader for SAMO Emotion Detection.

This module implements comprehensive GoEmotions dataset loading and preprocessing
following the data documentation strategies for BERT fine-tuning.

Key Features:
- Multi-label emotion support (27 categories)
- Class imbalance handling through weighted sampling
- Text preprocessing optimized for emotional understanding
- Domain adaptation preparation for journal entries
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GoEmotions emotion categories (27 emotions + neutral)
GOEMOTIONS_EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Emotion mappings for readability
EMOTION_ID_TO_LABEL = {i: emotion for i, emotion in enumerate(GOEMOTIONS_EMOTIONS)}
EMOTION_LABEL_TO_ID = {emotion: i for i, emotion in enumerate(GOEMOTIONS_EMOTIONS)}


class GoEmotionsPreprocessor:
    """Preprocessing pipeline for GoEmotions dataset following SAMO requirements."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """Initialize preprocessor with BERT tokenizer.
        
        Args:
            model_name: Hugging Face model name for tokenizer
            max_length: Maximum sequence length for BERT processing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(f"Initialized preprocessor with {model_name}, max_length={max_length}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving emotional signals.
        
        Following data documentation strategies for emotional understanding.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text preserving emotional context
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace while preserving structure
        text = ' '.join(text.split())
        
        # Preserve emotional punctuation patterns (!!!, ???)
        # But normalize to consistent representation
        text = text.replace('!!!', ' [EXCITE] ')
        text = text.replace('???', ' [CONFUSE] ')
        
        # Preserve ALL CAPS emotional expressions
        words = text.split()
        processed_words = []
        for word in words:
            if word.isupper() and len(word) > 2:
                processed_words.append(f"[CAPS] {word.lower()}")
            else:
                processed_words.append(word)
        
        text = ' '.join(processed_words)
        
        # Normalize repeated characters (soooo -> so [REPEAT])
        import re
        text = re.sub(r'(.)\1{2,}', r'\1 [REPEAT]', text)
        
        return text.strip()
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts for BERT processing.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with tokenized inputs for BERT
        """
        # Clean texts first
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Tokenize with BERT tokenizer
        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded


class GoEmotionsDataLoader:
    """Main data loader for GoEmotions dataset with SAMO-specific adaptations."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """Initialize GoEmotions data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
            model_name: Model name for tokenizer
            max_length: Maximum sequence length
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        """
        self.cache_dir = cache_dir or "./data/cache"
        self.model_name = model_name
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize preprocessor
        self.preprocessor = GoEmotionsPreprocessor(model_name, max_length)
        
        # Data storage
        self.raw_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        
        logger.info("Initialized GoEmotions data loader")
    
    def download_dataset(self) -> Dataset:
        """Download GoEmotions dataset from Hugging Face.
        
        Returns:
            Raw GoEmotions dataset
        """
        logger.info("Downloading GoEmotions dataset...")
        
        try:
            # Load from Hugging Face datasets
            dataset = load_dataset("go_emotions", cache_dir=self.cache_dir)
            
            # Combine train/validation/test splits for our own splitting
            # GoEmotions comes pre-split, but we want our own validation strategy
            train_data = dataset["train"]
            val_data = dataset["validation"] 
            test_data = dataset["test"]
            
            # Combine all data for custom splitting
            all_texts = []
            all_labels = []
            
            for split_data in [train_data, val_data, test_data]:
                all_texts.extend(split_data["text"])
                all_labels.extend(split_data["labels"])
            
            # Create combined dataset
            combined_data = {
                "text": all_texts,
                "labels": all_labels
            }
            
            self.raw_dataset = Dataset.from_dict(combined_data)
            
            logger.info(f"Downloaded {len(self.raw_dataset)} examples")
            logger.info(f"Example: {self.raw_dataset[0]}")
            
            return self.raw_dataset
            
        except Exception as e:
            logger.error(f"Failed to download GoEmotions dataset: {e}")
            raise
    
    def analyze_dataset_statistics(self) -> Dict[str, any]:
        """Analyze dataset statistics for class imbalance and multi-label patterns.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")
        
        logger.info("Analyzing dataset statistics...")
        
        # Count emotion frequencies
        emotion_counts = np.zeros(len(GOEMOTIONS_EMOTIONS))
        multi_label_count = 0
        total_examples = len(self.raw_dataset)
        
        for example in self.raw_dataset:
            labels = example["labels"]
            if len(labels) > 1:
                multi_label_count += 1
            
            for label_id in labels:
                emotion_counts[label_id] += 1
        
        # Calculate statistics
        emotion_frequencies = emotion_counts / total_examples
        
        stats = {
            "total_examples": total_examples,
            "multi_label_percentage": (multi_label_count / total_examples) * 100,
            "emotion_frequencies": dict(zip(GOEMOTIONS_EMOTIONS, emotion_frequencies)),
            "emotion_counts": dict(zip(GOEMOTIONS_EMOTIONS, emotion_counts.astype(int))),
            "most_frequent_emotions": sorted(
                zip(GOEMOTIONS_EMOTIONS, emotion_frequencies), 
                key=lambda x: x[1], reverse=True
            )[:5],
            "least_frequent_emotions": sorted(
                zip(GOEMOTIONS_EMOTIONS, emotion_frequencies), 
                key=lambda x: x[1]
            )[:5]
        }
        
        # Log key statistics
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Multi-label examples: {multi_label_count} ({stats['multi_label_percentage']:.1f}%)")
        logger.info(f"Most frequent emotions: {stats['most_frequent_emotions']}")
        logger.info(f"Least frequent emotions: {stats['least_frequent_emotions']}")
        
        return stats
    
    def compute_class_weights(self) -> np.ndarray:
        """Compute class weights for handling imbalanced dataset.
        
        Returns:
            Array of class weights for each emotion category
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")
        
        logger.info("Computing class weights for imbalanced dataset...")
        
        # Count positive examples for each emotion
        emotion_counts = np.zeros(len(GOEMOTIONS_EMOTIONS))
        total_examples = len(self.raw_dataset)
        
        for example in self.raw_dataset:
            for label_id in example["labels"]:
                emotion_counts[label_id] += 1
        
        # Compute inverse frequency weights with smoothing
        # Add small constant to prevent division by zero
        epsilon = 1e-7
        emotion_frequencies = emotion_counts / total_examples
        class_weights = 1.0 / (emotion_frequencies + epsilon)
        
        # Normalize weights to prevent extremely large values
        class_weights = class_weights / np.mean(class_weights)
        
        # Cap maximum weight to prevent over-emphasis on very rare emotions
        max_weight = 10.0
        class_weights = np.clip(class_weights, 0.1, max_weight)
        
        self.class_weights = class_weights
        
        logger.info(f"Computed class weights - min: {class_weights.min():.2f}, max: {class_weights.max():.2f}")
        
        return class_weights
    
    def create_train_val_test_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create stratified train/validation/test splits.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")
        
        logger.info("Creating train/validation/test splits...")
        
        # Convert to pandas for easier manipulation
        df = self.raw_dataset.to_pandas()
        
        # For stratification with multi-label data, use the most frequent emotion
        # as the stratification key
        stratify_labels = []
        for labels in df["labels"]:
            if len(labels) > 0:
                # Use first label as stratification key
                stratify_labels.append(labels[0])
            else:
                # Handle edge case of no labels (shouldn't happen in GoEmotions)
                stratify_labels.append(len(GOEMOTIONS_EMOTIONS) - 1)  # neutral
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        # Second split: separate validation from training
        train_stratify = []
        for labels in train_val_df["labels"]:
            train_stratify.append(labels[0] if len(labels) > 0 else len(GOEMOTIONS_EMOTIONS) - 1)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_stratify
        )
        
        # Convert back to datasets
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)
        self.test_dataset = Dataset.from_pandas(test_df)
        
        logger.info(f"Created splits - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def prepare_datasets(self, force_download: bool = False) -> Dict[str, Dataset]:
        """Complete pipeline to prepare GoEmotions datasets.
        
        Args:
            force_download: Whether to force re-download of dataset
            
        Returns:
            Dictionary with train/val/test datasets and metadata
        """
        logger.info("Starting GoEmotions dataset preparation...")
        
        # Download dataset
        if self.raw_dataset is None or force_download:
            self.download_dataset()
        
        # Analyze statistics
        stats = self.analyze_dataset_statistics()
        
        # Compute class weights
        class_weights = self.compute_class_weights()
        
        # Create splits
        train_ds, val_ds, test_ds = self.create_train_val_test_splits()
        
        result = {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
            "statistics": stats,
            "class_weights": class_weights,
            "emotion_mapping": {
                "id_to_label": EMOTION_ID_TO_LABEL,
                "label_to_id": EMOTION_LABEL_TO_ID
            }
        }
        
        logger.info("✅ GoEmotions dataset preparation complete!")
        
        return result


def create_goemotions_loader(
    cache_dir: Optional[str] = None,
    model_name: str = "bert-base-uncased"
) -> GoEmotionsDataLoader:
    """Factory function to create configured GoEmotions data loader.
    
    Args:
        cache_dir: Directory for caching data
        model_name: Model name for tokenizer
        
    Returns:
        Configured GoEmotionsDataLoader instance
    """
    return GoEmotionsDataLoader(
        cache_dir=cache_dir,
        model_name=model_name
    )


if __name__ == "__main__":
    # Test the data loader
    print("Testing GoEmotions Dataset Loader...")
    
    loader = create_goemotions_loader()
    datasets = loader.prepare_datasets()
    
    print("\nDataset preparation complete!")
    print(f"Train examples: {len(datasets['train'])}")
    print(f"Validation examples: {len(datasets['validation'])}")
    print(f"Test examples: {len(datasets['test'])}")
    print(f"Multi-label percentage: {datasets['statistics']['multi_label_percentage']:.1f}%") 
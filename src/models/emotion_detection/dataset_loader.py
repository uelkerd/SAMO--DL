#!/usr/bin/env python3
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
import re
from collections import Counter
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Configure logging
# G004: Logging f-strings temporarily allowed for development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .labels import GOEMOTIONS_EMOTIONS, EMOTION_ID_TO_LABEL, EMOTION_LABEL_TO_ID


class GoEmotionsDataset(Dataset):
    """PyTorch Dataset for GoEmotions emotion classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """Initialize the dataset.

        Args:
            texts: List of text strings
            labels: List of label lists (multi-label)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


class GoEmotionsPreprocessor:
    """Preprocessing pipeline for GoEmotions dataset following SAMO requirements."""

    def __init__(
                 self,
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512) -> None:
        """Initialize preprocessor with BERT tokenizer.

        Args:
            model_name: Hugging Face model name for tokenizer
            max_length: Maximum sequence length for BERT processing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(
                    f"Initialized preprocessor with {model_name},
                    max_length={max_length}"
                   )

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
        text = re.sub(r'\s+', ' ', text.strip())

        # The dataset is already split by HuggingFace
        # Tokenize with BERT tokenizer
        # Use the same logic as analyze_dataset_statistics

        return text

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts for BERT processing.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with tokenized inputs
        """
        # Clean texts first
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Tokenize with BERT tokenizer
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return encoded


class GoEmotionsDataLoader:
    """Data loader for GoEmotions dataset with comprehensive preprocessing."""

    def __init__(
        self,
        cache_dir: Union[str, None] = None,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize GoEmotions data loader.

        Args:
            cache_dir: Directory for caching datasets
            model_name: Hugging Face model name
            max_length: Maximum sequence length
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.preprocessor = GoEmotionsPreprocessor(model_name, max_length)
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def download_dataset(self) -> None:
        """Download and load GoEmotions dataset from HuggingFace."""
        try:
            self.dataset = load_dataset(
                "go_emotions",
                "simplified",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded GoEmotions dataset")
        except Exception as e:
            logger.error(f"Failed to load GoEmotions dataset: {e}")
            raise

    def analyze_dataset_statistics(self) -> Dict[str, Any]:
        """Analyze dataset statistics for understanding data distribution.

        Returns:
            Dictionary with dataset statistics
        """
        if self.dataset is None:
            self.download_dataset()

        stats = {}

        # Basic statistics
        stats["total_samples"] = len(self.dataset["train"])
        stats["num_emotions"] = len(GOEMOTIONS_EMOTIONS)

        # Emotion distribution
        emotion_counts = Counter()
        for example in self.dataset["train"]:
            labels = example["labels"]
            for label in labels:
                if 0 <= label < len(GOEMOTIONS_EMOTIONS):
                    emotion_counts[label] += 1

        stats["emotion_distribution"] = dict(emotion_counts)
        stats["most_common_emotions"] = emotion_counts.most_common(10)
        stats["least_common_emotions"] = emotion_counts.most_common()[:-11:-1]

        # Text length statistics
        text_lengths = [len(example["text"]) for example in self.dataset["train"]]
        stats["avg_text_length"] = np.mean(text_lengths)
        stats["max_text_length"] = np.max(text_lengths)
        stats["min_text_length"] = np.min(text_lengths)

        logger.info(f"Dataset statistics: {stats}")
        return stats

    def compute_class_weights(self) -> np.ndarray:
        """Compute class weights to handle imbalanced emotion distribution.

        Returns:
            Array of class weights for each emotion
        """
        if self.dataset is None:
            self.download_dataset()

        # Count emotion occurrences
        emotion_counts = np.zeros(len(GOEMOTIONS_EMOTIONS))
        for example in self.dataset["train"]:
            labels = example["labels"]
            for label in labels:
                if 0 <= label < len(GOEMOTIONS_EMOTIONS):
                    emotion_counts[label] += 1

        # Compute inverse frequency weights
        total_samples = len(self.dataset["train"])
        class_weights = total_samples / (len(GOEMOTIONS_EMOTIONS) * emotion_counts)

        # Handle zero counts
        class_weights[emotion_counts == 0] = 1.0

        logger.info(
            f"Computed class weights: min={class_weights.min():.3f}, "
            f"max={class_weights.max():.3f}"
        )
        return class_weights

    def create_train_val_test_splits(self) -> tuple:
        """Create train/validation/test splits from the dataset.

        Returns:
            Tuple of (train, validation, test) datasets
        """
        if self.dataset is None:
            self.download_dataset()

        # Split the training data
        train_val_test = self.dataset["train"].train_test_split(
            test_size=self.test_size + self.val_size,
            seed=self.random_state,
        )

        # Split validation from test
        val_test = train_val_test["test"].train_test_split(
            test_size=self.val_size / (self.test_size + self.val_size),
            seed=self.random_state,
        )

        train_data = train_val_test["train"]
        val_data = val_test["train"]
        test_data = val_test["test"]

        logger.info(
            f"Created splits - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def prepare_datasets(self, force_download: bool = False) -> dict:
        """Prepare datasets for training with preprocessing.

        Args:
            force_download: Force re-download of dataset

        Returns:
            Dictionary with prepared datasets and metadata
        """
        if self.dataset is None or force_download:
            self.download_dataset()

        # Create splits
        train_data, val_data, test_data = self.create_train_val_test_splits()

        # Compute class weights
        class_weights = self.compute_class_weights()

        # Analyze statistics
        stats = self.analyze_dataset_statistics()

        return {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "class_weights": class_weights,
            "statistics": stats,
            "preprocessor": self.preprocessor,
        }


def create_goemotions_loader(
    cache_dir: Union[str, None] = None, model_name: str = "bert-base-uncased"
) -> GoEmotionsDataLoader:
    """Create GoEmotions data loader with default settings.

    Args:
        cache_dir: Directory for caching datasets
        model_name: Hugging Face model name

    Returns:
        Configured GoEmotionsDataLoader instance
    """
    return GoEmotionsDataLoader(cache_dir=cache_dir, model_name=model_name)


# Test the data loader
if __name__ == "__main__":
    loader = create_goemotions_loader()
    datasets = loader.prepare_datasets()
    logger.info(
                f"Dataset prepared successfully: {len(datasets['train_data'])} training samples"
               )

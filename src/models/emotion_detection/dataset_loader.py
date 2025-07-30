# G004: Logging f-strings temporarily allowed for development
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

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from typing import Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GoEmotions emotion categories (27 emotions + neutral)
GOEMOTIONS_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grie",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relie",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Emotion mappings for readability
EMOTION_ID_TO_LABEL = dict(enumerate(GOEMOTIONS_EMOTIONS))
EMOTION_LABEL_TO_ID = {emotion: i for i, emotion in enumerate(GOEMOTIONS_EMOTIONS)}


class GoEmotionsPreprocessor:
    """Preprocessing pipeline for GoEmotions dataset following SAMO requirements."""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512) -> None:
        """Initialize preprocessor with BERT tokenizer.

        Args:
            model_name: Hugging Face model name for tokenizer
            max_length: Maximum sequence length for BERT processing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info("Initialized preprocessor with {model_name}, max_length={max_length}")

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
        text = " ".join(text.split())

        # Preserve emotional punctuation patterns (!!!, ???)
        # But normalize to consistent representation
        text = text.replace("!!!", " [EXCITE] ")
        text = text.replace("???", " [CONFUSE] ")

        # Preserve ALL CAPS emotional expressions
        words = text.split()
        processed_words = []
        for word in words:
            if word.isupper() and len(word) > 2:
                processed_words.append("[CAPS] {word.lower()}")
            else:
                processed_words.append(word)

        text = " ".join(processed_words)

        # Normalize repeated characters (soooo -> so [REPEAT])
        import re

        text = re.sub(r"(.)\1{2,}", r"\1 [REPEAT]", text)

        return text.strip()

    def tokenize_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
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
            return_tensors="pt",
        )

        return encoded


class GoEmotionsDataLoader:
    """Main data loader for GoEmotions dataset with SAMO-specific adaptations."""

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

    def download_dataset(self) -> None:
        """Download GoEmotions dataset from Hugging Face Hub."""
        try:
            logger.info("Downloading GoEmotions dataset...")
            # Explicitly disable token usage for public dataset
            dataset = load_dataset("go_emotions", cache_dir=self.cache_dir, token=False)
            self.dataset = dataset
            logger.info("✅ GoEmotions dataset downloaded successfully.")
        except Exception as e:
            logger.error("Failed to download GoEmotions dataset: {e}")
            raise

    def analyze_dataset_statistics(self) -> dict[str, Any]:
        """Analyze and log statistics about the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")

        # Combine labels from all splits to get overall statistics
        all_labels = []
        for split in self.dataset:
            all_labels.extend(
                [label for labels in self.dataset[split]["labels"] for label in labels]
            )

        # Calculate label frequencies
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)

        # Calculate statistics
        emotion_frequencies = {
            emotion: count / total_labels for emotion, count in label_counts.items()
        }

        stats = {
            "total_examples": total_labels,
            "multi_label_percentage": (
                total_labels - sum(1 for count in label_counts.values() if count == 1)
            )
            / total_labels
            * 100,
            "emotion_frequencies": emotion_frequencies,
            "emotion_counts": label_counts,
            "most_frequent_emotions": sorted(
                emotion_frequencies.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "least_frequent_emotions": sorted(
                emotion_frequencies.items(),
                key=lambda x: x[1],
            )[:5],
        }

        # Log key statistics
        logger.info("Total examples: {total_labels}")
        logger.info(
            "Multi-label examples: {total_labels - sum(1 for count in label_counts.values() if count == 1)} ({stats['multi_label_percentage']:.1f}%)"
        )
        logger.info("Most frequent emotions: {stats['most_frequent_emotions']}")
        logger.info("Least frequent emotions: {stats['least_frequent_emotions']}")

        return stats

    def compute_class_weights(self) -> np.ndarray:
        """Compute class weights for handling data imbalance."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")

        # Use the same logic as analyze_dataset_statistics
        all_labels = []
        for split in self.dataset:
            all_labels.extend(
                [label for labels in self.dataset[split]["labels"] for label in labels]
            )

        label_counts = Counter(all_labels)
        total_samples = sum(label_counts.values())

        # Calculate weights: inverse frequency
        class_weights = np.array(
            [
                total_samples / label_counts.get(i, 1)  # Use .get for safety
                for i in range(len(GOEMOTIONS_EMOTIONS))
            ]
        )

        # Normalize weights
        class_weights = class_weights / np.sum(class_weights)

        logger.info(
            "Computed class weights. Min: {class_weights.min():.4f}, Max: {class_weights.max():.4f}"
        )
        return class_weights

    def create_train_val_test_splits(self) -> tuple:
        """Create train/val/test splits from the dataset.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download_dataset() first.")

        logger.info("Creating train/val/test splits...")

        # The dataset is already split by HuggingFace
        train_ds = self.dataset["train"]
        val_ds = self.dataset["validation"]
        test_ds = self.dataset["test"]

        # Log split sizes
        logger.info("Train set: {len(train_ds)} examples")
        logger.info("Validation set: {len(val_ds)} examples")
        logger.info("Test set: {len(test_ds)} examples")

        return train_ds, val_ds, test_ds

    def prepare_datasets(self, force_download: bool = False) -> dict:
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
                "label_to_id": EMOTION_LABEL_TO_ID,
            },
        }

        logger.info("✅ GoEmotions dataset preparation complete!")

        return result


def create_goemotions_loader(
    cache_dir: Union[str, None] = None, model_name: str = "bert-base-uncased"
) -> GoEmotionsDataLoader:
    """Factory function to create configured GoEmotions data loader.

    Args:
        cache_dir: Directory for caching data
        model_name: Model name for tokenizer

    Returns:
        Configured GoEmotionsDataLoader instance
    """
    return GoEmotionsDataLoader(cache_dir=cache_dir, model_name=model_name)


if __name__ == "__main__":
    # Test the data loader

    loader = create_goemotions_loader()
    datasets = loader.prepare_datasets()

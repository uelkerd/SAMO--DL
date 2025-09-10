#!/usr/bin/env python3
"""
SAMO-Enhanced BERT Emotion Classifier

This module provides an enhanced BERT-based emotion classification model
optimized for journal entries and emotional text processing in the SAMO-DL system.

Key Features:
- BERT-base-uncased backbone for robust text understanding
- Multi-label emotion classification (27 emotions + neutral)
- Temperature scaling for calibrated predictions
- Dropout regularization to prevent overfitting
- Comprehensive error handling and logging
- SAMO-specific optimizations for journal entries
"""

import logging
import warnings
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class SAMOBERTEmotionClassifier(nn.Module):
    """
    SAMO-enhanced BERT emotion classifier for multi-label emotion detection.

    Architecture:
    - BERT-base-uncased backbone
    - Two-layer classification head for non-linear feature combination
    - Sigmoid activation for independent emotion predictions
    - Temperature scaling for calibrated predictions
    - Dropout regularization to prevent overfitting
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 28,  # 27 emotions + neutral
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize SAMO BERT emotion classifier.

        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion categories (27 + neutral)
            config: Optional configuration dictionary
        """
        super().__init__()

        # Set default config
        default_config = {
            "hidden_dropout_prob": 0.3,
            "classifier_dropout_prob": 0.5,
            "freeze_bert_layers": 6,
            "temperature": 1.0,
        }

        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.classifier_dropout_prob = config["classifier_dropout_prob"]
        self.freeze_bert_layers = config["freeze_bert_layers"]
        self.temperature = nn.Parameter(torch.ones(1) * config["temperature"])
        self.class_weights = None
        self.prediction_threshold = 0.6  # Updated from 0.5 to 0.6 based on calibration

        # Load BERT model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = self.hidden_dropout_prob
        self.config.attention_probs_dropout_prob = self.hidden_dropout_prob

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.bert_hidden_size = self.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, num_emotions),
        )

        # Initialize classification layers
        self._init_classification_layers()

        # Freeze BERT layers if specified
        if self.freeze_bert_layers > 0:
            self._freeze_bert_layers(self.freeze_bert_layers)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        logger.info(f"✅ SAMO BERT Emotion Classifier initialized on {self.device}")

    def _init_classification_layers(self) -> None:
        """Initialize classification layers with proper weight initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _set_bert_layers_grad(self, num_layers: int, requires_grad: bool) -> None:
        """Set gradient requirements for BERT layers."""
        if num_layers <= 0:
            return

        # Set embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = requires_grad

        # Set encoder layers
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = requires_grad

        action = "Unfrozen" if requires_grad else "Frozen"
        logger.info(f"{action} {num_layers} BERT layers")

    def _freeze_bert_layers(self, num_layers: int) -> None:
        """Freeze the first num_layers of BERT."""
        self._set_bert_layers_grad(num_layers, False)

    def unfreeze_bert_layers(self, num_layers: int) -> None:
        """Unfreeze the first num_layers of BERT."""
        self._set_bert_layers_grad(num_layers, True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the BERT emotion classifier.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs (optional)

        Returns:
            Logits for emotion classification
        """
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token representation for classification
        # Check if pooler_output exists, fallback to first token if not
        if bert_outputs.pooler_output is not None:
            pooled_output = bert_outputs.pooler_output
        else:
            # Fallback to first token ([CLS]) hidden state
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]

        # Pass through classification head
        logits = self.classifier(pooled_output)

        # Apply temperature scaling
        logits = logits / self.temperature

        return logits

    def predict_emotions(
        self,
        texts: Union[str, List[str]],
        threshold: float = None,
        top_k: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Union[List[List[str]], List[List[float]], List[List[float]]]]:
        """
        Predict emotions for given texts.

        Args:
            texts: Single text or list of texts
            threshold: Prediction threshold (uses default if None)
            top_k: Return top-k emotions per text
            batch_size: Batch size for processing

        Returns:
            Dictionary with emotions, probabilities, and predictions
        """
        if threshold is None:
            threshold = self.prediction_threshold

        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        all_emotions = []
        all_probabilities = []
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                token_type_ids = encoded.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                # Get predictions
                logits = self.forward(input_ids, attention_mask, token_type_ids)
                probabilities = torch.sigmoid(logits)

                # Apply threshold
                predictions = (probabilities > threshold).float()

                # Get top-k if specified
                if top_k is not None:
                    _, top_k_indices = torch.topk(probabilities, top_k, dim=1)
                    predictions = torch.zeros_like(probabilities)
                    predictions.scatter_(1, top_k_indices, 1.0)

                # Convert to lists
                batch_predictions = predictions.cpu().numpy()
                batch_probabilities = probabilities.cpu().numpy()

                # Get emotion names for predictions
                from .emotion_labels import GOEMOTIONS_EMOTIONS
                for pred in batch_predictions:
                    emotions = [
                        GOEMOTIONS_EMOTIONS[i] for i, p in enumerate(pred) if p > 0
                    ]
                    all_emotions.append(emotions)

                all_probabilities.extend(batch_probabilities.tolist())
                all_predictions.extend(batch_predictions.tolist())

        return {
            "emotions": all_emotions,
            "probabilities": all_probabilities,
            "predictions": all_predictions,
        }

    def set_temperature(self, temperature: float) -> None:
        """Set temperature scaling parameter."""
        self.temperature.data.fill_(temperature)
        logger.info(f"Set temperature to {temperature}")

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_frozen_parameters(self) -> int:
        """Count number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for multi-label emotion classification."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize weighted BCE loss.

        Args:
            class_weights: Class weights for balancing loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Model predictions
            targets: Ground truth labels

        Returns:
            Weighted BCE loss
        """
        # Compute BCE loss with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights.unsqueeze(0)

        # Apply reduction
        if self.reduction == "mean":
            return bce_loss.mean()
        if self.reduction == "sum":
            return bce_loss.sum()
        return bce_loss


class EmotionDataset(Dataset):
    """Dataset for emotion classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        """
        Initialize emotion dataset.

        Args:
            texts: List of text samples
            labels: List of label lists (multi-label)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index."""
        text = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert labels to tensor
        label_tensor = torch.tensor(labels, dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": label_tensor,
        }


def create_samo_bert_emotion_classifier(
    model_name: str = "bert-base-uncased",
    num_emotions: int = 28,
    class_weights: Optional[np.ndarray] = None,
    freeze_bert_layers: int = 6,
) -> Tuple[SAMOBERTEmotionClassifier, WeightedBCELoss]:
    """
    Create SAMO BERT emotion classifier with loss function.

    Args:
        model_name: Hugging Face model name
        num_emotions: Number of emotion categories
        class_weights: Optional class weights for imbalanced data
        freeze_bert_layers: Number of BERT layers to freeze

    Returns:
        Tuple of (model, loss_function)
    """
    # Convert class weights to tensor if provided
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Create model with only the parameters that override defaults
    config = {
        "freeze_bert_layers": freeze_bert_layers,
    }

    model = SAMOBERTEmotionClassifier(
        model_name=model_name,
        num_emotions=num_emotions,
        config=config,
    )

    # Create loss function
    loss_function = WeightedBCELoss(class_weights=class_weights_tensor)

    return model, loss_function


def evaluate_emotion_classifier(
    model: SAMOBERTEmotionClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate emotion classifier performance.

    Args:
        model: Trained emotion classifier
        dataloader: Data loader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    from .config import EMOTION_CLASSIFICATION_THRESHOLD
    threshold = EMOTION_CLASSIFICATION_THRESHOLD
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            targets = batch["labels"].to(device)

            # Get predictions
            logits = model(input_ids, attention_mask, token_type_ids)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average="micro", zero_division=0
    )

    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_micro": f1,
        "f1_macro": macro_f1,
    }


if __name__ == "__main__":
    # Test the emotion classifier
    print("🧪 Testing SAMO BERT Emotion Classifier")
    print("=" * 50)

    try:
        # Create model
        print("1. Creating SAMO BERT Emotion Classifier...")
        model, loss_fn = create_samo_bert_emotion_classifier()
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        print(f"   Frozen parameters: {model.count_frozen_parameters():,}")

        # Test prediction
        print("\n2. Testing emotion prediction...")
        test_texts = [
            "I am so happy today! This is amazing!",
            "I feel really sad and disappointed about this situation.",
            "I'm feeling anxious and worried about the future.",
        ]

        results = model.predict_emotions(test_texts, threshold=0.3)

        for i, text in enumerate(test_texts):
            print(f"\nText: {text}")
            print(f"Emotions: {results['emotions'][i]}")
            print(f"Top probabilities: {[f'{p:.3f}' for p in results['probabilities'][i][:5]]}")

        print("\n✅ SAMO BERT Emotion Classifier test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing emotion classifier: {e}")
        raise

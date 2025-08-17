#!/usr/bin/env python3
"""
BERT-based Emotion Classifier for SAMO Deep Learning.

This module provides a BERT-based multi-label emotion classification model
trained on the GoEmotions dataset for journal entry analysis.
"""

import logging
import warnings
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .labels import GOEMOTIONS_EMOTIONS

# Configure logging
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__

# Suppress warnings for cleaner output
warnings.filterwarnings"ignore", category=UserWarning


class BERTEmotionClassifiernn.Module:
    """BERT-based emotion classifier for multi-label emotion detection.

    Architecture:
    - BERT-base-uncased backbone
    - Two-layer classification head for non-linear feature combination
    - Sigmoid activation for independent emotion predictions
    - Dropout regularization to prevent overfitting
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 28,  # 27 emotions + neutral
        hidden_dropout_prob: float = 0.3,
        classifier_dropout_prob: float = 0.5,
        freeze_bert_layers: int = 0,
        temperature: float = 1.0,  # Temperature scaling for calibration
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize BERT emotion classifier.

        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion categories 27 + neutral
            hidden_dropout_prob: Dropout rate for BERT hidden layers
            classifier_dropout_prob: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze initially
            temperature: Temperature scaling parameter for probability calibration
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.freeze_bert_layers = freeze_bert_layers
        self.temperature = temperature
        self.prediction_threshold = 0.6  # Updated from 0.5 to 0.6 based on calibration
        self.class_weights = class_weights
        self.emotion_labels = GOEMOTIONS_EMOTIONS[:num_emotions]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrainedmodel_name
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = hidden_dropout_prob

        self.bert = AutoModel.from_pretrainedmodel_name, config=config

        self.bert_hidden_size = config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropoutclassifier_dropout_prob,
            nn.Linearself.bert_hidden_size, self.bert_hidden_size,
            nn.ReLU(),
            nn.Dropoutclassifier_dropout_prob,
            nn.Linearself.bert_hidden_size, self.num_emotions,
        )

        self.temperature = nn.Parameter(torch.ones1)

        # Initialize classification layers
        self._init_classification_layers()

        # Freeze BERT layers if specified
        if freeze_bert_layers > 0:
            self._freeze_bert_layersfreeze_bert_layers

    def _init_classification_layersself -> None:
        """Initialize classification layers with proper weight initialization."""
        for module in self.classifier:
            if isinstancemodule, nn.Linear:
                nn.init.xavier_uniform_module.weight
                nn.init.zeros_module.bias

    def _freeze_bert_layersself, num_layers: int -> None:
        """Freeze the first num_layers of BERT.

        Args:
            num_layers: Number of BERT layers to freeze
        """
        if num_layers <= 0:
            return

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        for i in range(min(num_layers, lenself.bert.encoder.layer)):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        logger.infof"Froze {num_layers} BERT layers"

    def unfreeze_bert_layersself, num_layers: int -> None:
        """Unfreeze the first num_layers of BERT.

        Args:
            num_layers: Number of BERT layers to unfreeze
        """
        if num_layers <= 0:
            return

        # Unfreeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True

        # Unfreeze encoder layers
        for i in range(min(num_layers, lenself.bert.encoder.layer)):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        logger.infof"Unfroze {num_layers} BERT layers"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the BERT emotion classifier.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs optional

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
        pooled_output = bert_outputs.pooler_output

        # Pass through classification head
        logits = self.classifierpooled_output

        # Apply temperature scaling
        logits = logits / self.temperature

        return logits

    def set_temperatureself, temperature: float -> None:
        """Set temperature scaling parameter.

        Args:
            temperature: Temperature value for scaling
        """
        self.temperature.data.fill_temperature
        logger.infof"Set temperature to {temperature}"

    def predict_emotions(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, Union[List[str], torch.Tensor, List[float]]], List[List[int]]]:
        """Predict emotions for given texts.

        Args:
            texts: Single text or list of texts ignored if input_ids provided
            threshold: Prediction threshold for binary classification
            top_k: Return top-k emotions per text
            input_ids: Pre-tokenized input IDs for testing
            attention_mask: Pre-tokenized attention mask for testing

        Returns:
            Dictionary with predictions, probabilities, and emotion names, or list of predictions for testing
        """
        # Handle direct input_ids/attention_mask for testing
        if input_ids is not None and attention_mask is not None:
            self.eval()
            with torch.no_grad():
                logits = self.forwardinput_ids, attention_mask
                probabilities = torch.sigmoidlogits
                predictions = probabilities > threshold.float()
                return predictions.cpu().numpy().tolist()

        # Original implementation for text input
        if isinstancetexts, str:
            texts = [texts]

        # Tokenize texts
        tokenizer = AutoTokenizer.from_pretrainedself.model_name
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoded["input_ids"].toself.device
        attention_mask = encoded["attention_mask"].toself.device

        # Set model to evaluation mode
        self.eval()

        with torch.no_grad():
            # Get logits
            logits = self.forwardinput_ids, attention_mask
            probabilities = torch.sigmoidlogits

            # Apply threshold
            predictions = probabilities > threshold.float()

            # Get top-k emotions if specified
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topkprobabilities, top_k, dim=1
                predictions = torch.zeros_likeprobabilities
                predictions.scatter_1, top_k_indices, 1.0

        # Convert to lists
        predictions_list = predictions.cpu().numpy().tolist()
        probabilities_list = probabilities.cpu().numpy().tolist()

        # Get emotion names
        emotion_names = []
        for pred in predictions_list:
            emotions = [GOEMOTIONS_EMOTIONS[i] for i, p in enumeratepred if p > 0]
            emotion_names.appendemotions

        return {
            "emotions": emotion_names,
            "probabilities": probabilities_list,
            "predictions": predictions_list,
        }

    def count_parametersself -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_frozen_parametersself -> int:
        """Count number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class WeightedBCELossnn.Module:
    """Weighted Binary Cross Entropy Loss for multi-label emotion classification."""

    def __init__(
        self, class_weights: Optional[torch.Tensor] = None, reduction: str = "mean"
    ) -> None:
        """Initialize weighted BCE loss.

        Args:
            class_weights: Class weights for balancing loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forwardself, logits: torch.Tensor, targets: torch.Tensor -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Model predictions
            targets: Ground truth labels

        Returns:
            Weighted BCE loss
        """
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoidlogits

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy(
            probabilities, targets.float(), reduction="none"
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights.unsqueeze0

        # Apply reduction
        if self.reduction == "mean":
            return bce_loss


class EmotionDatasetDataset:
    """Dataset for emotion classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        """Initialize emotion dataset.

        Args:
            texts: List of text samples
            labels: List of label lists multi-label
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__self -> int:
        """Return dataset length."""
        return lenself.texts

    def __getitem__self, idx: int -> Dict[str, torch.Tensor]:
        """Get item at index.

        Args:
            idx: Index of item

        Returns:
            Dictionary with tokenized inputs and labels
        """
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
        label_tensor = torch.tensorlabels, dtype=torch.float

        return {
            "input_ids": encoding["input_ids"].squeeze0,
            "attention_mask": encoding["attention_mask"].squeeze0,
            "labels": label_tensor,
        }


def create_bert_emotion_classifier(
    model_name: str = "bert-base-uncased",
    class_weights: Optional[np.ndarray] = None,
    freeze_bert_layers: int = 6,
) -> Tuple[BERTEmotionClassifier, WeightedBCELoss]:
    """Create BERT emotion classifier with loss function.

    Args:
        model_name: Hugging Face model name
        class_weights: Class weights for loss function
        freeze_bert_layers: Number of BERT layers to freeze

    Returns:
        Tuple of model, loss_function
    """
    model = BERTEmotionClassifier(
        model_name=model_name,
        freeze_bert_layers=freeze_bert_layers,
    )

    if class_weights is not None:
        class_weights_tensor = torch.tensorclass_weights, dtype=torch.float
        loss_function = WeightedBCELossclass_weights=class_weights_tensor
    else:
        loss_function = WeightedBCELoss()

    return model, loss_function


def evaluate_emotion_classifier(
    model: BERTEmotionClassifier,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.2,  # Lowered from 0.5 to capture more predictions
) -> Dict[str, float]:
    """Evaluate emotion classifier performance.

    Args:
        model: BERT emotion classifier
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Prediction threshold

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].todevice
            attention_mask = batch["attention_mask"].todevice
            targets = batch["labels"].todevice

            logits = modelinput_ids, attention_mask
            probabilities = torch.sigmoidlogits
            predictions = probabilities > threshold.float()

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_predictions = torch.catall_predictions, dim=0
    all_targets = torch.catall_targets, dim=0

    # Convert to numpy for sklearn metrics
    predictions_np = all_predictions.numpy()
    targets_np = all_targets.numpy()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions_np, average="micro", zero_division=0
    )

    # Calculate macro F1 for better class balance assessment
    macro_f1 = f1_scoretargets_np, predictions_np, average="macro", zero_division=0

    return {
        "precision": precision,
        "recall": recall,
        "f1_micro": f1,
        "f1_macro": macro_f1,
    }

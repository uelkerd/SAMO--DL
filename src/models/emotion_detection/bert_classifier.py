# G004: Logging f-strings temporarily allowed for development
"""BERT Emotion Classifier for SAMO Deep Learning.

This module implements the BERT-based emotion detection model following the
training strategies from the model training playbook for 27-category emotion
classification with multi-label support.

Key Features:
- BERT-base-uncased foundation with emotional fine-tuning
- Multi-label classification with sigmoid activation
- Progressive unfreezing strategy for transfer learning
- Class-weighted loss for imbalanced data handling
- Temperature scaling for confidence calibration
"""

import logging
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from .dataset_loader import GOEMOTIONS_EMOTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class BERTEmotionClassifier(nn.Module):
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
    ) -> None:
        """Initialize BERT emotion classifier.

        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion categories (27 + neutral)
            hidden_dropout_prob: Dropout rate for BERT hidden layers
            classifier_dropout_prob: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze initially
        """
        super().__init__()

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.freeze_bert_layers = freeze_bert_layers

        # Load BERT configuration and modify for our task
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = hidden_dropout_prob

        # Initialize BERT backbone
        self.bert = AutoModel.from_pretrained(model_name, config=config)

        # Get BERT hidden size (768 for bert-base)
        self.bert_hidden_size = config.hidden_size

        # Two-layer classification head for non-linear feature combination
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, self.num_emotions),
        )

        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize classification layers with Xavier initialization
        self._init_classification_layers()

        # Apply initial layer freezing if specified
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)

        logger.info(
            f"Initialized BERT emotion classifier with {self.count_parameters():,} parameters"
        )

    def _init_classification_layers(self) -> None:
        """Initialize classification layers with Xavier initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _freeze_bert_layers(self, num_layers: int) -> None:
        """Freeze specified number of BERT layers for progressive unfreezing.

        Args:
            num_layers: Number of layers to freeze (0 = none, 12 = all)
        """
        # Freeze embedding layer
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        logger.info("Frozen {num_layers} BERT layers for progressive training", extra={"format_args": True})

    def unfreeze_bert_layers(self, num_layers: int) -> None:
        """Unfreeze BERT layers for progressive unfreezing strategy.

        Args:
            num_layers: Number of additional layers to unfreeze
        """
        # Unfreeze embedding layer if unfreezing any layers
        if num_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = True

        # Calculate which layers to unfreeze
        total_layers = len(self.bert.encoder.layer)
        currently_frozen = sum(
            1
            for layer in self.bert.encoder.layer
            if not next(layer.parameters()).requires_grad
        )

        layers_to_unfreeze = min(num_layers, currently_frozen)
        start_layer = total_layers - currently_frozen

        # Unfreeze layers from the top
        for i in range(start_layer, start_layer + layers_to_unfreeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        logger.info("Unfroze {layers_to_unfreeze} additional BERT layers", extra={"format_args": True})

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through BERT emotion classifier.

        Args:
            input_ids: Token IDs from BERT tokenizer
            attention_mask: Attention mask for padding tokens
            token_type_ids: Token type IDs (optional)
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary with logits, probabilities, and optional attention weights
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention_weights,
        )

        # Use [CLS] token representation for classification
        pooled_output = bert_outputs.pooler_output

        # Classification head
        logits = self.classifier(pooled_output)

        # Apply temperature scaling for calibrated probabilities
        calibrated_logits = logits / self.temperature
        probabilities = torch.sigmoid(calibrated_logits)

        outputs = {
            "logits": logits,
            "probabilities": probabilities,
            "calibrated_logits": calibrated_logits,
        }

        if return_attention_weights:
            outputs["attention_weights"] = bert_outputs.attentions

        return outputs

    def predict_emotions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        threshold: float = 0.5,
        top_k: int | None = None,
    ) -> dict[str, list[str] | torch.Tensor | list[float]]:
        """Predict emotions for input text with confidence scores.

        Args:
            input_ids: Token IDs from BERT tokenizer
            attention_mask: Attention mask for padding tokens
            token_type_ids: Token type IDs (optional)
            threshold: Probability threshold for emotion prediction
            top_k: Return top K emotions regardless of threshold

        Returns:
            Dictionary with predicted emotions, probabilities, and confidence info
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            probabilities = outputs["probabilities"].cpu().numpy()

        # Handle batch dimension
        if probabilities.ndim == 2:
            probabilities = probabilities[0]  # Take first example if batch

        # Get emotion predictions
        if top_k is not None:
            # Return top K emotions
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            predicted_emotions = [GOEMOTIONS_EMOTIONS[i] for i in top_indices]
            emotion_scores = probabilities[top_indices].tolist()
        else:
            # Use threshold-based prediction
            predicted_indices = np.where(probabilities >= threshold)[0]
            predicted_emotions = [GOEMOTIONS_EMOTIONS[i] for i in predicted_indices]
            emotion_scores = probabilities[predicted_indices].tolist()

        # Get primary emotion (highest probability)
        primary_emotion_idx = np.argmax(probabilities)
        primary_emotion = GOEMOTIONS_EMOTIONS[primary_emotion_idx]
        primary_confidence = probabilities[primary_emotion_idx]

        return {
            "predicted_emotions": predicted_emotions,
            "emotion_scores": emotion_scores,
            "primary_emotion": primary_emotion,
            "primary_confidence": float(primary_confidence),
            "all_probabilities": probabilities.tolist(),
            "emotion_mapping": dict(
                zip(GOEMOTIONS_EMOTIONS, probabilities.tolist(), strict=False)
            ),
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for imbalanced multi-label classification.

    Implements class weighting to handle emotion frequency imbalance in GoEmotions.
    """

    def __init__(
        self, class_weights: torch.Tensor | None = None, reduction: str = "mean"
    ) -> None:
        """Initialize weighted BCE loss.

        Args:
            class_weights: Tensor of shape [num_classes] with class weights
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

        if class_weights is not None:
            logger.info(
                f"Initialized WeightedBCELoss with class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}"
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted binary cross entropy loss.

        Args:
            logits: Model predictions of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size, num_classes]

        Returns:
            Computed loss tensor
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            # Ensure class weights are on the same device
            weights = self.class_weights.to(logits.device)
            # Apply weights: shape [batch_size, num_classes]
            weighted_loss = bce_loss * weights.unsqueeze(0)
        else:
            weighted_loss = bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        if self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification with tokenization."""

    def __init__(
        self,
        texts: list[str],
        labels: list[list[int]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        """Initialize emotion dataset.

        Args:
            texts: List of text strings
            labels: List of emotion label lists (multi-label)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_emotions = len(GOEMOTIONS_EMOTIONS)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get dataset item with tokenization."""
        text = str(self.texts[idx])
        label_ids = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create multi-label target vector
        target = torch.zeros(self.num_emotions, dtype=torch.float32)
        for label_id in label_ids:
            if 0 <= label_id < self.num_emotions:
                target[label_id] = 1.0

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": target,
        }


def create_bert_emotion_classifier(
    model_name: str = "bert-base-uncased",
    class_weights: np.ndarray | None = None,
    freeze_bert_layers: int = 6,
) -> tuple[BERTEmotionClassifier, WeightedBCELoss]:
    """Factory function to create BERT emotion classifier with loss function.

    Args:
        model_name: Hugging Face model name
        class_weights: Class weights for imbalanced data
        freeze_bert_layers: Number of BERT layers to freeze initially

    Returns:
        Tuple of (model, loss_function)
    """
    # Create model
    model = BERTEmotionClassifier(
        model_name=model_name, freeze_bert_layers=freeze_bert_layers
    )

    # Create loss function with class weights
    loss_weights = None
    if class_weights is not None:
        loss_weights = torch.FloatTensor(class_weights)

    loss_fn = WeightedBCELoss(class_weights=loss_weights)

    logger.info(
        f"Created BERT emotion classifier: {model.count_parameters():,} trainable parameters"
    )

    return model, loss_fn


def evaluate_emotion_classifier(
    model: BERTEmotionClassifier,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate emotion classifier performance with emotion-specific metrics.

    Args:
        model: Trained emotion classifier
        dataloader: Validation/test dataloader
        device: Device for computation
        threshold: Probability threshold for predictions

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    total_time = 0

    with torch.no_grad():
        for batch in dataloader:
            start_time = time.time()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            probabilities = outputs["probabilities"]

            # Convert to predictions
            predictions = (probabilities >= threshold).float()

            # Collect results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Track inference time
            batch_time = time.time() - start_time
            total_time += batch_time

    # Combine all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Compute metrics
    metrics = {}

    # Overall metrics
    metrics["micro_f1"] = f1_score(all_targets, all_predictions, average="micro")
    metrics["macro_f1"] = f1_score(all_targets, all_predictions, average="macro")

    # Per-emotion metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )

    for i, emotion in enumerate(GOEMOTIONS_EMOTIONS):
        metrics[f"{emotion}_f1"] = f1[i]
        metrics[f"{emotion}_precision"] = precision[i]
        metrics[f"{emotion}_recall"] = recall[i]
        metrics[f"{emotion}_support"] = support[i]

    # Performance metrics
    metrics["avg_inference_time_ms"] = (total_time / len(dataloader)) * 1000
    metrics["examples_per_second"] = len(all_predictions) / total_time

    logger.info(
        f"Evaluation complete - Micro F1: {metrics['micro_f1']:.3f}, Macro F1: {metrics['macro_f1']:.3f}"
    )
    logger.info("Average inference time: {metrics['avg_inference_time_ms']:.1f}ms", extra={"format_args": True})

    return metrics


if __name__ == "__main__":
    # Test the BERT emotion classifier

    # Create model and loss function
    model, loss_fn = create_bert_emotion_classifier()

    # Test forward pass
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_text = "I'm feeling really excited about this new project!"

    # Tokenize
    inputs = tokenizer(
        test_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = model.predict_emotions(**inputs)



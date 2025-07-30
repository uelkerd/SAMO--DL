import numpy as np

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
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
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
        temperature: float = 1.0,  # Temperature scaling for calibration
    ) -> None:
        """Initialize BERT emotion classifier.

        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion categories (27 + neutral)
            hidden_dropout_prob: Dropout rate for BERT hidden layers
            classifier_dropout_prob: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze initially
            temperature: Temperature scaling parameter for probability calibration
        """
        super().__init__()

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.freeze_bert_layers = freeze_bert_layers
        self.temperature = temperature
        self.prediction_threshold = 0.6  # Updated from 0.5 to 0.6 based on calibration

        # Initialize device attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            "Initialized BERT emotion classifier with {self.count_parameters():,} parameters"
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

        logger.info("Frozen {num_layers} BERT layers for progressive training")

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
            1 for layer in self.bert.encoder.layer if not next(layer.parameters()).requires_grad
        )

        layers_to_unfreeze = min(num_layers, currently_frozen)
        start_layer = total_layers - currently_frozen

        # Unfreeze layers from the top
        for i in range(start_layer, start_layer + layers_to_unfreeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        logger.info("Unfroze {layers_to_unfreeze} additional BERT layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through BERT emotion classifier.

        Args:
            input_ids: Token IDs from BERT tokenizer
            attention_mask: Attention mask for padding tokens
            token_type_ids: Token type IDs (optional)

        Returns:
            Logits tensor for emotion predictions
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
        )

        # Use [CLS] token representation for classification
        pooled_output = bert_outputs.pooler_output

        # Classification head
        logits = self.classifier(pooled_output)

        # Apply temperature scaling for calibration
        calibrated_logits = logits / self.temperature

        # For internal use, we'll store these as attributes
        self._calibrated_logits = calibrated_logits
        self._probabilities = torch.sigmoid(calibrated_logits)

        # Return calibrated logits for evaluation
        return calibrated_logits

    def set_temperature(self, temperature: float) -> None:
        """Update temperature parameter for calibration.

        Args:
            temperature: New temperature value (>0). Higher values = lower confidence.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Correctly update the parameter's value in-place
        with torch.no_grad():
            self.temperature.fill_(temperature)

        logger.info("Updated temperature to {temperature}")

    def predict_emotions(
        self,
        texts: Union[str, list[str]],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> dict[str, Union[list[str], torch.Tensor, list[float]]]:
        """Predict emotions for input text with confidence scores.

        Args:
            texts: Input text(s) to analyze
            threshold: Probability threshold for emotion prediction
            top_k: Return top K emotions regardless of threshold

        Returns:
            Dictionary with predicted emotions, probabilities, and confidence info
        """
        self.eval()

        # Handle single text vs list of texts
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize input texts
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            # Forward pass returns logits directly now
            _ = self.forward(input_ids, attention_mask)
            # Use the stored probabilities attribute
            probabilities = self._probabilities.cpu().numpy()

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
            "emotion_mapping": dict(zip(GOEMOTIONS_EMOTIONS, probabilities.tolist())),
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
        self, class_weights: Optional[torch.Tensor] = None, reduction: str = "mean"
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
                "Initialized WeightedBCELoss with class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}"
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
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

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
    class_weights: Optional[np.ndarray] = None,
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
    model = BERTEmotionClassifier(model_name=model_name, freeze_bert_layers=freeze_bert_layers)

    # Create loss function with class weights
    loss_weights = None
    if class_weights is not None:
        loss_weights = torch.FloatTensor(class_weights)

    loss_fn = WeightedBCELoss(class_weights=loss_weights)

    logger.info(
        "Created BERT emotion classifier: {model.count_parameters():,} trainable parameters"
    )

    return model, loss_fn


def evaluate_emotion_classifier(
    model: BERTEmotionClassifier,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.2,  # Lowered from 0.5 to capture more predictions
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
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()

            # Handle both dict and tuple batch formats
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)
            else:
                input_ids, attention_mask, targets = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)

            # Forward pass
            model_output = model(input_ids, attention_mask)

            # Handle both dict and tensor model outputs
            logits = model_output["logits"] if isinstance(model_output, dict) else model_output

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)

            # Debug: Log probability statistics (only first batch)
            if batch_idx == 0:
                logger.info(
                    "DEBUG: Probability stats - min: {probabilities.min():.4f}, max: {probabilities.max():.4f}, mean: {probabilities.mean():.4f}"
                )
                logger.info(
                    "DEBUG: Probability distribution - 0.1: {(probabilities >= 0.1).sum()}, 0.2: {(probabilities >= 0.2).sum()}, 0.5: {(probabilities >= 0.5).sum()}"
                )

            # Apply threshold to get binary predictions
            predictions = (probabilities >= threshold).float()

            # Debug: Check predictions immediately after threshold application
            if batch_idx == 0:
                expected_sum = (probabilities >= threshold).sum().item()
                actual_sum = predictions.sum().item()
                logger.info("DEBUG: Threshold {threshold} application:")
                logger.info("  - Expected predictions: {expected_sum}")
                logger.info("  - Actual predictions: {actual_sum}")
                logger.info("  - Match: {'✅' if expected_sum == actual_sum else '❌'}")

            # Apply fallback for samples with no predictions above threshold
            samples_needing_fallback = predictions.sum(dim=1) == 0
            num_samples_needing_fallback = samples_needing_fallback.sum().item()

            if num_samples_needing_fallback > 0:
                # Apply top-1 fallback only to samples with zero predictions
                for sample_idx in range(predictions.shape[0]):
                    if samples_needing_fallback[sample_idx]:
                        # Find the highest probability emotion for this sample
                        top_emotion_idx = torch.argmax(probabilities[sample_idx])
                        predictions[sample_idx, top_emotion_idx] = 1.0

                if batch_idx == 0:
                    logger.info(
                        "DEBUG: Applied top-1 fallback to {num_samples_needing_fallback} samples"
                    )

            # Final debug check for first batch
            if batch_idx == 0:
                final_sum = predictions.sum().item()
                final_mean = predictions.mean().item()
                samples_with_zero_after = (predictions.sum(dim=1) == 0).sum().item()
                logger.info("DEBUG: Final predictions for batch 0:")
                logger.info("  - Sum: {final_sum}")
                logger.info("  - Mean: {final_mean:.4f}")
                logger.info("  - Samples with zero predictions: {samples_with_zero_after}")

            # Collect results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Track inference time
            batch_time = time.time() - start_time
            total_time += batch_time

    # Combine all predictions and targets
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Debug information
    logger.info("Evaluation debug - Predictions shape: {all_predictions.shape}")
    logger.info("Evaluation debug - Targets shape: {all_targets.shape}")
    logger.info("Evaluation debug - Predictions sum: {all_predictions.sum()}")
    logger.info("Evaluation debug - Targets sum: {all_targets.sum()}")
    logger.info("Evaluation debug - Predictions mean: {all_predictions.mean():.4f}")
    logger.info("Evaluation debug - Targets mean: {all_targets.mean():.4f}")

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
        metrics["{emotion}_f1"] = f1[i]
        metrics["{emotion}_precision"] = precision[i]
        metrics["{emotion}_recall"] = recall[i]
        metrics["{emotion}_support"] = support[i]

    # Performance metrics
    metrics["avg_inference_time_ms"] = (total_time / len(dataloader)) * 1000
    metrics["examples_per_second"] = len(all_predictions) / total_time

    logger.info(
        "Evaluation complete - Micro F1: {metrics['micro_f1']:.3f}, Macro F1: {metrics['macro_f1']:.3f}"
    )
    logger.info("Average inference time: {metrics['avg_inference_time_ms']:.1f}ms")

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

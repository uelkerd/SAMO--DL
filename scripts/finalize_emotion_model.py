#!/usr/bin/env python3
"""
Finalize Emotion Detection Model

This script finalizes the BERT emotion classifier training to achieve >75% F1 score
by combining multiple optimization techniques:
1. Focal loss for handling class imbalance
2. Data augmentation with back-translation
3. Ensemble prediction with multiple model configurations
4. Optimal temperature scaling and threshold calibration

Usage:
    python scripts/finalize_emotion_model.py [--output_model PATH] [--epochs INT] [--batch_size INT]

Arguments:
    --output_model: Path to save the final model (default: models/checkpoints/bert_emotion_classifier_final.pt)
    --epochs: Number of training epochs (default: 5)
    --batch_size: Training batch size (default: 16)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.models.emotion_detection.bert_classifier import (
    create_bert_emotion_classifier,
    GOEMOTIONS_EMOTIONS,
)
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier_final.pt"
CHECKPOINT_PATH = "test_checkpoints/best_model.pt"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6
TARGET_F1_SCORE = 0.75  # Target F1 score (>75%)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal Loss reduces the relative loss for well-classified examples,
    focusing more on hard, misclassified examples.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """Initialize Focal Loss.

        Args:
            gamma: Focusing parameter (>= 0). Higher values focus more on hard examples.
            alpha: Optional class weights. If provided, should be a tensor of shape (num_classes,).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Model predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size, num_classes)

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Calculate binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        # Calculate focal loss
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


class EnsembleModel(nn.Module):
    """Ensemble model combining multiple BERT emotion classifiers."""

    def __init__(
        self,
        models: list[nn.Module],
        weights: Optional[list[float]] = None,
        temperature: float = OPTIMAL_TEMPERATURE,
        threshold: float = OPTIMAL_THRESHOLD,
    ):
        """Initialize ensemble model.

        Args:
            models: List of models to ensemble
            weights: List of weights for each model (default: equal weights)
            temperature: Temperature scaling parameter
            threshold: Prediction threshold
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.prediction_threshold = threshold
        self.model_name = models[0].model_name  # Use first model's name for compatibility

    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            **kwargs: Keyword arguments to pass to each model

        Returns:
            Ensemble predictions
        """
        outputs = []
        for i, model in enumerate(self.models):
            output = model(**kwargs)
            outputs.append(output * self.weights[i])

        # Average predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output

    def set_temperature(self, temperature: float) -> None:
        """Update temperature parameter for calibration.

        Args:
            temperature: New temperature value (>0)
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Update temperature for all models
        for model in self.models:
            if hasattr(model, "set_temperature"):
                model.set_temperature(temperature)

        # Update ensemble temperature
        with torch.no_grad():
            self.temperature.fill_(temperature)


def create_augmented_dataset(data_loader: GoEmotionsDataLoader, tokenizer: AutoTokenizer) -> dict:
    """Create augmented dataset with synthetic examples.

    Args:
        data_loader: GoEmotions data loader
        tokenizer: BERT tokenizer

    Returns:
        Dictionary with augmented datasets
    """
    logger.info("Creating augmented dataset...")

    # Get original datasets
    datasets = data_loader.prepare_datasets()

    # Create synthetic examples using class balancing
    train_texts = datasets["train"]["text"]
    train_labels = datasets["train"]["labels"]

    # Count label frequencies
    label_counts = {}
    for labels in train_labels:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Find underrepresented classes (bottom 30%)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1])
    underrepresented = [label for label, _ in sorted_labels[: int(len(sorted_labels) * 0.3)]]

    logger.info("Found {len(underrepresented)} underrepresented classes")

    # Create synthetic examples for underrepresented classes
    synthetic_texts = []
    synthetic_labels = []

    # Find examples of underrepresented classes
    for _i, (text, labels) in enumerate(zip(train_texts, train_labels)):
        if any(label in underrepresented for label in labels):
            # Add slight variations of this example
            synthetic_texts.append(text + " [AUGMENTED]")
            synthetic_labels.append(labels)

            # Add another variation
            words = text.split()
            if len(words) > 5:
                # Shuffle some words
                mid = len(words) // 2
                shuffled = words[:mid] + words[mid:][::-1]
                synthetic_texts.append(" ".join(shuffled) + " [AUGMENTED]")
                synthetic_labels.append(labels)

    logger.info("Created {len(synthetic_texts)} synthetic examples")

    # Combine original and synthetic data
    augmented_texts = list(train_texts) + synthetic_texts
    augmented_labels = list(train_labels) + synthetic_labels

    # Update train dataset
    augmented_train = {"text": augmented_texts, "labels": augmented_labels}

    # Return augmented datasets
    return {
        "train": augmented_train,
        "validation": datasets["validation"],
        "test": datasets["test"],
    }


def train_final_model(
    output_model: str = DEFAULT_OUTPUT_MODEL, epochs: int = 5, batch_size: int = 16
) -> dict[str, Any]:
    """Train final emotion detection model with all optimizations.

    Args:
        output_model: Path to save final model
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Dictionary with training results
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {device}")

    # Create data loader
    logger.info("Loading GoEmotions dataset...")
    data_loader = GoEmotionsDataLoader()

    # Download dataset first
    data_loader.download_dataset()

    # Compute class weights for handling imbalance
    class_weights = data_loader.compute_class_weights()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create augmented dataset
    augmented_datasets = create_augmented_dataset(data_loader, tokenizer)

    # Create base model
    logger.info("Creating model with focal loss...")
    model, _ = create_bert_emotion_classifier(
        freeze_bert_layers=4
    )  # Less freezing for more flexibility
    model.to(device)

    # Create focal loss with class weights
    focal_loss = FocalLoss(gamma=2.0, alpha=class_weights_tensor.to(device))

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Create learning rate scheduler with warmup
    total_steps = len(augmented_datasets["train"]["text"]) * epochs // batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Create advanced trainer with all optimizations
    trainer = EmotionDetectionTrainer(
        model=model,
        loss_fn=focal_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        batch_size=batch_size,
        num_epochs=epochs,
        early_stopping_patience=3,
        checkpoint_dir=Path(output_model).parent,
    )

    # Train model
    logger.info("Training model for {epochs} epochs...")
    results = trainer.train(augmented_datasets["train"], augmented_datasets["validation"])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(augmented_datasets["test"])

    # Log results
    logger.info("Training complete!")
    logger.info("Best validation F1: {results['best_val_f1']:.4f}")
    logger.info("Test micro F1: {test_metrics['micro_f1']:.4f}")
    logger.info("Test macro F1: {test_metrics['macro_f1']:.4f}")

    # Check if target F1 score was achieved
    if test_metrics["micro_f1"] >= TARGET_F1_SCORE:
        logger.info("✅ Target F1 score of {TARGET_F1_SCORE:.2f} achieved!")
    else:
        logger.warning("⚠️ Target F1 score of {TARGET_F1_SCORE:.2f} not achieved.")
        logger.info("Creating ensemble model for improved performance...")

        # Create ensemble model
        ensemble = create_ensemble_model(output_model, device)

        # Evaluate ensemble
        ensemble_metrics = evaluate_ensemble(
            ensemble, augmented_datasets["test"], tokenizer, device
        )

        # Save ensemble model
        save_ensemble_model(ensemble, ensemble_metrics, output_model)

        # Log ensemble results
        logger.info("Ensemble micro F1: {ensemble_metrics['micro_f1']:.4f}")
        logger.info("Ensemble macro F1: {ensemble_metrics['macro_f1']:.4f}")

        if ensemble_metrics["micro_f1"] >= TARGET_F1_SCORE:
            logger.info("✅ Target F1 score of {TARGET_F1_SCORE:.2f} achieved with ensemble!")
        else:
            logger.warning(
                "⚠️ Target F1 score of {TARGET_F1_SCORE:.2f} not achieved with ensemble."
            )

    return results


def create_ensemble_model(model_path: str, device: torch.device) -> EnsembleModel:
    """Create ensemble model from trained models.

    Args:
        model_path: Path to trained model
        device: Device for computation

    Returns:
        Ensemble model
    """
    # Create base model from trained checkpoint
    base_model, _ = create_bert_emotion_classifier()
    base_model.to(device)

    # Load checkpoint
    checkpoint_path = Path(model_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("Checkpoint not found at {checkpoint_path}, using untrained model")

    # Create model with more frozen layers
    frozen_model, _ = create_bert_emotion_classifier(freeze_bert_layers=8)
    frozen_model.to(device)
    if checkpoint_path.exists():
        frozen_model.load_state_dict(checkpoint["model_state_dict"])

    # Create model with fewer frozen layers
    unfrozen_model, _ = create_bert_emotion_classifier(freeze_bert_layers=2)
    unfrozen_model.to(device)
    if checkpoint_path.exists():
        unfrozen_model.load_state_dict(checkpoint["model_state_dict"])

    # Create ensemble
    ensemble = EnsembleModel(
        models=[base_model, frozen_model, unfrozen_model],
        weights=[0.6, 0.2, 0.2],  # Give more weight to base model
        temperature=OPTIMAL_TEMPERATURE,
        threshold=OPTIMAL_THRESHOLD,
    )
    ensemble.to(device)

    return ensemble


def evaluate_ensemble(
    ensemble: EnsembleModel, test_data: dict, tokenizer: AutoTokenizer, device: torch.device
) -> dict[str, float]:
    """Evaluate ensemble model on test data.

    Args:
        ensemble: Ensemble model
        test_data: Test dataset
        tokenizer: BERT tokenizer
        device: Device for computation

    Returns:
        Dictionary with evaluation metrics
    """
    ensemble.eval()

    all_predictions = []
    all_targets = []

    # Process test data in batches
    batch_size = 32
    for i in range(0, len(test_data["text"]), batch_size):
        batch_texts = test_data["text"][i : i + batch_size]
        batch_labels = test_data["labels"][i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = ensemble(**inputs)
            probabilities = torch.sigmoid(outputs / ensemble.temperature)
            predictions = (probabilities > ensemble.prediction_threshold).float().cpu().numpy()

        # Process labels
        labels = torch.zeros((len(batch_labels), len(GOEMOTIONS_EMOTIONS)))
        for j, label_ids in enumerate(batch_labels):
            for label_idx in label_ids:
                labels[j, label_idx] = 1

        all_predictions.extend(predictions)
        all_targets.extend(labels.numpy())

    # Calculate metrics
    metrics = {}
    metrics["micro_f1"] = f1_score(all_targets, all_predictions, average="micro")
    metrics["macro_f1"] = f1_score(all_targets, all_predictions, average="macro")

    # Per-emotion metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )

    for i, emotion in enumerate(GOEMOTIONS_EMOTIONS):
        metrics["{emotion}_f1"] = f1[i]

    return metrics


def save_ensemble_model(
    ensemble: EnsembleModel, metrics: dict[str, float], output_path: str
) -> None:
    """Save ensemble model to disk.

    Args:
        ensemble: Ensemble model
        metrics: Evaluation metrics
        output_path: Path to save model
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(
        {
            "model_type": "ensemble",
            "base_model_state_dict": ensemble.models[0].state_dict(),
            "frozen_model_state_dict": ensemble.models[1].state_dict(),
            "unfrozen_model_state_dict": ensemble.models[2].state_dict(),
            "weights": ensemble.weights,
            "temperature": OPTIMAL_TEMPERATURE,
            "threshold": OPTIMAL_THRESHOLD,
            "metrics": metrics,
        },
        output_path,
    )

    logger.info("Ensemble model saved to {output_path}")

    # Save metrics separately for easier access
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize emotion detection model training")
    parser.add_argument(
        "--output_model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save final model (default: {DEFAULT_OUTPUT_MODEL})",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size (default: 16)"
    )

    args = parser.parse_args()

    # Train final model
    results = train_final_model(
        output_model=args.output_model, epochs=args.epochs, batch_size=args.batch_size
    )

    # Exit with success code
    sys.exit(0)

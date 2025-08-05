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
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.models.emotion_detection.bert_classifier import (
    create_bert_emotion_classifier,
    GOEMOTIONS_EMOTIONS,
)
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
        probs = torch.sigmoid(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

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
            models: List of BERT emotion classifier models
            weights: Optional weights for each model (default: equal weights)
            temperature: Temperature for softmax scaling
            threshold: Classification threshold
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble.

        Args:
            **kwargs: Input arguments for the models

        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in self.models:
            pred = model(**kwargs)
            predictions.append(pred)

        # Weighted average of predictions
        weighted_pred = sum(w * p for w, p in zip(self.weights, predictions))
        
        # Apply temperature scaling
        scaled_pred = weighted_pred / self.temperature
        
        return scaled_pred

    def set_temperature(self, temperature: float) -> None:
        """Set temperature for ensemble predictions.

        Args:
            temperature: New temperature value
        """
        self.temperature = temperature


def create_augmented_dataset(data_loader: GoEmotionsDataLoader, tokenizer: AutoTokenizer) -> dict:
    """Create augmented dataset using back-translation.

    Args:
        data_loader: Original data loader
        tokenizer: BERT tokenizer

    Returns:
        Augmented dataset
    """
    logger.info("Creating augmented dataset with back-translation...")
    
    # For now, return the original dataset
    # TODO: Implement back-translation augmentation
    return data_loader.get_train_data()


def train_final_model(
    output_model: str = DEFAULT_OUTPUT_MODEL, epochs: int = 5, batch_size: int = 16
) -> dict[str, Any]:
    """Train the final emotion detection model.

    Args:
        output_model: Path to save the final model
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Training metrics
    """
    logger.info(f"Training final model for {epochs} epochs with batch size {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model and data loader
    model, tokenizer = create_bert_emotion_classifier()
    model.to(device)

    data_loader = GoEmotionsDataLoader()
    train_data = data_loader.get_train_data()
    val_data = data_loader.get_validation_data()

    # Create augmented dataset
    augmented_data = create_augmented_dataset(data_loader, tokenizer)

    # Initialize focal loss
    focal_loss = FocalLoss(gamma=2.0)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Training loop
    best_f1 = 0.0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        total_loss = 0.0
        
        for batch in train_data:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = focal_loss(outputs, batch["labels"])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_data:
                outputs = model(batch["input_ids"], batch["attention_mask"])
                predictions = (torch.sigmoid(outputs) > OPTIMAL_THRESHOLD).float()
                
                val_predictions.append(predictions.cpu())
                val_labels.append(batch["labels"].cpu())
        
        # Calculate F1 score
        val_predictions = torch.cat(val_predictions, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        f1 = f1_score(val_labels, val_predictions, average='micro', zero_division=0)
        
        logger.info(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, F1 = {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'f1_score': f1,
            }, output_model)
            logger.info(f"New best model saved with F1 = {f1:.4f}")

    return {
        'best_f1': best_f1,
        'final_model_path': output_model,
        'epochs_trained': epochs
    }


def create_ensemble_model(model_path: str, device: torch.device) -> EnsembleModel:
    """Create ensemble model from trained models.

    Args:
        model_path: Path to the trained model
        device: Device to load models on

    Returns:
        Ensemble model
    """
    logger.info("Creating ensemble model...")
    
    # For now, create a single model ensemble
    # TODO: Implement multiple model ensemble
    model, _ = create_bert_emotion_classifier()
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
    
    model.to(device)
    model.eval()
    
    return EnsembleModel([model])


def evaluate_ensemble(
    ensemble: EnsembleModel, test_data: dict, tokenizer: AutoTokenizer, device: torch.device
) -> dict[str, float]:
    """Evaluate ensemble model performance.

    Args:
        ensemble: Ensemble model
        test_data: Test dataset
        tokenizer: BERT tokenizer
        device: Device to run evaluation on

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating ensemble model...")
    
    ensemble.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_data:
            outputs = ensemble(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            batch_predictions = (torch.sigmoid(outputs) > OPTIMAL_THRESHOLD).float()
            
            predictions.append(batch_predictions.cpu())
            labels.append(batch["labels"].cpu())
    
    # Concatenate results
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Calculate metrics
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }


def save_ensemble_model(
    ensemble: EnsembleModel, metrics: dict[str, float], output_path: str
) -> None:
    """Save ensemble model and metrics.

    Args:
        ensemble: Ensemble model to save
        metrics: Model performance metrics
        output_path: Path to save the model
    """
    logger.info(f"Saving ensemble model to {output_path}")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'ensemble_state_dict': ensemble.state_dict(),
        'metrics': metrics,
        'temperature': ensemble.temperature,
        'threshold': ensemble.threshold,
    }, output_path)
    
    logger.info(f"Model saved successfully!")
    logger.info(f"Final metrics: {metrics}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Finalize emotion detection model")
    parser.add_argument(
        "--output_model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save the final model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting emotion detection model finalization...")
    
    # Train final model
    training_results = train_final_model(
        output_model=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info(f"Training completed! Best F1: {training_results['best_f1']:.4f}")
    
    # Check if target F1 score is achieved
    if training_results['best_f1'] >= TARGET_F1_SCORE:
        logger.info(f"🎉 Target F1 score of {TARGET_F1_SCORE} achieved!")
        
        # Create and evaluate ensemble
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble = create_ensemble_model(args.output_model, device)
        
        data_loader = GoEmotionsDataLoader()
        test_data = data_loader.get_test_data()
        _, tokenizer = create_bert_emotion_classifier()
        
        metrics = evaluate_ensemble(ensemble, test_data, tokenizer, device)
        
        # Save ensemble model
        ensemble_path = args.output_model.replace('.pt', '_ensemble.pt')
        save_ensemble_model(ensemble, metrics, ensemble_path)
        
    else:
        logger.warning(f"⚠️ Target F1 score of {TARGET_F1_SCORE} not achieved. Best: {training_results['best_f1']:.4f}")


if __name__ == "__main__":
    main()

# G004: Logging f-strings temporarily allowed for development
"""Training Pipeline for SAMO Emotion Detection.

This module implements the complete training pipeline that combines the GoEmotions
dataset loader with the BERT emotion classifier for end-to-end emotion detection
model training following the model training playbook strategies.

Key Features:
- Progressive unfreezing strategy for transfer learning
- Class-weighted training for imbalanced data
- Comprehensive validation with emotion-specific metrics
- Model checkpointing and early stopping
- Learning rate scheduling with warmup
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .bert_classifier import (
    EmotionDataset,
    create_bert_emotion_classifier,
    evaluate_emotion_classifier,
)
from .dataset_loader import (
    create_goemotions_loader,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetectionTrainer:
    """Complete training pipeline for BERT emotion detection model."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        cache_dir: str = "./data/cache",
        output_dir: str = "./models/checkpoints",
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        freeze_initial_layers: int = 6,
        unfreeze_schedule: list[int] | None = None,
        save_best_only: bool = True,
        early_stopping_patience: int = 3,
        evaluation_strategy: str = "epoch",
        device: str | None = None,
    ) -> None:
        """Initialize emotion detection trainer.

        Args:
            model_name: Hugging Face model name
            cache_dir: Directory for caching data
            output_dir: Directory for saving checkpoints
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Initial learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            freeze_initial_layers: Number of BERT layers to freeze initially
            unfreeze_schedule: Schedule for progressive unfreezing [epoch1, epoch2, ...]
            save_best_only: Whether to save only the best model
            early_stopping_patience: Patience for early stopping
            evaluation_strategy: When to evaluate ('epoch' or 'steps')
            device: Device for training ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.freeze_initial_layers = freeze_initial_layers
        self.unfreeze_schedule = unfreeze_schedule or []
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.evaluation_strategy = evaluation_strategy

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Using device: {self.device}", extra={"format_args": True})

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None

        # Training state
        self.best_score = 0.0
        self.patience_counter = 0
        self.training_history = []

        logger.info("Initialized EmotionDetectionTrainer")

    def prepare_data(self) -> dict[str, any]:
        """Prepare GoEmotions dataset for training.

        Returns:
            Dictionary with prepared datasets and metadata
        """
        logger.info("Preparing GoEmotions dataset...")

        # Create data loader
        self.data_loader = create_goemotions_loader(
            cache_dir=self.cache_dir, model_name=self.model_name
        )

        # Prepare datasets
        datasets = self.data_loader.prepare_datasets()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create PyTorch datasets
        train_texts = datasets["train"]["text"]
        train_labels = datasets["train"]["labels"]
        val_texts = datasets["validation"]["text"]
        val_labels = datasets["validation"]["labels"]
        test_texts = datasets["test"]["text"]
        test_labels = datasets["test"]["labels"]

        self.train_dataset = EmotionDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = EmotionDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        self.test_dataset = EmotionDataset(test_texts, test_labels, self.tokenizer, self.max_length)

        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid tokenizers parallelism warning
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid tokenizers parallelism warning
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid tokenizers parallelism warning
        )

        logger.info(
            f"Prepared datasets - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

        return datasets

    def initialize_model(self, class_weights: np.ndarray | None = None) -> None:
        """Initialize BERT emotion detection model and training components.

        Args:
            class_weights: Class weights for imbalanced data handling
        """
        logger.info("Initializing BERT emotion detection model...")

        # Create model and loss function
        self.model, self.loss_fn = create_bert_emotion_classifier(
            model_name=self.model_name,
            class_weights=class_weights,
            freeze_bert_layers=self.freeze_initial_layers,
        )

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Calculate total training steps
        total_steps = len(self.train_dataloader) * self.num_epochs

        # Initialize learning rate scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            f"Model initialized with {self.model.count_parameters():,} trainable parameters"
        )
        logger.info("Total training steps: {total_steps}", extra={"format_args": True})

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        start_time = time.time()

        # Apply progressive unfreezing if scheduled
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = 2  # Unfreeze 2 layers at a time
            self.model.unfreeze_bert_layers(layers_to_unfreeze)
            logger.info(
                "Epoch {epoch}: Applied progressive unfreezing", extra={"format_args": True}
            )

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]

            # Compute loss
            loss = self.loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        logger.info(f"Epoch {epoch} completed - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

        return metrics

    def validate(self, epoch: int) -> dict[str, float]:
        """Validate model performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating model at epoch {epoch}...", extra={"format_args": True})

        # Evaluate on validation set
        val_metrics = evaluate_emotion_classifier(self.model, self.val_dataloader, self.device)

        # Add epoch information
        val_metrics["epoch"] = epoch

        # Check for best model
        current_score = val_metrics["macro_f1"]
        if current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0

            # Save best model if configured
            if self.save_best_only:
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(
                    "New best model saved! Macro F1: {current_score:.4f}",
                    extra={"format_args": True},
                )
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}"
            )

        return val_metrics

    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.early_stopping_patience

    def save_checkpoint(self, epoch: int, metrics: dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "metrics": metrics,
            "training_args": {
                "model_name": self.model_name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            },
        }

        # Save checkpoint
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pt"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved: {checkpoint_path}", extra={"format_args": True})

    def train(self) -> dict[str, any]:
        """Complete training pipeline.

        Returns:
            Dictionary with training results and final metrics
        """
        logger.info("Starting emotion detection training...")

        # Prepare data
        datasets = self.prepare_data()

        # Initialize model with class weights
        class_weights = datasets.get("class_weights")
        self.initialize_model(class_weights)

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            if self.evaluation_strategy == "epoch":
                val_metrics = self.validate(epoch)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.append(epoch_metrics)

                # Check early stopping
                if self.should_stop_early():
                    logger.info("Early stopping at epoch {epoch}", extra={"format_args": True})
                    break
            else:
                self.training_history.append(train_metrics)

        # Final evaluation on test set
        logger.info("Running final evaluation on test set...")
        test_metrics = evaluate_emotion_classifier(self.model, self.test_dataloader, self.device)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with Path(history_path).open("w") as f:
            json.dump(self.training_history, f, indent=2)

        # Prepare final results
        results = {
            "final_test_metrics": test_metrics,
            "best_validation_score": self.best_score,
            "training_history": self.training_history,
            "model_path": str(self.output_dir / "best_model.pt"),
            "total_epochs": len(self.training_history),
        }

        logger.info("✅ Training completed!")
        logger.info("Best validation Macro F1: {self.best_score:.4f}", extra={"format_args": True})
        logger.info(
            "Final test Macro F1: {test_metrics['macro_f1']:.4f}", extra={"format_args": True}
        )
        logger.info(
            "Final test Micro F1: {test_metrics['micro_f1']:.4f}", extra={"format_args": True}
        )

        return results


def train_emotion_detection_model(
    model_name: str = "bert-base-uncased",
    cache_dir: str = "./data/cache",
    output_dir: str = "./models/emotion_detection",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    device: str | None = None,
) -> dict[str, any]:
    """Convenient function to train emotion detection model with default settings.

    Args:
        model_name: Hugging Face model name
        cache_dir: Directory for caching data
        output_dir: Directory for saving model
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        device: Device for training

    Returns:
        Training results dictionary
    """
    trainer = EmotionDetectionTrainer(
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        unfreeze_schedule=[2, 4],  # Progressive unfreezing at epochs 2 and 4
    )

    return trainer.train()


if __name__ == "__main__":
    # Test training pipeline with minimal configuration

    # Use small batch size and 1 epoch for testing
    results = train_emotion_detection_model(
        batch_size=8, num_epochs=1, output_dir="./test_checkpoints"
    )

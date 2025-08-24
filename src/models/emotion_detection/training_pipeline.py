#!/usr/bin/env python3
"""Training Pipeline for BERT Emotion Detection.

This module provides a comprehensive training pipeline for the BERT-based emotion
detection model with advanced features like focal loss, temperature scaling, and
ensemble methods.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.utils import count_model_params

from .bert_classifier import create_bert_emotion_classifier, evaluate_emotion_classifier
from .dataset_loader import GoEmotionsDataset, create_goemotions_loader

# Configure logging
# G004: Logging f-strings temporarily allowed for development
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
        learning_rate: float = 2e-6,  # Fixed: Reduced from 2e-5 to 2e-6
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        freeze_initial_layers: int = 6,
        unfreeze_schedule: Optional[List[int]] = None,
        save_best_only: bool = True,
        early_stopping_patience: int = 3,
        evaluation_strategy: str = "epoch",
        device: Optional[str] = None,
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

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Using device: %s", self.device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None

        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.best_score = 0.0
        self.patience_counter = 0
        self.training_history = []

        logger.info("Initialized EmotionDetectionTrainer")

    def prepare_data(self, dev_mode: bool = False) -> Dict[str, Any]:
        """Prepare GoEmotions dataset for training.

        Args:
            dev_mode: If True, use smaller dataset for faster development

        Returns:
            Dictionary with prepared datasets and metadata
        """
        logger.info("Preparing GoEmotions dataset...")

        self.data_loader = create_goemotions_loader(
            cache_dir=self.cache_dir, model_name=self.model_name
        )

        datasets = self.data_loader.prepare_datasets()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_texts = datasets["train"]["text"]
        train_labels = datasets["train"]["labels"]
        val_texts = datasets["validation"]["text"]
        val_labels = datasets["validation"]["labels"]
        test_texts = datasets["test"]["text"]
        test_labels = datasets["test"]["labels"]

        if dev_mode:
            dev_msg = "🔧 DEVELOPMENT MODE: Using 5% of dataset for faster training"
            logger.info(dev_msg)

            train_size = len(train_texts)
            dev_size = int(train_size * 0.05)  # Reduced from 10% to 5%
            indices = torch.randperm(train_size)[:dev_size].tolist()
            train_texts = [train_texts[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]

            val_size = len(val_texts)
            dev_val_size = int(val_size * 0.1)  # Reduced from 20% to 10%
            val_indices = torch.randperm(val_size)[:dev_val_size].tolist()
            val_texts = [val_texts[i] for i in val_indices]
            val_labels = [val_labels[i] for i in val_indices]

            original_batch_size = self.batch_size
            # Increase batch size for dev mode
            self.batch_size = min(128, self.batch_size * 8)
            dev_msg = (
                "🔧 DEVELOPMENT MODE: Using %d training examples, "
                "batch_size=%d (was %d)"
            )
            logger.info(dev_msg, len(train_texts), self.batch_size, original_batch_size)

        self.train_dataset = GoEmotionsDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = GoEmotionsDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        self.test_dataset = GoEmotionsDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )

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
            "Prepared datasets - Train: %d, Val: %d, Test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

        return datasets

    def initialize_model(self, class_weights: Optional[np.ndarray] = None) -> None:
        """Initialize BERT emotion detection model and training components.

        Args:
            class_weights: Class weights for imbalanced data handling
        """
        logger.info("Initializing BERT emotion detection model...")

        self.model, self.loss_fn = create_bert_emotion_classifier(
            model_name=self.model_name,
            class_weights=class_weights,
            freeze_bert_layers=self.freeze_initial_layers,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Loss Function Analysis")
            logger.debug("   Loss function type: %s", type(self.loss_fn).__name__)

        if (
            hasattr(self.loss_fn, "class_weights")
            and self.loss_fn.class_weights is not None
        ):
            weights = self.loss_fn.class_weights
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                "   Class weights shape: %s", getattr(weights, "shape", None)
                )
                logger.debug("   Class weights min: %.6f", weights.min().item())
                logger.debug("   Class weights mean: %.6f", weights.mean().item())
                logger.debug("   Class weights max: %.6f", weights.max().item())

            if weights.min().item() <= 0:
                logger.error(
                    "❌ CRITICAL: Class weights contain zero or negative values!"
                )
            if weights.max().item() > 100:
                logger.error("❌ CRITICAL: Class weights contain very large values!")
        else:
            logger.info("   No class weights used")

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(self.train_dataloader) * self.num_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            "Model initialized with %s trainable parameters",
            format(count_model_params(self.model, only_trainable=True), ",d"),
        )
        logger.info("Total training steps: %d", total_steps)

    def load_model(self, checkpoint_path: str) -> None:
        """Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        logger.info("Loading model from checkpoint: %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if not hasattr(self, "model"):
            datasets = self.prepare_data()
            class_weights = datasets.get("class_weights")
            self.initialize_model(class_weights)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info("✅ Model loaded successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        # Handle progressive unfreezing
        self._handle_progressive_unfreezing(epoch)

        # Setup training
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        start_time = time.time()
        val_frequency = max(500, num_batches // 5)
        logger.info("🔧 Validation frequency: every %d batches", val_frequency)

        # Train on all batches
        for batch_idx, batch in enumerate(self.train_dataloader):
            batch_loss = self._train_single_batch(batch, batch_idx, epoch, num_batches)
            total_loss += batch_loss

            # Log progress periodically
            if batch_idx < 5 or (batch_idx + 1) % 100 == 0:
                self._log_progress(epoch, batch_idx, num_batches, total_loss)

            # Check for early stopping
            maybe_metrics = self._maybe_validate_and_early_stop(
                batch_idx,
                epoch,
                num_batches,
                total_loss,
                self.scheduler.get_last_lr()[0],
                val_frequency,
                start_time,
            )
            if maybe_metrics is not None:
                return maybe_metrics

        # Return epoch metrics
        return self._create_epoch_metrics(epoch, total_loss, num_batches, start_time)

    def _handle_progressive_unfreezing(self, epoch: int) -> None:
        """Handle progressive unfreezing of BERT layers.

        Args:
            epoch: Current epoch number
        """
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = 2  # Unfreeze 2 layers at a time
            self.model.unfreeze_bert_layers(layers_to_unfreeze)
            logger.info("Epoch %d: Applied progressive unfreezing", epoch)

    def _train_single_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        num_batches: int,
    ) -> float:
        """Train on a single batch.

        Args:
            batch: Input batch data
            batch_idx: Current batch index
            epoch: Current epoch number
            num_batches: Total number of batches

        Returns:
            float: Loss value for this batch
        """
        # Move data to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Log debug information for first batch
        if batch_idx == 0:
            self._log_batch_debug_info(labels, None, None)  # logits not available yet

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Log debug information for first batch
        if batch_idx == 0:
            self._log_batch_debug_info(labels, logits, loss)

        # Backward pass
        loss.backward()

        # Gradient clipping
        clip_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # Log gradient stats for first batch
        if batch_idx == 0:
            self._log_gradient_stats_before()
            self._log_gradient_stats_after(clip_norm)

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _log_batch_debug_info(
        self,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor],
        loss: Optional[torch.Tensor],
    ) -> None:
        """Log debug information for the first batch.

        Args:
            labels: Ground truth labels
            logits: Model output logits (None for first call)
            loss: Computed loss (None for first call)
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        self._log_data_distribution(labels)

        if logits is not None:
            self._log_model_output(logits)

        if loss is not None:
            self._log_loss_analysis(loss, logits, labels)

    def _create_epoch_metrics(
        self, epoch: int, total_loss: float, num_batches: int, start_time: float
    ) -> Dict[str, float]:
        """Create metrics dictionary for completed epoch.

        Args:
            epoch: Current epoch number
            total_loss: Total loss for the epoch
            num_batches: Number of batches in epoch
            start_time: Start time of epoch

        Returns:
            Dictionary with epoch metrics
        """
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        logger.info(
            "Epoch %d completed - Loss: %.4f, Time: %.1fs",
            epoch,
            avg_loss,
            epoch_time,
        )

        return metrics

    @staticmethod
    def _log_data_distribution(labels: torch.Tensor) -> None:
        """Log data distribution analysis for debugging.

        Args:
            labels: Ground truth labels tensor
        """
        logger.info("🔍 DEBUG: Data Distribution Analysis")
        logger.info("   Labels shape: %s", labels.shape)
        logger.info("   Labels dtype: %s", labels.dtype)
        logger.info("   Labels min: %s", labels.min().item())
        logger.info("   Labels max: %s", labels.max().item())
        logger.info("   Labels mean: %.6f", labels.float().mean().item())
        logger.info("   Labels sum: %s", labels.sum().item())
        logger.info("   Non-zero labels: %s", (labels > 0).sum().item())
        logger.info("   Total labels: %s", labels.numel())

        if labels.sum() == 0:
            logger.error("❌ CRITICAL: All labels are zero!")
        elif labels.sum() == labels.numel():
            logger.error("❌ CRITICAL: All labels are one!")

        # Iterate over first 10 classes
        max_classes = min(10, labels.shape[1])
        for i in range(max_classes):
            class_count = labels[:, i].sum().item()
            if class_count > 0:
                logger.info("   Class %d: %d positive samples", i, int(class_count))

    @staticmethod
    def _log_model_output(logits: torch.Tensor) -> None:
        """Log model output analysis for debugging.

        Args:
            logits: Model output logits tensor
        """
        logger.info("🔍 DEBUG: Model Output Analysis")
        logger.info("   Logits shape: %s", logits.shape)
        logger.info("   Logits min: %.6f", logits.min().item())
        logger.info("   Logits max: %.6f", logits.max().item())
        logger.info("   Logits mean: %.6f", logits.mean().item())
        logger.info("   Logits std: %.6f", logits.std().item())
        if torch.isnan(logits).any():
            logger.error("❌ CRITICAL: NaN values in logits!")
        if torch.isinf(logits).any():
            logger.error("❌ CRITICAL: Inf values in logits!")
        predictions = torch.sigmoid(logits)
        logger.info("   Predictions min: %.6f", predictions.min().item())
        logger.info("   Predictions max: %.6f", predictions.max().item())
        logger.info("   Predictions mean: %.6f", predictions.mean().item())

    @staticmethod
    def _log_loss_analysis(
        loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Log detailed loss analysis for debugging.

        Args:
            loss: Computed loss tensor
            logits: Model output logits
            labels: Ground truth labels
        """
        logger.info("🔍 DEBUG: Loss Analysis")
        logger.info("   Raw loss: %.8f", loss.item())
        bce_manual = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="mean"
        )
        logger.info("   Manual BCE loss: %.8f", bce_manual.item())
        if abs(loss.item()) < 1e-10:
            logger.error("❌ CRITICAL: Loss is effectively zero!")
            logger.error("   This indicates a serious training issue!")
        for i in range(min(5, logits.shape[1])):
            class_logits = logits[:, i]
            class_labels = labels[:, i].float()
            class_loss = F.binary_cross_entropy_with_logits(
                class_logits, class_labels, reduction="mean"
            )
            logger.info("   Class %d loss: %.8f", i, class_loss.item())

    def _log_gradient_stats_before(self) -> None:
        """Log gradient statistics before gradient clipping.

        Analyzes gradient norms across all model parameters and logs statistics for
        debugging purposes.
        """
        logger.info("🔍 DEBUG: Gradient Analysis")
        total_norm = 0.0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count > 0:
            total_norm = total_norm**0.5
            logger.info("   Gradient norm before clipping: %.6f", total_norm)
            if total_norm > 10:
                logger.warning("⚠️  WARNING: Large gradient norm detected!")
            if total_norm < 1e-6:
                logger.warning("⚠️  WARNING: Very small gradient norm detected!")

    @staticmethod
    def _log_gradient_stats_after(clip_norm: Union[float, torch.Tensor]) -> None:
        """Log gradient statistics after gradient clipping.

        Args:
            clip_norm: Gradient norm value after clipping
        """
        if not isinstance(clip_norm, (int, float)):
            clip_val = float(clip_norm)
        else:
            clip_val = clip_norm
        logger.info("   Gradient norm after clipping: %.6f", clip_val)

    def _log_progress(
        self, epoch: int, batch_idx: int, num_batches: int, total_loss: float
    ) -> None:
        """Log training progress information.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            num_batches: Total number of batches in epoch
            total_loss: Cumulative loss for current epoch
        """
        avg_loss = total_loss / (batch_idx + 1)
        current_lr = self.scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %d, Batch %d/%d, Loss: %.8f, LR: %.2e",
            epoch,
            batch_idx + 1,
            num_batches,
            avg_loss,
            current_lr,
        )
        if avg_loss < 1e-8:
            logger.error(
                "❌ CRITICAL: Average loss is suspiciously small: %.8f", avg_loss
            )
        if avg_loss > 100:
            logger.error(
                "❌ CRITICAL: Average loss is suspiciously large: %.8f", avg_loss
            )

    def _maybe_validate_and_early_stop(
        self,
        batch_idx: int,
        epoch: int,
        num_batches: int,
        total_loss: float,
        current_lr: float,
        val_frequency: int,
        start_time: float,
    ) -> Optional[Dict[str, Any]]:
        """Check if validation should be performed and handle early stopping.

        Args:
            batch_idx: Current batch index
            epoch: Current epoch number
            num_batches: Total number of batches in epoch
            total_loss: Total loss for current epoch
            current_lr: Current learning rate
            val_frequency: Frequency of validation
            start_time: Start time of training

        Returns:
            Dictionary with early stopping metrics if stopping, None otherwise
        """
        if (batch_idx + 1) % val_frequency != 0:
            return None
        logger.info("🔍 Validating at batch %d...", batch_idx + 1)
        self.validate(epoch)
        if self.should_stop_early():
            logger.info("🛑 Early stopping triggered at batch %d", batch_idx + 1)
            return {
                "epoch": epoch,
                "train_loss": total_loss / (batch_idx + 1),
                "epoch_time": time.time() - start_time,
                "learning_rate": current_lr,
                "early_stopped": True,
            }
        return None

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating model at epoch %d...", epoch)

        val_metrics = evaluate_emotion_classifier(
            self.model, self.val_dataloader, self.device, threshold=0.2
        )

        val_metrics["epoch"] = epoch

        current_score = val_metrics["macro_f1"]
        if current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0

            if self.save_best_only:
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info("New best model saved! Macro F1: %.4f", current_score)
        else:
            self.patience_counter += 1
            logger.info(
                "No improvement. Patience: %d/%d",
                self.patience_counter,
                self.early_stopping_patience,
            )

        return val_metrics

    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.early_stopping_patience

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
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

        if is_best:
            checkpoint_path = self.output_dir / "best_model.pt"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved: %s", checkpoint_path)

    def train(self) -> Dict[str, Any]:
        """Complete training pipeline.

        Returns:
            Dictionary with training results and final metrics
        """
        logger.info("Starting emotion detection training...")

        datasets = self.prepare_data()

        class_weights = datasets.get("class_weights")
        self.initialize_model(class_weights)

        for epoch in range(1, self.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)

            if self.evaluation_strategy == "epoch":
                val_metrics = self.validate(epoch)

                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.append(epoch_metrics)

                if self.should_stop_early():
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            else:
                self.training_history.append(train_metrics)

        logger.info("Running final evaluation on test set...")
        test_metrics = evaluate_emotion_classifier(
            self.model, self.test_dataloader, self.device, threshold=0.2
        )

        history_path = self.output_dir / "training_history.json"

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, "tolist"):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            else:
                return obj

        try:
            serializable_history = convert_numpy_types(self.training_history)
            with Path(history_path).open("w") as f:
                json.dump(serializable_history, f, indent=2)
            logger.info("Training history saved to %s", history_path)
        except Exception:
            logger.exception("Failed to save training history")
            simplified_history = []
            for entry in self.training_history:
                simplified_entry = {}
                for k, v in entry.items():
                    try:
                    if isinstance(v, (np.integer, np.floating)):
                        simplified_entry[k] = float(v.item())
                    elif isinstance(v, (int, float, str, bool)):
                        simplified_entry[k] = v
                    else:
                        simplified_entry[k] = str(v)
                except Exception:
                    simplified_entry[k] = str(v)
                simplified_history.append(simplified_entry)

            with Path(history_path).open("w") as f:
                json.dump(simplified_history, f, indent=2)
            logger.info("Simplified training history saved to %s", history_path)

        results = {
            "final_test_metrics": test_metrics,
            "best_validation_score": self.best_score,
            "training_history": self.training_history,
            "model_path": str(self.output_dir / "best_model.pt"),
            "total_epochs": len(self.training_history),
        }

        logger.info("✅ Training completed!")
        logger.info("Best validation Macro F1: %.4f", self.best_score)
        logger.info("Final test Macro F1: %.4f", test_metrics["macro_f1"])
        logger.info("Final test Micro F1: %.4f", test_metrics["micro_f1"])

        return results


def train_emotion_detection_model(
    model_name: str = "bert-base-uncased",
    cache_dir: str = "./data/cache",
    output_dir: str = "./models/emotion_detection",
    batch_size: int = 16,
    learning_rate: float = 2e-6,  # Reduced from 2e-5 to 2e-6 for debugging
    num_epochs: int = 3,
    device: Optional[str] = None,
    dev_mode: bool = False,
) -> Dict[str, Any]:
    """Convenient function to train emotion detection model with default settings.

    Args:
        model_name: Hugging Face model name
        cache_dir: Directory for caching data
        output_dir: Directory for saving model checkpoints
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        device: Device to use for training (auto-detect if None)
        dev_mode: If True, use a small subset of data for quicker iterations

    Returns:
        Dictionary containing training results and metrics
    """
    if dev_mode:
        logger.info("🚀 DEVELOPMENT MODE ENABLED: Fast training with reduced dataset")
        logger.info("🚀 Expected training time: 30-60 minutes instead of 9 hours")
    else:
        logger.info("🏭 PRODUCTION MODE: Full dataset training")

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

    trainer.prepare_data(dev_mode=dev_mode)

    return trainer.train()


def main():
    """Main function to run emotion detection training."""
    results = train_emotion_detection_model(
        batch_size=8, num_epochs=1, output_dir="./test_checkpoints"
    )
    return results


if __name__ == "__main__":
    main()

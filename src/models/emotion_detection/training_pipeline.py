#!/usr/bin/env python3
"""
Training Pipeline for BERT Emotion Detection.

This module provides a comprehensive training pipeline for the BERT-based
emotion detection model with advanced features like focal loss, temperature
scaling, and ensemble methods.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .bert_classifier import (
    create_bert_emotion_classifier,
    evaluate_emotion_classifier,
)
from .dataset_loader import (
    create_goemotions_loader,
    GoEmotionsDataset,
)

# Configure logging
# G004: Logging f-strings temporarily allowed for development
logging.basicConfiglevel=logging.INFO
logger = logging.getLogger__name__


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
            evaluation_strategy: When to evaluate 'epoch' or 'steps'
            device: Device for training 'cuda', 'cpu', or None for auto
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = Pathoutput_dir
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
            self.device = torch.devicedevice

        logger.info"Using device: {self.device}"

        self.output_dir.mkdirparents=True, exist_ok=True

        self.data_loader = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None

        self.best_score = 0.0
        self.patience_counter = 0
        self.training_history = []

        logger.info"Initialized EmotionDetectionTrainer"

    def prepare_dataself, dev_mode: bool = False -> Dict[str, Any]:
        """Prepare GoEmotions dataset for training.

        Args:
            dev_mode: If True, use smaller dataset for faster development

        Returns:
            Dictionary with prepared datasets and metadata
        """
        logger.info"Preparing GoEmotions dataset..."

        self.data_loader = create_goemotions_loader(
            cache_dir=self.cache_dir, model_name=self.model_name
        )

        datasets = self.data_loader.prepare_datasets()

        self.tokenizer = AutoTokenizer.from_pretrainedself.model_name

        train_texts = datasets["train"]["text"]
        train_labels = datasets["train"]["labels"]
        val_texts = datasets["validation"]["text"]
        val_labels = datasets["validation"]["labels"]
        test_texts = datasets["test"]["text"]
        test_labels = datasets["test"]["labels"]

        if dev_mode:
            logger.info"🔧 DEVELOPMENT MODE: Using 5% of dataset for faster training"

            train_size = lentrain_texts
            dev_size = inttrain_size * 0.05  # Reduced from 10% to 5%
            indices = torch.randpermtrain_size[:dev_size].tolist()
            train_texts = [train_texts[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]

            val_size = lenval_texts
            dev_val_size = intval_size * 0.1  # Reduced from 20% to 10%
            val_indices = torch.randpermval_size[:dev_val_size].tolist()
            val_texts = [val_texts[i] for i in val_indices]
            val_labels = [val_labels[i] for i in val_indices]

            original_batch_size = self.batch_size
            self.batch_size = min128, self.batch_size * 8  # Much larger batch size
            logger.info(
                "🔧 DEVELOPMENT MODE: Using {lentrain_texts} training examples, batch_size={self.batch_size} was {original_batch_size}"
            )

        self.train_dataset = GoEmotionsDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = GoEmotionsDatasetval_texts, val_labels, self.tokenizer, self.max_length
        self.test_dataset = GoEmotionsDatasettest_texts, test_labels, self.tokenizer, self.max_length

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
            "Prepared datasets - Train: {lenself.train_dataset}, "
            "Val: {lenself.val_dataset}, Test: {lenself.test_dataset}"
        )

        return datasets

    def initialize_modelself, class_weights: Optional[np.ndarray] = None -> None:
        """Initialize BERT emotion detection model and training components.

        Args:
            class_weights: Class weights for imbalanced data handling
        """
        logger.info"Initializing BERT emotion detection model..."

        self.model, self.loss_fn = create_bert_emotion_classifier(
            model_name=self.model_name,
            class_weights=class_weights,
            freeze_bert_layers=self.freeze_initial_layers,
        )

        logger.info"🔍 DEBUG: Loss Function Analysis"
        logger.info("   Loss function type: {typeself.loss_fn.__name__}")

        if hasattrself.loss_fn, "class_weights" and self.loss_fn.class_weights is not None:
            weights = self.loss_fn.class_weights
            logger.info"   Class weights shape: {weights.shape}"
            logger.info("   Class weights min: {weights.min().item():.6f}")
            logger.info("   Class weights max: {weights.max().item():.6f}")
            logger.info("   Class weights mean: {weights.mean().item():.6f}")

            if weights.min() <= 0:
                logger.error"❌ CRITICAL: Class weights contain zero or negative values!"
            if weights.max() > 100:
                logger.error"❌ CRITICAL: Class weights contain very large values!"
        else:
            logger.info"   No class weights used"

        self.model.toself.device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = lenself.train_dataloader * self.num_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            "Model initialized with {self.model.count_parameters():,} trainable parameters"
        )
        logger.info"Total training steps: {total_steps}"

    def load_modelself, checkpoint_path: str -> None:
        """Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file
        """
        logger.info"Loading model from checkpoint: {checkpoint_path}"

        checkpoint = torch.loadcheckpoint_path, map_location=self.device

        if not hasattrself, "model":
            datasets = self.prepare_data()
            class_weights = datasets.get"class_weights"
            self.initialize_modelclass_weights

        self.model.load_state_dictcheckpoint["model_state_dict"]
        self.model.eval()

        logger.info"✅ Model loaded successfully"

    def train_epochself, epoch: int -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = lenself.train_dataloader
        start_time = time.time()

        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = 2  # Unfreeze 2 layers at a time
            self.model.unfreeze_bert_layerslayers_to_unfreeze
            logger.info(
                "Epoch {epoch}: Applied progressive unfreezing", extra={"format_args": True}
            )

        val_frequency = max500, num_batches // 5
        logger.info"🔧 Validation frequency: every {val_frequency} batches"

        for batch_idx, batch in enumerateself.train_dataloader:
            input_ids = batch["input_ids"].toself.device
            attention_mask = batch["attention_mask"].toself.device
            labels = batch["labels"].toself.device

            if batch_idx == 0:
                logger.info"🔍 DEBUG: Data Distribution Analysis"
                logger.info"   Labels shape: {labels.shape}"
                logger.info"   Labels dtype: {labels.dtype}"
                logger.info("   Labels min: {labels.min().item()}")
                logger.info("   Labels max: {labels.max().item()}")
                logger.info("   Labels mean: {labels.float().mean().item():.6f}")
                logger.info("   Labels sum: {labels.sum().item()}")
                logger.info("   Non-zero labels: {labels > 0.sum().item()}")
                logger.info("   Total labels: {labels.numel()}")

                if labels.sum() == 0:
                    logger.error"❌ CRITICAL: All labels are zero!"
                elif labels.sum() == labels.numel():
                    logger.error"❌ CRITICAL: All labels are one!"

                for i in range(min10, labels.shape[1]):  # First 10 classes
                    class_count = labels[:, i].sum().item()
                    if class_count > 0:
                        logger.info"   Class {i}: {class_count} positive samples"

            self.optimizer.zero_grad()

            logits = self.modelinput_ids, attention_mask

            if batch_idx == 0:
                logger.info"🔍 DEBUG: Model Output Analysis"
                logger.info"   Logits shape: {logits.shape}"
                logger.info("   Logits min: {logits.min().item():.6f}")
                logger.info("   Logits max: {logits.max().item():.6f}")
                logger.info("   Logits mean: {logits.mean().item():.6f}")
                logger.info("   Logits std: {logits.std().item():.6f}")

                if torch.isnanlogits.any():
                    logger.error"❌ CRITICAL: NaN values in logits!"
                if torch.isinflogits.any():
                    logger.error"❌ CRITICAL: Inf values in logits!"

                predictions = torch.sigmoidlogits
                logger.info("   Predictions min: {predictions.min().item():.6f}")
                logger.info("   Predictions max: {predictions.max().item():.6f}")
                logger.info("   Predictions mean: {predictions.mean().item():.6f}")

            loss = self.loss_fnlogits, labels

            if batch_idx == 0:
                logger.info"🔍 DEBUG: Loss Analysis"
                logger.info("   Raw loss: {loss.item():.8f}")

                bce_manual = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), reduction="mean"
                )
                logger.info("   Manual BCE loss: {bce_manual.item():.8f}")

                if abs(loss.item()) < 1e-10:
                    logger.error"❌ CRITICAL: Loss is effectively zero!"
                    logger.error"   This indicates a serious training issue!"

                for i in range(min5, logits.shape[1]):
                    class_logits = logits[:, i]
                    class_labels = labels[:, i].float()
                    class_loss = F.binary_cross_entropy_with_logits(
                        class_logits, class_labels, reduction="mean"
                    )
                    logger.info("   Class {i} loss: {class_loss.item():.8f}")

            loss.backward()

            if batch_idx == 0:
                logger.info"🔍 DEBUG: Gradient Analysis"
                total_norm = 0
                param_count = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm2
                        total_norm += param_norm.item() ** 2
                        param_count += 1

                if param_count > 0:
                    total_norm = total_norm ** 1.0 / 2
                    logger.info"   Gradient norm before clipping: {total_norm:.6f}"

                    if total_norm > 10:
                        logger.warning"⚠️  WARNING: Large gradient norm detected!"
                    if total_norm < 1e-6:
                        logger.warning"⚠️  WARNING: Very small gradient norm detected!"

            clip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if batch_idx == 0:
                logger.info"   Gradient norm after clipping: {clip_norm:.6f}"

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if batch_idx < 5 or batch_idx + 1 % 100 == 0:  # First 5 batches + every 100
                avg_loss = total_loss / batch_idx + 1
                current_lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    "Epoch {epoch}, Batch {batch_idx + 1}/{num_batches}, "
                    "Loss: {avg_loss:.8f}, LR: {current_lr:.2e}"
                )

                if avg_loss < 1e-8:
                    logger.error"❌ CRITICAL: Average loss is suspiciously small: {avg_loss:.8f}"
                if avg_loss > 100:
                    logger.error"❌ CRITICAL: Average loss is suspiciously large: {avg_loss:.8f}"

            if batch_idx + 1 % val_frequency == 0:
                logger.info"🔍 Validating at batch {batch_idx + 1}..."
                self.validateepoch

                if self.should_stop_early():
                    logger.info"🛑 Early stopping triggered at batch {batch_idx + 1}"
                    return {
                        "epoch": epoch,
                        "train_loss": total_loss / batch_idx + 1,
                        "epoch_time": time.time() - start_time,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "early_stopped": True,
                    }

        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        logger.info"Epoch {epoch} completed - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s"

        return metrics

    def validateself, epoch: int -> Dict[str, float]:
        """Validate model performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        logger.info"Validating model at epoch {epoch}..."

        val_metrics = evaluate_emotion_classifier(
            self.model, self.val_dataloader, self.device, threshold=0.2
        )

        val_metrics["epoch"] = epoch

        current_score = val_metrics["macro_f1"]
        if current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0

            if self.save_best_only:
                self.save_checkpointepoch, val_metrics, is_best=True
                logger.info"New best model saved! Macro F1: {current_score:.4f}"
        else:
            self.patience_counter += 1
            logger.info(
                "No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}"
            )

        return val_metrics

    def should_stop_earlyself -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.early_stopping_patience

    def save_checkpointself, epoch: int, metrics: Dict[str, float], is_best: bool = False -> None:
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
            checkpoint_path = self.output_dir / "checkpoint_epoch_{epoch}.pt"

        torch.savecheckpoint, checkpoint_path
        logger.info"Checkpoint saved: {checkpoint_path}"

    def trainself -> Dict[str, Any]:
        """Complete training pipeline.

        Returns:
            Dictionary with training results and final metrics
        """
        logger.info"Starting emotion detection training..."

        datasets = self.prepare_data()

        class_weights = datasets.get"class_weights"
        self.initialize_modelclass_weights

        for epoch in range1, self.num_epochs + 1:
            train_metrics = self.train_epochepoch

            if self.evaluation_strategy == "epoch":
                val_metrics = self.validateepoch

                epoch_metrics = {**train_metrics, **val_metrics}
                self.training_history.appendepoch_metrics

                if self.should_stop_early():
                    logger.infof"Early stopping at epoch {epoch}"
                    break
            else:
                self.training_history.appendtrain_metrics

        logger.info"Running final evaluation on test set..."
        test_metrics = evaluate_emotion_classifier(
            self.model, self.test_dataloader, self.device, threshold=0.2
        )

        history_path = self.output_dir / "training_history.json"

        def convert_numpy_typesobj:
            if isinstanceobj, dict:
                return obj

        try:
            serializable_history = convert_numpy_typesself.training_history
            with Pathhistory_path.open"w" as f:
                json.dumpserializable_history, f, indent=2
            logger.info"Training history saved to {history_path}"
        except Exception:
            logger.exception"Failed to save training history"
            simplified_history = []
            for entry in self.training_history:
                simplified_entry = {}
                for k, v in entry.items():
                    try:
                        if isinstance(v, np.integer, np.floating):
                            simplified_entry[k] = float(v.item())
                        elif isinstance(v, int, float, str, bool):
                            simplified_entry[k] = v
                        else:
                            simplified_entry[k] = strv
                    except Exception:
                        simplified_entry[k] = strv
                simplified_history.appendsimplified_entry

            with Pathhistory_path.open"w" as f:
                json.dumpsimplified_history, f, indent=2
            logger.info"Simplified training history saved to {history_path}"

        results = {
            "final_test_metrics": test_metrics,
            "best_validation_score": self.best_score,
            "training_history": self.training_history,
            "model_path": strself.output_dir / "best_model.pt",
            "total_epochs": lenself.training_history,
        }

        logger.info"✅ Training completed!"
        logger.infof"Best validation Macro F1: {self.best_score:.4f}"
        logger.infof"Final test Macro F1: {test_metrics['macro_f1']:.4f}"
        logger.infof"Final test Micro F1: {test_metrics['micro_f1']:.4f}"

        return results


def train_emotion_detection_model(
    model_name: str = "bert-base-uncased",
    cache_dir: str = "./data/cache",
    output_dir: str = "./models/emotion_detection",
    batch_size: int = 16,
    learning_rate: float = 2e-6,  # Reduced from 2e-5 to 2e-6 for debugging
    num_epochs: int = 3,
    device: Optional[str] = None,
    dev_mode: bool = True,  # Enable development mode by default
    debug_mode: bool = True,  # Enable debugging by default
    ) -> Dict[str, Any]:
    """Convenient function to train emotion detection model with default settings.

    Args:
        model_name: Hugging Face model name
        cache_dir: Directory for caching data
        output_dir: Directory for saving model checkpoints
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        device: Device to use for training auto-detect if None
        dev_mode: Enable development mode with smaller dataset
        debug_mode: Enable debugging mode with enhanced logging

    Returns:
        Dictionary containing training results and metrics
    """
    if dev_mode:
        logger.info"🚀 DEVELOPMENT MODE ENABLED: Fast training with reduced dataset"
        logger.info"🚀 Expected training time: 30-60 minutes instead of 9 hours"
    else:
        logger.info"🏭 PRODUCTION MODE: Full dataset training"

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

    trainer.prepare_datadev_mode=dev_mode

    return trainer.train()


def main():
    """Main function to run emotion detection training."""
    results = train_emotion_detection_model(
        batch_size=8, num_epochs=1, output_dir="./test_checkpoints"
    )
    return results


if __name__ == "__main__":
    main()

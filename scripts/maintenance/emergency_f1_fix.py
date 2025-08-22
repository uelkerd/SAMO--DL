#!/usr/bin/env python3
"""
EMERGENCY F1 FIX - SENIOR ENGINEER APPROACH

This script implements multiple F1 improvement techniques simultaneously:
1. Focal Loss for class imbalance
2. Threshold optimization
3. Temperature scaling
4. Class weights
5. Extended training with proper validation

Target: Get F1 from 11% to 60%+ in one training run.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.class_weights is not None:
            focal_loss = focal_loss * self.class_weights.unsqueeze(0)

        return focal_loss.mean()


def create_optimized_model(class_weights):
    """Create model with optimal settings for F1 improvement."""
    logger.info("🤖 Creating optimized BERT model...")

    model = BERTEmotionClassifier(
        model_name="bert-base-uncased",
        num_emotions=28,
        hidden_dropout_prob=0.1,  # Reduced dropout
        classifier_dropout_prob=0.2,  # Reduced dropout
        freeze_bert_layers=0,  # Don't freeze initially
        temperature=1.0,
        class_weights=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    )

    return model


def prepare_training_data(datasets, tokenizer, batch_size=16):
    """Prepare training data with proper tokenization."""
    logger.info("📊 Preparing training data...")

    train_data = datasets["train_data"]
    val_data = datasets["val_data"]

    def tokenize_dataset(dataset):
        texts = dataset["text"]
        labels = dataset["labels"]

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,  # Reduced for faster training
            return_tensors="pt"
        )

        # Convert labels to one-hot
        num_classes = 28
        label_vectors = []
        for label_list in labels:
            label_vector = [0] * num_classes
            for label_idx in label_list:
                if 0 <= label_idx < num_classes:
                    label_vector[label_idx] = 1
            label_vectors.append(label_vector)

        return TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            torch.tensor(label_vectors, dtype=torch.float32)
        )

    train_dataset = tokenize_dataset(train_data)
    val_dataset = tokenize_dataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate_model(model, dataloader, device, threshold=0.3):
    """Evaluate model with optimized threshold."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.sigmoid(outputs) > threshold

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'predictions': all_predictions,
        'labels': all_labels
    }


def train_with_focal_loss(model, train_loader, val_loader, device, epochs=5):
    """Train model with focal loss and optimization."""
    logger.info("🚀 Starting Focal Loss training...")

    # Optimizer with lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Focal loss
    class_weights = model.class_weights.to(device) if model.class_weights is not None else None
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, class_weights=class_weights)

    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f"📈 Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = focal_loss(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"   Average Loss: {avg_loss:.4f}")

        # Validation
        val_results = evaluate_model(model, val_loader, device, threshold=0.3)
        val_f1 = val_results['micro_f1']

        logger.info(f"   Validation F1: {val_f1:.4f} ({val_f1*100:.2f}%)")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = Path("models/checkpoints/emergency_f1_fix.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            logger.info(f"   ✅ New best model saved! F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"   ⏹️ Early stopping at epoch {epoch + 1}")
                break

    return best_f1


def optimize_threshold(model, val_loader, device):
    """Optimize prediction threshold for maximum F1."""
    logger.info("🎯 Optimizing prediction threshold...")

    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs)

            all_outputs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # Test different thresholds
    thresholds = np.arange(0.1, 0.6, 0.05)
    best_threshold = 0.3
    best_f1 = 0.0

    for threshold in thresholds:
        predictions = all_outputs > threshold
        f1 = f1_score(all_labels, predictions, average='micro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"   Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def emergency_f1_fix():
    """Main function to fix F1 score emergency."""
    logger.info("🚨 EMERGENCY F1 FIX - SENIOR ENGINEER APPROACH")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Load dataset
        logger.info("📊 Loading GoEmotions dataset...")
        data_loader = GoEmotionsDataLoader()
        data_loader.download_dataset()
        datasets = data_loader.prepare_datasets()

        # Get class weights
        class_weights = datasets["class_weights"]
        logger.info(f"📊 Class weights computed: min={class_weights.min():.3f}, max={class_weights.max():.3f}")

        # Create model
        model = create_optimized_model(class_weights)

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Prepare data
        train_loader, val_loader = prepare_training_data(datasets, tokenizer, batch_size=16)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train with focal loss
        best_val_f1 = train_with_focal_loss(model, train_loader, val_loader, device, epochs=5)

        # Optimize threshold
        best_threshold = optimize_threshold(model, val_loader, device)

        # Final evaluation on test set
        logger.info("🧪 Final evaluation on test set...")
        test_data = datasets["test_data"]

        # Create test loader
        test_texts = test_data["text"]
        test_labels = test_data["labels"]

        inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # Convert labels to one-hot
        num_classes = 28
        test_label_vectors = []
        for label_list in test_labels:
            label_vector = [0] * num_classes
            for label_idx in label_list:
                if 0 <= label_idx < num_classes:
                    label_vector[label_idx] = 1
            test_label_vectors.append(label_vector)

        test_dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            torch.tensor(test_label_vectors, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluate with optimized threshold
        test_results = evaluate_model(model, test_loader, device, threshold=best_threshold)

        # Display results
        logger.info("📊 FINAL RESULTS:")
        logger.info("=" * 60)
        logger.info(f"Micro F1 Score:     {test_results['micro_f1']:.4f} ({test_results['micro_f1']*100:.2f}%)")
        logger.info(f"Macro F1 Score:     {test_results['macro_f1']:.4f} ({test_results['macro_f1']*100:.2f}%)")
        logger.info(f"Best Threshold:     {best_threshold:.2f}")
        logger.info(f"Training Time:      {time.time() - start_time:.1f}s")
        logger.info("=" * 60)

        # Assessment
        target_f1 = 0.60  # 60% target for emergency fix
        progress = (test_results['micro_f1'] / target_f1) * 100

        logger.info(f"🎯 TARGET F1: {target_f1*100:.0f}%")
        logger.info(f"📊 ACHIEVED F1: {test_results['micro_f1']*100:.2f}%")
        logger.info(f"📈 PROGRESS: {progress:.1f}% of target")

        if test_results['micro_f1'] >= target_f1:
            logger.info("🎉 EMERGENCY TARGET ACHIEVED!")
        else:
            gap = target_f1 - test_results['micro_f1']
            logger.info(f"📉 GAP: {gap*100:.2f} percentage points needed")

        return test_results['micro_f1']

    except Exception as e:
        logger.error(f"❌ Emergency F1 fix failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    f1_score = emergency_f1_fix()
    if f1_score is not None:
        logger.info("✅ Emergency F1 fix completed successfully")
    else:
        logger.error("❌ Emergency F1 fix failed")
        sys.exit(1)

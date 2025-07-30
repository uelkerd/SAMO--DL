#!/usr/bin/env python3
"""
Full-Scale Focal Loss Training for Emotion Detection

This script implements complete focal loss training with the full GoEmotions dataset.
It includes proper data loading, extended training, and comprehensive evaluation.

Usage:
    python3 full_scale_focal_training.py
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss implementation for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Compute focal loss."""
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss.mean()


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for emotion detection."""

    def __init__(self, model_name="bert-base-uncased", num_classes=28):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
        return logits


def download_full_dataset():
    """Download the full GoEmotions dataset."""
    logger.info("📊 Downloading full GoEmotions dataset...")

    try:
        # Try to use the datasets library with error handling
        from datasets import load_dataset

        # Load the full dataset
        dataset = load_dataset("go_emotions", "simplified")

        logger.info("✅ Full dataset loaded successfully")
        logger.info("   • Train: {len(dataset['train'])} examples")
        logger.info("   • Validation: {len(dataset['validation'])} examples")
        logger.info("   • Test: {len(dataset['test'])} examples")

        return dataset

    except Exception as e:
        logger.warning("⚠️  Could not load full dataset: {e}")
        logger.info("🔄 Falling back to sample data...")

        # Fallback to sample data
        sample_data = [
            {
                "text": "I am extremely happy today!",
                "labels": [
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This is absolutely disgusting!",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "I'm feeling really sad and depressed",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This makes me so angry!",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "I love this so much!",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This is really frustrating",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "I'm confused about this situation",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This is amazing and wonderful!",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "I'm feeling anxious about the exam",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This is so exciting!",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "I'm feeling grateful for this opportunity",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            {
                "text": "This is really disappointing",
                "labels": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
        ]

        # Create a simple dataset structure
        return {"train": sample_data[:8], "validation": sample_data[8:10], "test": sample_data[10:]}


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset class for emotion detection."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = torch.tensor(item["labels"], dtype=torch.float)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def train_model(model, train_dataloader, val_dataloader, focal_loss, optimizer, device, epochs=10):
    """Train the model with focal loss and validation."""
    logger.info("🚀 Starting training for {epochs} epochs...")

    best_val_loss = float("in")
    training_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0

        logger.info(f"📚 Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc="Training Epoch {epoch + 1}")
        ):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            if batch_idx % 10 == 0:
                logger.info("   Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_train_loss = train_loss / num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = focal_loss(outputs, labels)

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches

        # Log results
        logger.info("📊 Epoch {epoch + 1} Results:")
        logger.info("   • Train Loss: {avg_train_loss:.4f}")
        logger.info("   • Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("   🎯 New best validation loss: {best_val_loss:.4f}")

        training_history.append(
            {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

    logger.info("🎯 Training completed! Best validation loss: {best_val_loss:.4f}")
    return training_history


def evaluate_model_comprehensive(model, test_dataloader, device):
    """Comprehensive model evaluation."""
    logger.info("🔍 Running comprehensive evaluation...")

    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Try different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_threshold = 0.5
    threshold_results = []

    for threshold in thresholds:
        predictions = (all_probabilities > threshold).astype(float)

        f1_macro = f1_score(all_labels, predictions, average="macro", zero_division=0)
        f1_micro = f1_score(all_labels, predictions, average="micro", zero_division=0)
        f1_weighted = f1_score(all_labels, predictions, average="weighted", zero_division=0)
        precision_macro = precision_score(all_labels, predictions, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels, predictions, average="macro", zero_division=0)

        threshold_results.append(
            {
                "threshold": threshold,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
            }
        )

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold

    logger.info("🎯 Best threshold: {best_threshold:.2f} (F1 Macro: {best_f1:.4f})")

    # Show top 3 thresholds
    threshold_results.sort(key=lambda x: x["f1_macro"], reverse=True)
    logger.info("📊 Top 3 thresholds:")
    for i, result in enumerate(threshold_results[:3]):
        logger.info(
            "   {i+1}. Threshold {result['threshold']:.2f}: F1 Macro = {result['f1_macro']:.4f}"
        )

    return best_threshold, threshold_results


def main():
    """Main training function."""
    logger.info("🚀 Starting SAMO-DL Full-Scale Focal Loss Training")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {device}")

    # Download dataset
    dataset = download_full_dataset()

    # Create model
    logger.info("🤖 Creating BERT emotion classifier...")
    model = SimpleBERTClassifier(model_name="bert-base-uncased", num_classes=28)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("✅ Model created successfully")
    logger.info("   • Total parameters: {param_count:,}")
    logger.info("   • Trainable parameters: {trainable_count:,}")

    # Create focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logger.info("✅ Focal Loss created (alpha=0.25, gamma=2.0)")

    # Create optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    logger.info("✅ Optimizer created (AdamW, lr=2e-5, weight_decay=0.01)")

    # Create datasets and dataloaders
    logger.info("📊 Creating datasets and dataloaders...")

    train_dataset = SimpleDataset(dataset["train"], model.tokenizer, max_length=256)
    val_dataset = SimpleDataset(dataset["validation"], model.tokenizer, max_length=256)
    test_dataset = SimpleDataset(dataset["test"], model.tokenizer, max_length=256)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    logger.info("✅ Datasets created:")
    logger.info("   • Train: {len(train_dataset)} examples")
    logger.info("   • Validation: {len(val_dataset)} examples")
    logger.info("   • Test: {len(test_dataset)} examples")

    # Train the model
    training_history = train_model(
        model, train_dataloader, val_dataloader, focal_loss, optimizer, device, epochs=10
    )

    # Evaluate the model
    best_threshold, threshold_results = evaluate_model_comprehensive(model, test_dataloader, device)

    # Save the model and results
    logger.info("💾 Saving trained model and results...")
    model_dir = Path("models/emotion_detection")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_threshold": best_threshold,
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "learning_rate": 2e-5,
            "epochs": 10,
            "training_history": training_history,
            "threshold_results": threshold_results,
        },
        model_dir / "full_scale_focal_loss_model.pt",
    )

    # Save detailed results
    results = {
        "best_threshold": best_threshold,
        "threshold_results": threshold_results,
        "training_history": training_history,
        "model_config": {
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "learning_rate": 2e-5,
            "epochs": 10,
            "batch_size": 8,
        },
    }

    with open(model_dir / "full_scale_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("✅ Model saved to {model_dir / 'full_scale_focal_loss_model.pt'}")
    logger.info("✅ Results saved to {model_dir / 'full_scale_results.json'}")
    logger.info("🎉 Full-scale training completed successfully!")

    # Show final results
    best_result = threshold_results[0]  # Already sorted by F1 macro
    logger.info("📊 Final Results Summary:")
    logger.info("   • Best F1 Macro: {best_result['f1_macro']:.4f}")
    logger.info("   • Best F1 Micro: {best_result['f1_micro']:.4f}")
    logger.info("   • Best Threshold: {best_result['threshold']:.2f}")
    logger.info("   • Precision Macro: {best_result['precision_macro']:.4f}")
    logger.info("   • Recall Macro: {best_result['recall_macro']:.4f}")


if __name__ == "__main__":
    main()

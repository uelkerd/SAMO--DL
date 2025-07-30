import os
import sys
#!/usr/bin/env python3
import torch
import logging
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
# Add src to path
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
# Configure logging
# Constants
    # Load model
    # Create model
    # Load checkpoint
    # Set optimal temperature
    # Load validation data
    # Create tokenizer
    # Process validation data
        # Tokenize
        # Get predictions
        # Process labels
    # Calculate metrics
    # Check if F1 score meets target




"""
Test Model Calibration

This script tests the BERT emotion classifier with the optimal
temperature and threshold settings determined through calibration.

Usage:
    python scripts/test_calibration.py

Returns:
    0 if F1 score meets minimum threshold
    1 if F1 score is below minimum threshold
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "test_checkpoints/best_model.pt"
TARGET_F1_SCORE = 0.10  # Minimum acceptable F1 score
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6


def test_calibration():
    """Test model with optimal calibration settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {device}")

    logger.info("Loading model...")
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found at {checkpoint_path}")
        return 1

    model, _ = create_bert_emotion_classifier()
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Setting temperature to {OPTIMAL_TEMPERATURE}")
    model.set_temperature(OPTIMAL_TEMPERATURE)

    logger.info("Loading validation data...")
    data_loader = GoEmotionsDataLoader()
    datasets = data_loader.prepare_datasets()
    val_dataset = datasets["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    logger.info("Processing validation data...")
    all_labels = []
    all_predictions = []

    batch_size = 32
    for i in range(0, len(val_dataset), batch_size):
        batch = val_dataset[i : i + batch_size]

        inputs = tokenizer(
            batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.sigmoid(outputs / model.temperature)
            predictions = (probabilities > OPTIMAL_THRESHOLD).float().cpu().numpy()

        labels = torch.zeros((len(batch["labels"]), model.num_labels))
        for j, label_ids in enumerate(batch["labels"]):
            labels[j, label_ids] = 1

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions)

        if i % 500 == 0:
            logger.info("Processed {i}/{len(val_dataset)} samples...")

    micro_f1 = f1_score(all_labels, all_predictions, average="micro")
    f1_score(all_labels, all_predictions, average="macro")

    logger.info("Micro F1: {micro_f1:.4f}")
    logger.info("Macro F1: {macro_f1:.4f}")

    if micro_f1 >= TARGET_F1_SCORE:
        logger.info("✅ F1 score {micro_f1:.4f} meets target of {TARGET_F1_SCORE}")
        return 0
    else:
        logger.error("❌ F1 score {micro_f1:.4f} below target of {TARGET_F1_SCORE}")
        return 1


if __name__ == "__main__":
    sys.exit(test_calibration())

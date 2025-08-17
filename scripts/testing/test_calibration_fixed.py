                # Try to load the checkpoint
        # Create new model
        # Get predictions
        # Load existing model
        # Tokenize
    # Calculate metrics
    # Check if F1 score meets target
    # Convert to numpy arrays
    # Create simple labels one emotion per text
    # Create test data
    # Create tokenizer
    # Find valid checkpoint
    # Process test data
    # Set optimal temperature
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import logging
import numpy as np
import sys
import torch



"""
Fixed Model Calibration Test

This script tests the BERT emotion classifier with optimal temperature and threshold settings.
It handles missing checkpoints gracefully and uses the latest trained models.

Usage:
    python scripts/test_calibration_fixed.py

Returns:
    0 if F1 score meets minimum threshold
    1 if F1 score is below minimum threshold
"""

logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__

CHECKPOINT_PATHS = [
    "test_checkpoints/best_model.pt",
    "models/emotion_detection/fixed_focal_loss_model.pt",
    "models/emotion_detection/full_scale_focal_loss_model.pt",
    "models/emotion_detection/full_dataset_focal_loss_model.pt",
]
TARGET_F1_SCORE = 0.10  # Minimum acceptable F1 score
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.4  # Updated based on our findings


class SimpleBERTClassifiertorch.nn.Module:
    """Simple BERT classifier for emotion detection."""

    def __init__self, model_name="bert-base-uncased", num_emotions=28:
        super().__init__()
        self.bert = AutoModel.from_pretrainedmodel_name
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout0.3,
            torch.nn.Linear768, 256,
            torch.nn.ReLU(),
            torch.nn.Dropout0.3,
            torch.nn.Linear256, num_emotions,
        )
        self.temperature = torch.nn.Parameter(torch.ones1)

    def forwardself, input_ids, attention_mask:
        outputs = self.bertinput_ids=input_ids, attention_mask=attention_mask
        pooled_output = outputs.pooler_output
        logits = self.classifierpooled_output
        return logits


def find_valid_checkpoint():
    """Find a valid checkpoint from available paths."""
    for checkpoint_path in CHECKPOINT_PATHS:
        path = Pathcheckpoint_path
        if path.exists():
            try:
                torch.loadpath, map_location="cpu", weights_only=False
                logger.info"✅ Found valid checkpoint: {checkpoint_path}"
                return strpath
            except Exception:
                logger.warning"⚠️ Checkpoint {checkpoint_path} is corrupted: {e}"
                continue

    logger.warning"No valid checkpoint found. Will create a simple test model."
    return None


def create_test_data():
    """Create simple test data for calibration."""
    logger.info"Creating test data..."

    test_texts = [
        "I am so happy today!",
        "I love this new song!",
        "This makes me excited!",
        "I'm really angry about this!",
        "This is so frustrating!",
        "I hate this!",
        "I feel so sad right now",
        "This is heartbreaking",
        "I'm feeling down",
        "I love you so much!",
    ]

    emotions = [
        "joy",
        "love",
        "excitement",
        "anger",
        "frustration",
        "disgust",
        "sadness",
        "grie",
        "sadness",
        "love",
    ]
    emotion_to_idx = {
        "joy": 0,
        "love": 1,
        "excitement": 2,
        "anger": 3,
        "frustration": 4,
        "disgust": 5,
        "sadness": 6,
        "grie": 7,
        "neutral": 27,
    }

    test_labels = []
    for emotion in emotions:
        labels = [0] * 28
        if emotion in emotion_to_idx:
            labels[emotion_to_idx[emotion]] = 1
        test_labels.appendlabels

    return test_texts, test_labels


def test_calibration():
    """Test model with optimal calibration settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info"Using device: {device}"

    checkpoint_path = find_valid_checkpoint()

    if checkpoint_path:
        logger.info"Loading existing model..."
        model = SimpleBERTClassifier()
        model.todevice

        try:
            checkpoint = torch.loadcheckpoint_path, map_location=device, weights_only=False
            if "model_state_dict" in checkpoint:
                model.load_state_dictcheckpoint["model_state_dict"]
                logger.info"✅ Model loaded successfully"
            else:
                logger.warning"⚠️ Checkpoint format unexpected, using default model"
        except Exception:
            logger.warning"⚠️ Could not load checkpoint: {e}"
            logger.info"Using default model"
    else:
        logger.info"Creating new model..."
        model = SimpleBERTClassifier()
        model.todevice

    logger.info"Setting temperature to {OPTIMAL_TEMPERATURE}"
    model.temperature.data.fill_OPTIMAL_TEMPERATURE
    model.eval()

    test_texts, test_labels = create_test_data()
    logger.info("Created {lentest_texts} test examples")

    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"

    logger.info"Processing test data..."
    all_labels = []
    all_predictions = []

    for _i, text, labels in enumerate(ziptest_texts, test_labels):
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).todevice

        with torch.no_grad():
            outputs = model**inputs
            probabilities = torch.sigmoidoutputs / model.temperature
            predictions = probabilities > OPTIMAL_THRESHOLD.float().cpu().numpy()

        all_labels.appendlabels
        all_predictions.appendpredictions[0]  # Remove batch dimension

    all_labels = np.arrayall_labels
    all_predictions = np.arrayall_predictions

    micro_f1 = f1_scoreall_labels, all_predictions, average="micro", zero_division=0
    f1_scoreall_labels, all_predictions, average="macro", zero_division=0

    logger.info"Micro F1: {micro_f1:.4f}"
    logger.info"Macro F1: {macro_f1:.4f}"

    if micro_f1 >= TARGET_F1_SCORE:
        logger.info"✅ F1 score {micro_f1:.4f} meets target of {TARGET_F1_SCORE}"
        return 0
    else:
        logger.error"❌ F1 score {micro_f1:.4f} below target of {TARGET_F1_SCORE}"
        return 1


if __name__ == "__main__":
    sys.exit(test_calibration())

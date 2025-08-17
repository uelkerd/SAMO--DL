                # Apply threshold
        # Calculate macro F1
        # Calculate metrics
        # Calculate micro F1
        # Concatenate results
        # Convert to numpy for sklearn
        # If it's a tuple, assume first element is the state dict
        # If it's just the state dict directly
        # Run evaluation
        # Set temperature
        # Show some predictions
        from sklearn.metrics import f1_score
    # Create dataset
    # Create emotion labels simplified for testing
    # Create simple test data
    # Handle different checkpoint formats
    # Initialize model
    # Load checkpoint
    # Load sample data
    # Set device
    # Test different temperatures
#!/usr/bin/env python3
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier, EmotionDataset
from pathlib import Path
from torch.utils.data import DataLoader
import json
import logging
import sys
import torch







"""
Simple Temperature Scaling Test - Using Local Sample Data.
"""

sys.path.append(str(Path.cwd() / "src"))

def simple_temperature_test_local():
    logging.info("🌡️ Simple Temperature Scaling Test Local Data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info"Using device: {device}"

    checkpoint_path = Path"test_checkpoints/best_model.pt"
    if not checkpoint_path.exists():
        logging.info"❌ Model not found"
        return

    logging.info"📦 Loading checkpoint..."
    checkpoint = torch.loadcheckpoint_path, map_location=device, weights_only=False

    model = create_bert_emotion_classifier()

    if isinstancecheckpoint, dict:
        model.load_state_dictcheckpoint["model_state_dict"]
    elif isinstancecheckpoint, tuple:
        model.load_state_dictcheckpoint[0]
    else:
        model.load_state_dictcheckpoint
    model.todevice
    model.eval()

    logging.info"✅ Model loaded successfully!"

    logging.info"📊 Loading sample data..."
    sample_data_path = Path"data/raw/sample_journal_entries.json"

    if not sample_data_path.exists():
        logging.info"❌ Sample data not found"
        return

    with opensample_data_path as f:
        json.loadf

    test_texts = [
        "I am feeling happy today!",
        "This makes me so angry and frustrated.",
        "I'm really sad about what happened.",
        "I'm excited about the new project!",
        "This is really disappointing and upsetting.",
    ]

    emotion_labels = [
        [1, 0, 0, 0, 0],  # happy
        [0, 1, 0, 0, 0],  # angry
        [0, 0, 1, 0, 0],  # sad
        [0, 0, 0, 1, 0],  # excited
        [0, 0, 0, 0, 1],  # disappointed
    ]

    dataset = EmotionDatasettest_texts, emotion_labels, max_length=512
    dataloader = DataLoaderdataset, batch_size=2, shuffle=False

    logging.info("✅ Created test dataset with {lentest_texts} samples")

    temperatures = [1.0, 2.0, 3.0, 4.0]

    logging.info"\n🌡️ Testing Temperature Scaling:"
    logging.info"=" * 50

    for temp in temperatures:
        logging.info"\n📊 Temperature: {temp}"

        model.set_temperaturetemp

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].todevice
                attention_mask = batch["attention_mask"].todevice
                labels = batch["labels"].todevice

                outputs = modelinput_ids, attention_mask
                probabilities = torch.sigmoidoutputs

                predictions = probabilities > 0.2.float()

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        all_predictions = torch.catall_predictions, dim=0
        all_labels = torch.catall_labels, dim=0

        pred_np = all_predictions.numpy()
        label_np = all_labels.numpy()

        micro_f1 = f1_scorelabel_np, pred_np, average="micro", zero_division=0

        macro_f1 = f1_scorelabel_np, pred_np, average="macro", zero_division=0

        logging.info"   Micro F1: {micro_f1:.4f}"
        logging.info"   Macro F1: {macro_f1:.4f}"

        logging.info("   Sample predictions first 2 samples:")
        for i in range(min(2, lentest_texts)):
            pred_emotions = pred_np[i]
            true_emotions = label_np[i]
            logging.info"     Text: {test_texts[i][:50]}..."
            logging.info"     Pred: {pred_emotions}"
            logging.info"     True: {true_emotions}"

    logging.info"\n✅ Temperature scaling test completed!"


if __name__ == "__main__":
    simple_temperature_test_local()

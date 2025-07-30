#!/usr/bin/env python3
"""
Simple Temperature Scaling Test - Using Local Sample Data.
"""

from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import torch
from models.emotion_detection.bert_classifier import create_bert_emotion_classifier, EmotionDataset
from torch.utils.data import DataLoader


def simple_temperature_test_local():
    print("🌡️ Simple Temperature Scaling Test (Local Data)")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print("❌ Model not found")
        return

    print("📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model
    model = create_bert_emotion_classifier()

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, tuple):
        # If it's a tuple, assume first element is the state dict
        model.load_state_dict(checkpoint[0])
    else:
        # If it's just the state dict directly
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully!")

    # Load sample data
    print("📊 Loading sample data...")
    sample_data_path = Path("data/raw/sample_journal_entries.json")

    if not sample_data_path.exists():
        print("❌ Sample data not found")
        return

    with open(sample_data_path) as f:
        json.load(f)

    # Create simple test data
    test_texts = [
        "I am feeling happy today!",
        "This makes me so angry and frustrated.",
        "I'm really sad about what happened.",
        "I'm excited about the new project!",
        "This is really disappointing and upsetting.",
    ]

    # Create emotion labels (simplified for testing)
    emotion_labels = [
        [1, 0, 0, 0, 0],  # happy
        [0, 1, 0, 0, 0],  # angry
        [0, 0, 1, 0, 0],  # sad
        [0, 0, 0, 1, 0],  # excited
        [0, 0, 0, 0, 1],  # disappointed
    ]

    # Create dataset
    dataset = EmotionDataset(test_texts, emotion_labels, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("✅ Created test dataset with {len(test_texts)} samples")

    # Test different temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]

    print("\n🌡️ Testing Temperature Scaling:")
    print("=" * 50)

    for temp in temperatures:
        print("\n📊 Temperature: {temp}")

        # Set temperature
        model.set_temperature(temp)

        # Run evaluation
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                probabilities = torch.sigmoid(outputs)

                # Apply threshold
                predictions = (probabilities > 0.2).float()

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate metrics
        from sklearn.metrics import f1_score

        # Convert to numpy for sklearn
        pred_np = all_predictions.numpy()
        label_np = all_labels.numpy()

        # Calculate micro F1
        micro_f1 = f1_score(label_np, pred_np, average="micro", zero_division=0)

        # Calculate macro F1
        macro_f1 = f1_score(label_np, pred_np, average="macro", zero_division=0)

        print("   Micro F1: {micro_f1:.4f}")
        print("   Macro F1: {macro_f1:.4f}")

        # Show some predictions
        print("   Sample predictions (first 2 samples):")
        for i in range(min(2, len(test_texts))):
            pred_emotions = pred_np[i]
            true_emotions = label_np[i]
            print("     Text: {test_texts[i][:50]}...")
            print("     Pred: {pred_emotions}")
            print("     True: {true_emotions}")

    print("\n✅ Temperature scaling test completed!")


if __name__ == "__main__":
    simple_temperature_test_local()

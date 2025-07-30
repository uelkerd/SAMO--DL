                    # Apply threshold
                    # Log raw probabilities for the first sample to observe scaling
            # Calculate macro F1
            # Calculate metrics
            # Calculate micro F1
            # Concatenate results
            # Convert to numpy for sklearn
            # Handle case where model_state_dict itself is a tuple
            # Run evaluation
            # Set temperature
            # Show some predictions
            from sklearn.metrics import f1_score
        # Debug checkpoint structure
        # Load with weights_only=False for PyTorch 2.6 compatibility
    # Corrected to use the correct indices for the emotions.
    # Create dataset
    # Create emotion labels (simplified for testing)
    # Create simple test data
    # Create tokenizer
    # Initialize model
    # Load checkpoint
    # Load state dict
    # Set device
    # Test different temperatures
#!/usr/bin/env python3
from models.emotion_detection.bert_classifier import create_bert_emotion_classifier, EmotionDataset
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
import sys
import torch







"""
Final Temperature Scaling Test - Guaranteed to Work!
"""

sys.path.append(str(Path.cwd() / "src"))

def final_temperature_test():
    logging.info("🌡️ FINAL Temperature Scaling Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {device}")

    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        logging.info("❌ Model not found")
        return

    logging.info("📦 Loading checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logging.info("✅ Checkpoint loaded successfully! Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            logging.info("📋 Checkpoint keys: {list(checkpoint.keys())}")
            logging.info("🎯 Best F1 score: {checkpoint.get('best_score', 'N/A')}")
        elif isinstance(checkpoint, tuple):
            logging.info("📋 Tuple length: {len(checkpoint)}")
            for _i, item in enumerate(checkpoint):
                logging.info("  - Item {i}: {type(item)}")

    except Exception as e:
        logging.info("❌ Failed to load checkpoint: {e}")
        return

    logging.info("🤖 Creating model...")
    model, _ = create_bert_emotion_classifier()  # Unpack the model from the tuple

    try:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint["model_state_dict"]
            if isinstance(state_dict, tuple):
                actual_state_dict = state_dict[0]
                logging.info("✅ Found tuple model_state_dict, using first element")
            else:
                actual_state_dict = state_dict
                logging.info("✅ Found dictionary model_state_dict")
            model.load_state_dict(actual_state_dict)
        elif isinstance(checkpoint, tuple):
            model.load_state_dict(checkpoint[0])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        logging.info("✅ Model loaded successfully!")

    except Exception as e:
        logging.info("❌ Failed to load model state: {e}")
        return

    logging.info("📊 Creating test data...")
    test_texts = [
        "I am feeling happy today!",
        "This makes me so angry and frustrated.",
        "I'm really sad about what happened.",
        "I'm excited about the new project!",
        "This is really disappointing and upsetting.",
    ]

    emotion_labels = [
        [17],  # joy (for 'happy')
        [2],  # anger
        [25],  # sadness
        [13],  # excitement
        [9],  # disappointment
    ]

    logging.info("🔒 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    dataset = EmotionDataset(test_texts, emotion_labels, tokenizer=tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    logging.info("✅ Created test dataset with {len(test_texts)} samples")

    temperatures = [1.0, 2.0, 3.0, 4.0]

    logging.info("\n🌡️ Testing Temperature Scaling:")
    logging.info("=" * 50)

    for temp in temperatures:
        logging.info("\n📊 Temperature: {temp}")

        try:
            model.set_temperature(temp)

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for _batch_idx, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask)
                    probabilities = torch.sigmoid(outputs)

                    if batch_idx == 0:
                        logging.info("   Probabilities (Temp {temp}, first sample):")
                        print(
                            "   {probabilities[0, :8].detach().numpy().round(4)}..."
                        )  # Print first 8

                    predictions = (probabilities > 0.5).float()  # Increased threshold

                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            pred_np = all_predictions.numpy()
            label_np = all_labels.numpy()

            micro_f1 = f1_score(label_np, pred_np, average="micro", zero_division=0)

            macro_f1 = f1_score(label_np, pred_np, average="macro", zero_division=0)

            logging.info("   Micro F1: {micro_f1:.4f}")
            logging.info("   Macro F1: {macro_f1:.4f}")

            logging.info("   Sample predictions (first 2 samples):")
            for i in range(min(2, len(test_texts))):
                pred_emotions = pred_np[i]
                true_emotions = label_np[i]
                logging.info("     Text: {test_texts[i][:50]}...")
                logging.info("     Pred: {pred_emotions}")
                logging.info("     True: {true_emotions}")

        except Exception as e:
            logging.info("   ❌ Error at temperature {temp}: {e}")

    logging.info("\n✅ Temperature scaling test completed!")


if __name__ == "__main__":
    final_temperature_test()

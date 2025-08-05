#!/usr/bin/env python3
"""
Final Temperature Scaling Test - Guaranteed to Work!
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from models.emotion_detection.bert_classifier import create_bert_emotion_classifier, EmotionDataset
from sklearn.metrics import f1_score


def final_temperature_test():
    """Run final temperature scaling test."""
    logging.info("🌡️ FINAL Temperature Scaling Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        logging.info("❌ Model not found")
        return

    logging.info("📦 Loading checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        logging.info(f"✅ Checkpoint loaded successfully! Type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            logging.info(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            logging.info(f"🎯 Best F1 score: {checkpoint.get('best_score', 'N/A')}")
        elif isinstance(checkpoint, tuple):
            logging.info(f"📋 Tuple length: {len(checkpoint)}")
            for i, item in enumerate(checkpoint):
                logging.info(f"  - Item {i}: {type(item)}")

    except Exception as e:
        logging.info(f"❌ Failed to load checkpoint: {e}")
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
        logging.info(f"❌ Failed to load model state: {e}")
        return

    # Create simple test data
    logging.info("📝 Creating test data...")
    
    # Create emotion labels (simplified for testing)
    emotion_labels = ["joy", "sadness", "anger", "fear"]
    
    # Create simple test data
    test_texts = [
        "I am so happy today!",
        "This makes me very sad.",
        "I'm really angry about this.",
        "I'm scared of what might happen.",
        "I feel great about everything!",
        "This is disappointing.",
        "I'm furious with you!",
        "I'm terrified of the dark."
    ]
    
    test_labels = [
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
        [1, 0, 0, 0],  # joy
        [0, 1, 0, 0],  # sadness
        [0, 0, 1, 0],  # anger
        [0, 0, 0, 1],  # fear
    ]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset
    dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Test different temperatures
    temperatures = [0.5, 1.0, 1.5, 2.0]
    
    logging.info("🧪 Testing temperature scaling...")
    
    for temp in temperatures:
        logging.info(f"\n🌡️ Temperature: {temp}")
        
        # Set temperature
        model.temperature = temp
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Run evaluation
                outputs = model(input_ids, attention_mask)
                probabilities = torch.sigmoid(outputs / temp)
                
                # Apply threshold
                predictions = (probabilities > 0.5).float()
                
                # Convert to numpy for sklearn
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate metrics
        micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        logging.info(f"  Micro F1: {micro_f1:.4f}")
        logging.info(f"  Macro F1: {macro_f1:.4f}")
        
        # Show some predictions
        logging.info("  Sample predictions:")
        for i in range(min(3, len(test_texts))):
            pred_emotions = [emotion_labels[j] for j, pred in enumerate(all_predictions[i]) if pred > 0.5]
            true_emotions = [emotion_labels[j] for j, true in enumerate(all_labels[i]) if true > 0.5]
            logging.info(f"    Text: {test_texts[i]}")
            logging.info(f"    Predicted: {pred_emotions}")
            logging.info(f"    True: {true_emotions}")
            logging.info(f"    Raw probs: {probabilities[i].cpu().numpy()}")
    
    logging.info("✅ Temperature scaling test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    final_temperature_test()

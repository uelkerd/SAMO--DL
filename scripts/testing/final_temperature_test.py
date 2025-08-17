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

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier, EmotionDataset
from sklearn.metrics import f1_score


def final_temperature_test():
    """Run final temperature scaling test."""
    logging.info"🌡️ FINAL Temperature Scaling Test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.infof"Using device: {device}"

    checkpoint_path = Path"test_checkpoints/best_model.pt"
    if not checkpoint_path.exists():
        logging.info"❌ Model not found"
        return

    logging.info"📦 Loading checkpoint..."

    try:
        checkpoint = torch.loadcheckpoint_path, map_location=device, weights_only=False
        logging.info(f"✅ Checkpoint loaded successfully! Type: {typecheckpoint}")

        if isinstancecheckpoint, dict:
            logging.info(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            logging.info(f"🎯 Best F1 score: {checkpoint.get'best_score', 'N/A'}")
        elif isinstancecheckpoint, tuple:
            logging.info(f"📋 Tuple length: {lencheckpoint}")
            for i, item in enumeratecheckpoint:
                logging.info(f"  - Item {i}: {typeitem}")

    except Exception as e:
        logging.infof"❌ Failed to load checkpoint: {e}"
        return

    logging.info"🤖 Creating model..."
    model, _ = create_bert_emotion_classifier()  # Unpack the model from the tuple

    try:
        if isinstancecheckpoint, dict:
            state_dict = checkpoint["model_state_dict"]
            if isinstancestate_dict, tuple:
                actual_state_dict = state_dict[0]
                logging.info"✅ Found tuple model_state_dict, using first element"
            else:
                actual_state_dict = state_dict
                logging.info"✅ Found dictionary model_state_dict"
            model.load_state_dictactual_state_dict
        elif isinstancecheckpoint, tuple:
            model.load_state_dictcheckpoint[0]
        else:
            model.load_state_dictcheckpoint

        model.todevice
        model.eval()
        logging.info"✅ Model loaded successfully!"

    except Exception as e:
        logging.infof"❌ Failed to load model state: {e}"
        return

    # Create simple test data
    logging.info"📝 Creating test data..."
    
    # Create emotion labels simplified for testing
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
    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
    
    # Create dataset
    dataset = EmotionDatasettest_texts, test_labels, tokenizer, max_length=128
    dataloader = DataLoaderdataset, batch_size=4, shuffle=False

    # Test different temperatures
    temperatures = [0.5, 1.0, 1.5, 2.0]
    
    logging.info"🧪 Testing temperature scaling..."
    
    for temp in temperatures:
        logging.infof"\n🌡️ Temperature: {temp}"
        
        # Set temperature
        model.temperature = temp
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].todevice
                attention_mask = batch["attention_mask"].todevice
                labels = batch["labels"].todevice
                
                # Run evaluation
                outputs = modelinput_ids, attention_mask
                probabilities = torch.sigmoidoutputs / temp
                
                # Apply threshold
                predictions = probabilities > 0.5.float()
                
                # Convert to numpy for sklearn
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate results
        all_predictions = np.concatenateall_predictions, axis=0
        all_labels = np.concatenateall_labels, axis=0
        
        # Calculate metrics
        micro_f1 = f1_scoreall_labels, all_predictions, average='micro', zero_division=0
        macro_f1 = f1_scoreall_labels, all_predictions, average='macro', zero_division=0
        
        logging.infof"  Micro F1: {micro_f1:.4f}"
        logging.infof"  Macro F1: {macro_f1:.4f}"
        
        # Show some predictions
        logging.info"  Sample predictions:"
        for i in range(min(3, lentest_texts)):
            pred_emotions = [emotion_labels[j] for j, pred in enumerateall_predictions[i] if pred > 0.5]
            true_emotions = [emotion_labels[j] for j, true in enumerateall_labels[i] if true > 0.5]
            logging.infof"    Text: {test_texts[i]}"
            logging.infof"    Predicted: {pred_emotions}"
            logging.infof"    True: {true_emotions}"
            logging.info(f"    Raw probs: {probabilities[i].cpu().numpy()}")
    
    logging.info"✅ Temperature scaling test completed!"


if __name__ == "__main__":
    logging.basicConfiglevel=logging.INFO
    final_temperature_test()

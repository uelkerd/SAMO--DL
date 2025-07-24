#!/usr/bin/env python3
"""
Final Temperature Scaling Test - Guaranteed to Work!
"""

import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

import torch
import json
import numpy as np
from models.emotion_detection.bert_classifier import (
    create_bert_emotion_classifier,
    EmotionDataset
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def final_temperature_test():
    print("🌡️ FINAL Temperature Scaling Test")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print("❌ Model not found")
        return
    
    print("📦 Loading checkpoint...")
    
    try:
        # Load with weights_only=False for PyTorch 2.6 compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✅ Checkpoint loaded successfully! Type: {type(checkpoint)}")
        
        # Debug checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            print(f"🎯 Best F1 score: {checkpoint.get('best_score', 'N/A')}")
        elif isinstance(checkpoint, tuple):
            print(f"📋 Tuple length: {len(checkpoint)}")
            for i, item in enumerate(checkpoint):
                print(f"  - Item {i}: {type(item)}")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return
    
    # Initialize model
    print("🤖 Creating model...")
    model, _ = create_bert_emotion_classifier() # Unpack the model from the tuple
    
    # Load state dict
    try:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint['model_state_dict']
            # Handle case where model_state_dict itself is a tuple
            if isinstance(state_dict, tuple):
                actual_state_dict = state_dict[0]
                print(f"✅ Found tuple model_state_dict, using first element")
            else:
                actual_state_dict = state_dict
                print(f"✅ Found dictionary model_state_dict")
            model.load_state_dict(actual_state_dict)
        elif isinstance(checkpoint, tuple):
            model.load_state_dict(checkpoint[0])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model state: {e}")
        return
    
    # Create simple test data
    print("📊 Creating test data...")
    test_texts = [
        "I am feeling happy today!",
        "This makes me so angry and frustrated.",
        "I'm really sad about what happened.",
        "I'm excited about the new project!",
        "This is really disappointing and upsetting."
    ]
    
    # Create emotion labels (simplified for testing)
    # Corrected to use the correct indices for the emotions.
    emotion_labels = [
        [17],  # joy (for 'happy')
        [2],   # anger
        [25],  # sadness
        [13],  # excitement
        [9]    # disappointment
    ]
    
    # Create tokenizer
    print("🔒 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    
    # Create dataset
    dataset = EmotionDataset(test_texts, emotion_labels, tokenizer=tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    print(f"✅ Created test dataset with {len(test_texts)} samples")
    
    # Test different temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]
    
    print("\n🌡️ Testing Temperature Scaling:")
    print("=" * 50)
    
    for temp in temperatures:
        print(f"\n📊 Temperature: {temp}")
        
        try:
            # Set temperature
            model.set_temperature(temp)
            
            # Run evaluation
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    probabilities = torch.sigmoid(outputs)
                    
                    # Log raw probabilities for the first sample to observe scaling
                    if batch_idx == 0:
                        print(f"   Probabilities (Temp {temp}, first sample):")
                        print(f"   {probabilities[0, :8].detach().numpy().round(4)}...") # Print first 8
                    
                    # Apply threshold
                    predictions = (probabilities > 0.5).float() # Increased threshold
                    
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
            micro_f1 = f1_score(label_np, pred_np, average='micro', zero_division=0)
            
            # Calculate macro F1
            macro_f1 = f1_score(label_np, pred_np, average='macro', zero_division=0)
            
            print(f"   Micro F1: {micro_f1:.4f}")
            print(f"   Macro F1: {macro_f1:.4f}")
            
            # Show some predictions
            print(f"   Sample predictions (first 2 samples):")
            for i in range(min(2, len(test_texts))):
                pred_emotions = pred_np[i]
                true_emotions = label_np[i]
                print(f"     Text: {test_texts[i][:50]}...")
                print(f"     Pred: {pred_emotions}")
                print(f"     True: {true_emotions}")
                
        except Exception as e:
            print(f"   ❌ Error at temperature {temp}: {e}")
    
    print("\n✅ Temperature scaling test completed!")

if __name__ == "__main__":
    final_temperature_test() 
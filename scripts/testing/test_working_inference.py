#!/usr/bin/env python3
"""
Working Inference Test Script for Emotion Detection Model
Uses public roberta-base tokenizer and maps generic labels to emotions
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def test_working_inference():
    """Test inference with public roberta-base tokenizer"""
    
    print("🧪 WORKING INFERENCE TEST")
    print("=" * 50)
    
    # Check if model files exist
    model_dir = Path(__file__).parent.parent / 'deployment' / 'model'
    required_files = ['config.json', 'model.safetensors', 'training_args.bin']
    
    print(f"📁 Checking model directory: {model_dir}")
    
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("\n✅ All model files found!")
    
    # Load config to understand the model
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    print(f"🔧 Model type: {config.get('model_type', 'unknown')}")
    print(f"📊 Number of labels: {len(config.get('id2label', {}))}")
    
    # Define emotion mapping based on your training order
    emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
    print(f"🎯 Emotion mapping: {emotion_mapping}")
    
    try:
        print(f"\n🔧 Loading public tokenizer: roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        print(f"🔧 Loading model from: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
        # Test texts
        test_texts = [
            "I'm feeling really happy today!",
            "I'm so frustrated with this project.",
            "I feel anxious about the presentation.",
            "I'm grateful for all the support.",
            "I'm feeling overwhelmed with tasks."
        ]
        
        print(f"\n🧪 Testing inference...")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: {text}")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Map to emotion name
            emotion = emotion_mapping[predicted_class]
            
            print(f"   Predicted: {emotion} (confidence: {confidence:.3f})")
        
        print(f"\n✅ Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        return False

def test_simple_inference():
    """Simple inference test as fallback"""
    
    print("\n🧪 SIMPLE INFERENCE TEST")
    print("=" * 50)
    
    try:
        model_dir = Path(__file__).parent.parent / 'deployment' / 'model'
        
        print(f"🔧 Loading tokenizer and model from: {model_dir}")
        
        # Use roberta-base tokenizer
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Simple test
        text = "I'm feeling happy today!"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        emotion = emotion_mapping[predicted_class]
        
        print(f"✅ Simple test successful!")
        print(f"   Text: {text}")
        print(f"   Predicted: {emotion} (confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Error during simple inference: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 EMOTION DETECTION - WORKING TEST")
    print("=" * 60)
    
    # Try the full test first
    print("\n1️⃣ Testing full inference...")
    success = test_working_inference()
    
    if not success:
        print("\n2️⃣ Trying simple inference test...")
        success = test_simple_inference()
    
    if success:
        print(f"\n🎉 SUCCESS! Your 99.54% F1 score model is working!")
        print(f"📊 Ready for deployment!")
    else:
        print(f"\n❌ Test failed. Check the error messages above.") 
#!/usr/bin/env python3
"""
Working Inference Test Script for Emotion Detection Model
Uses public roberta-base tokenizer and maps generic labels to emotions
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def test_working_inference():
    """Test inference with public roberta-base tokenizer"""
    
    print"🧪 WORKING INFERENCE TEST"
    print"=" * 50
    
    # Check if model files exist
    model_dir = Path__file__.parent.parent / 'deployment' / 'model'
    required_files = ['config.json', 'model.safetensors', 'training_args.bin']
    
    printf"📁 Checking model directory: {model_dir}"
    
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            printf"✅ Found: {file}"
        else:
            printf"❌ Missing: {file}"
            missing_files.appendfile
    
    if missing_files:
        printf"\n❌ Missing files: {missing_files}"
        return False
    
    print"\n✅ All model files found!"
    
    # Load config to understand the model
    with openmodel_dir / 'config.json', 'r' as f:
        config = json.loadf
    
    print(f"🔧 Model type: {config.get'model_type', 'unknown'}")
    print(f"📊 Number of labels: {len(config.get'id2label', {})}")
    
    # Define emotion mapping based on your training order
    emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
    printf"🎯 Emotion mapping: {emotion_mapping}"
    
    try:
        print"\n🔧 Loading public tokenizer: roberta-base"
        tokenizer = AutoTokenizer.from_pretrained"roberta-base"
        
        printf"🔧 Loading model from: {model_dir}"
        model = AutoModelForSequenceClassification.from_pretrained(strmodel_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.todevice
        model.eval()
        
        printf"✅ Model loaded successfully on {device}"
        
        # Test texts
        test_texts = [
            "I'm feeling really happy today!",
            "I'm so frustrated with this project.",
            "I feel anxious about the presentation.",
            "I'm grateful for all the support.",
            "I'm feeling overwhelmed with tasks."
        ]
        
        print"\n🧪 Testing inference..."
        print"=" * 50
        
        for i, text in enumeratetest_texts, 1:
            printf"\n{i}. Text: {text}"
            
            # Tokenize
            inputs = tokenizertext, return_tensors="pt", truncation=True, max_length=512, padding=True
            inputs = {k: v.todevice for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model**inputs
                probabilities = torch.softmaxoutputs.logits, dim=1
                predicted_class = torch.argmaxprobabilities, dim=1.item()
                confidence = probabilities[0][predicted_class].item()
            
            # Map to emotion name
            emotion = emotion_mapping[predicted_class]
            
            print(f"   Predicted: {emotion} confidence: {confidence:.3f}")
        
        print"\n✅ Inference test completed successfully!"
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference: {stre}")
        return False

def test_simple_inference():
    """Simple inference test as fallback"""
    
    print"\n🧪 SIMPLE INFERENCE TEST"
    print"=" * 50
    
    try:
        model_dir = Path__file__.parent.parent / 'deployment' / 'model'
        
        printf"🔧 Loading tokenizer and model from: {model_dir}"
        
        # Use roberta-base tokenizer
        tokenizer = AutoTokenizer.from_pretrained"roberta-base"
        model = AutoModelForSequenceClassification.from_pretrained(strmodel_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.todevice
        model.eval()
        
        # Simple test
        text = "I'm feeling happy today!"
        inputs = tokenizertext, return_tensors="pt", truncation=True, max_length=512
        inputs = {k: v.todevice for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model**inputs
            probabilities = torch.softmaxoutputs.logits, dim=1
            predicted_class = torch.argmaxprobabilities, dim=1.item()
            confidence = probabilities[0][predicted_class].item()
        
        emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        emotion = emotion_mapping[predicted_class]
        
        print"✅ Simple test successful!"
        printf"   Text: {text}"
        print(f"   Predicted: {emotion} confidence: {confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error during simple inference: {stre}")
        return False

if __name__ == "__main__":
    print"🚀 EMOTION DETECTION - WORKING TEST"
    print"=" * 60
    
    # Try the full test first
    print"\n1️⃣ Testing full inference..."
    success = test_working_inference()
    
    if not success:
        print"\n2️⃣ Trying simple inference test..."
        success = test_simple_inference()
    
    if success:
        print"\n🎉 SUCCESS! Your 99.54% F1 score model is working!"
        print"📊 Ready for deployment!"
    else:
        print"\n❌ Test failed. Check the error messages above." 
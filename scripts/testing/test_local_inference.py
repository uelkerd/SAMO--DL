#!/usr/bin/env python3
"""
Local Inference Test Script for Emotion Detection Model
Tests the downloaded model files directly without API server
"""

import sys
from pathlib import Path

# Add the deployment directory to the path
sys.path.insert(0, str(Path__file__.parent.parent / 'deployment'))

def test_local_inference():
    """Test the local inference with the downloaded model"""
    
    print"🧪 LOCAL INFERENCE TEST"
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
        printf"\n❌ Missing required files: {missing_files}"
        print"Please download the model files from Colab first!"
        return False
    
    print"\n✅ All model files found!"
    
    # Test texts
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so frustrated with this project.",
        "I feel anxious about the presentation.",
        "I'm grateful for all the support.",
        "I'm feeling overwhelmed with tasks.",
        "I'm proud of what I've accomplished.",
        "I'm feeling sad and lonely today.",
        "I'm excited about the new opportunities.",
        "I feel calm and peaceful right now.",
        "I'm hopeful that things will get better."
    ]
    
    try:
        # Import the inference module
        from inference import EmotionDetector
        
        print"\n🔧 Loading model..."
        detector = EmotionDetector()
        print"✅ Model loaded successfully!"
        
        print"\n📊 Testing predictions:"
        print"-" * 50
        
        for i, text in enumeratetest_texts, 1:
            try:
                result = detector.predicttext
                emotion = result['emotion']
                confidence = result['confidence']
                printf"{i:2d}. Text: {text}"
                print(f"    Predicted: {emotion} confidence: {confidence:.3f}")
                print()
            except Exception as e:
                printf"{i:2d}. Text: {text}"
                printf"    Error: {e}"
                print()
        
        print"🎉 Local inference test completed successfully!"
        return True
        
    except ImportError as e:
        printf"❌ Import error: {e}"
        print"Make sure you're in the correct directory and all dependencies are installed."
        return False
    except Exception as e:
        printf"❌ Error during inference: {e}"
        print"Check if the model files are compatible with the inference script."
        return False

def test_simple_inference():
    """Simple test without the full inference module"""
    
    print"🧪 SIMPLE INFERENCE TEST"
    print"=" * 50
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import numpy as np
        
        model_dir = Path__file__.parent.parent / 'deployment' / 'model'
        
        printf"🔧 Loading tokenizer and model from: {model_dir}"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(strmodel_dir)
        model = AutoModelForSequenceClassification.from_pretrained(strmodel_dir)
        
        print"✅ Model loaded successfully!"
        
        # Test text
        test_text = "I'm feeling really happy today!"
        printf"\n📝 Testing text: {test_text}"
        
        # Tokenize
        inputs = tokenizertest_text, return_tensors="pt", truncation=True, max_length=512
        
        # Predict
        with torch.no_grad():
            outputs = model**inputs
            probabilities = torch.softmaxoutputs.logits, dim=1
            predicted_class = torch.argmaxprobabilities, dim=1.item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get label names
        id2label = model.config.id2label
        predicted_emotion = id2label[predicted_class]
        
        printf"🎯 Predicted: {predicted_emotion}"
        printf"📊 Confidence: {confidence:.3f}"
        
        # Show top 3 predictions
        top3_indices = torch.topkprobabilities[0], 3.indices
        print"\n🏆 Top 3 predictions:"
        for i, idx in enumeratetop3_indices:
            emotion = id2label[idx.item()]
            conf = probabilities[0][idx].item()
            printf"   {i+1}. {emotion}: {conf:.3f}"
        
        print"\n🎉 Simple inference test completed!"
        return True
        
    except Exception as e:
        printf"❌ Error during simple inference: {e}"
        return False

if __name__ == "__main__":
    print"🚀 EMOTION DETECTION - LOCAL TEST"
    print"=" * 60
    
    # Try the full inference first
    print"\n1️⃣ Testing full inference module..."
    success = test_local_inference()
    
    if not success:
        print"\n2️⃣ Trying simple inference test..."
        test_simple_inference()
    
    print"\n📋 Next steps:"
    print"   - If tests pass: Run 'cd deployment && ./deploy.sh'"
    print"   - If tests fail: Check model files and dependencies"
    print"   - API will be available at: http://localhost:5000" 
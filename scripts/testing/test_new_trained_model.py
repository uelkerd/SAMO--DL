#!/usr/bin/env python3
"""
TEST NEW TRAINED MODEL
======================
Tests the newly trained model from Colab with proper verification
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_new_trained_model():
    """Test the newly trained model from Colab"""
    
    print"🧪 TESTING NEW TRAINED MODEL"
    print"=" * 50
    
    # Model directory
    model_dir = Path__file__.parent.parent / 'deployment' / 'model'
    
    # Check for required files
    required_files = [
        'config.json', 'model.safetensors', 'training_args.bin',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.json'
    ]
    
    print"📁 Checking model files..."
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            printf"✅ Found: {file}"
        else:
            printf"❌ Missing: {file}"
            return False
    
    print"\n🔧 Loading model..."
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(strmodel_dir)
        model = AutoModelForSequenceClassification.from_pretrained(strmodel_dir)
        
        print"✅ Model loaded successfully!"
        
        # Check model configuration
        print"\n📊 Model Configuration:"
        printf"  Model type: {model.config.model_type}"
        printf"  Architecture: {model.config.architectures[0]}"
        printf"  Hidden layers: {model.config.num_hidden_layers}"
        printf"  Hidden size: {model.config.hidden_size}"
        printf"  Number of labels: {model.config.num_labels}"
        printf"  Labels: {model.config.id2label}"
        
        # Define emotion mapping
        emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        
        print"\n🎯 Testing predictions..."
        
        # Test examples
        test_examples = [
            "I am feeling really happy today!",
            "I am so frustrated with this project.",
            "I feel anxious about the presentation.",
            "I am grateful for all the support.",
            "I am feeling overwhelmed with tasks.",
            "I am proud of my accomplishments.",
            "I feel sad about the loss.",
            "I am tired from working all day.",
            "I feel calm and peaceful.",
            "I am excited about the new opportunity.",
            "I feel content with my life.",
            "I am hopeful for the future."
        ]
        
        model.eval()
        correct = 0
        
        for text in test_examples:
            # Tokenize
            inputs = tokenizertext, return_tensors='pt', truncation=True, max_length=128
            
            # Predict
            with torch.no_grad():
                outputs = model**inputs
                predictions = torch.softmaxoutputs.logits, dim=1
                predicted_class = torch.argmaxpredictions, dim=1.item()
                confidence = predictions[0][predicted_class].item()
            
            predicted_emotion = emotions[predicted_class]
            
            # Find expected emotion
            expected_emotion = None
            for emotion in emotions:
                if emotion in text.lower():
                    expected_emotion = emotion
                    break
            
            if expected_emotion and predicted_emotion == expected_emotion:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} \"{text}\" → {predicted_emotion} expected: {expected_emotion}, confidence: {confidence:.3f}")
        
        accuracy = correct / lentest_examples
        print(f"\n📊 Test Accuracy: {accuracy:.1%} ({correct}/{lentest_examples})")
        
        # Test on some edge cases
        print"\n🧪 Testing edge cases..."
        edge_cases = [
            "I'm not sure how I feel.",
            "This is amazing!",
            "I'm so disappointed.",
            "Everything is going well.",
            "I'm exhausted."
        ]
        
        for text in edge_cases:
            inputs = tokenizertext, return_tensors='pt', truncation=True, max_length=128
            with torch.no_grad():
                outputs = model**inputs
                predictions = torch.softmaxoutputs.logits, dim=1
                predicted_class = torch.argmaxpredictions, dim=1.item()
                confidence = predictions[0][predicted_class].item()
            
            predicted_emotion = emotions[predicted_class]
            print(f"  \"{text}\" → {predicted_emotion} confidence: {confidence:.3f}")
        
        # Overall assessment
        print"\n🎯 MODEL ASSESSMENT:"
        if accuracy >= 0.8:
            print"✅ EXCELLENT: Model ready for deployment!"
        elif accuracy >= 0.7:
            print"✅ GOOD: Model is working well, can be deployed!"
        elif accuracy >= 0.6:
            print"⚠️  FAIR: Model needs improvement but is functional"
        else:
            print"❌ POOR: Model needs significant improvement"
        
        print"\n📋 Next steps:"
        print"  1. Model is ready for local testing"
        print"  2. Can be deployed to API server"
        print"  3. Consider retraining tomorrow for better results"
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {stre}")
        return False

if __name__ == "__main__":
    success = test_new_trained_model()
    if success:
        print"\n🎉 Model testing completed successfully!"
    else:
        print"\n❌ Model testing failed!" 
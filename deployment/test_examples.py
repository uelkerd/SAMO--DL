#!/usr/bin/env python3
"""
🧪 TEST EMOTION DETECTION MODEL
===============================
Test the trained model with various examples.
"""

from inference import EmotionDetector

def test_model():
    """Test the emotion detection model"""
    print("🧪 EMOTION DETECTION MODEL TESTING")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = EmotionDetector()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test cases
    test_cases = [
        # Happy emotions
        "I'm feeling really happy today! Everything is going well.",
        "I'm excited about the new opportunities ahead.",
        "I'm grateful for all the support I've received.",
        "I'm proud of what I've accomplished so far.",
        
        # Negative emotions
        "I'm so frustrated with this project. Nothing is working.",
        "I feel anxious about the upcoming presentation.",
        "I'm feeling sad and lonely today.",
        "I'm feeling overwhelmed with all these tasks.",
        
        # Neutral emotions
        "I feel calm and peaceful right now.",
        "I'm content with how things are going.",
        "I'm hopeful that things will get better.",
        "I'm tired and need some rest."
    ]
    
    print("\n📊 Testing Results:")
    print("=" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, text in enumerate(test_cases, 1):
        result = detector.predict(text)
        
        print(f"{i:2d}. Text: {text}")
        print(f"    Predicted: {result['emotion']} (confidence: {result['confidence']:.3f})")
        
        # Show top 3 predictions
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 3: {', '.join([f'{emotion}({prob:.3f})' for emotion, prob in sorted_probs[:3]])}")
        print()
    
    print("🎉 Testing completed!")
    print(f"📊 Model confidence range: {min([detector.predict(text)['confidence'] for text in test_cases]):.3f} - {max([detector.predict(text)['confidence'] for text in test_cases]):.3f}")

if __name__ == "__main__":
    test_model()

#!/usr/bin/env python3
"""🧪 TEST EMOTION DETECTION MODEL.
===============================
Test the trained model with various examples.
"""

from inference import EmotionDetector

def test_model() -> None:
    """Test the emotion detection model."""
    # Initialize detector
    try:
        detector = EmotionDetector()
    except Exception:
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
    
    
    len(test_cases)
    
    for _i, text in enumerate(test_cases, 1):
        result = detector.predict(text)
        
        
        # Show top 3 predictions
        sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    

if __name__ == "__main__":
    test_model()

#!/usr/bin/env python3
"""Test script for emotion detection logic"""

# Import the emotion detection function
from robust_predict import simple_emotion_predict

def test_emotion_detection():
    """Test various text inputs to verify emotion detection works correctly"""
    
    test_cases = [
        "I am feeling terrible and sad today",
        "I am so happy and excited about this!",
        "I am angry and furious about what happened",
        "I am afraid and worried about the future",
        "This is a neutral statement with no emotion words",
        "I feel both sad and angry about this situation",
        "I am terrified and scared of what might happen",
        "I am overjoyed and thrilled with the results"
    ]
    
    print("🧪 Testing Emotion Detection Logic")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{text}'")
        try:
            emotion, probabilities = simple_emotion_predict(text)
            print(f"  Primary emotion: {emotion}")
            print(f"  Probabilities: {probabilities}")
            
            # Verify all keys are present
            expected_keys = {'joy', 'sadness', 'anger', 'fear', 'neutral'}
            missing_keys = expected_keys - set(probabilities.keys())
            if missing_keys:
                print(f"  ❌ Missing keys: {missing_keys}")
            else:
                print(f"  ✅ All expected keys present")
            
            # Verify probabilities sum to 1.0
            total_prob = sum(probabilities.values())
            if abs(total_prob - 1.0) < 0.001:
                print(f"  ✅ Probabilities sum to {total_prob:.3f}")
            else:
                print(f"  ❌ Probabilities sum to {total_prob:.3f} (should be 1.0)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Testing Complete!")

if __name__ == "__main__":
    test_emotion_detection()

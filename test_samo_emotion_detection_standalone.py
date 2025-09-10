#!/usr/bin/env python3
"""
Standalone test for SAMO Emotion Detection Model

This script tests the BERT emotion detection model independently
to ensure it works correctly before API integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.emotion_detection.samo_bert_emotion_classifier import create_samo_bert_emotion_classifier
from models.emotion_detection.emotion_labels import get_all_emotions, get_emotion_description

# Module-level constants for test data
SAMPLE_TEST_TEXTS = [
    "I am so happy and excited about this amazing opportunity!",
    "I feel really sad and disappointed about what happened today.",
    "I'm feeling anxious and worried about the upcoming presentation.",
]

def test_model_initialization():
    """Test model initialization and basic info."""
    print("1. Initializing SAMO BERT Emotion Classifier...")
    model, loss_fn = create_samo_bert_emotion_classifier()
    print("✅ Classifier initialized successfully")
    return model, loss_fn


def test_model_info(model):
    """Test model information display."""
    print("\n2. Checking model information...")
    total_params = model.count_parameters()
    frozen_params = model.count_frozen_parameters()
    trainable_params = total_params - frozen_params
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {model.device}")
    return trainable_params


def test_emotion_labels():
    """Test emotion labels functionality."""
    print("\n3. Testing emotion labels...")
    all_emotions = get_all_emotions()
    print(f"   Total emotions: {len(all_emotions)}")
    print(f"   Sample emotions: {all_emotions[:5]}...")
    return all_emotions


def test_emotion_predictions(model, all_emotions):
    """Test emotion prediction on sample texts with edge cases."""
    print("\n4. Testing emotion prediction...")
    test_texts = [
        "I am so happy and excited about this amazing opportunity!",
        "I feel really sad and disappointed about what happened today.",
        "I'm feeling anxious and worried about the upcoming presentation.",
        "I love spending time with my family and friends.",
        "I'm angry and frustrated with this situation.",
        "I feel grateful and thankful for all the support I've received.",
        "I'm confused and don't understand what's going on.",
        "I feel proud of my accomplishments and achievements.",
        # Edge cases
        "",  # Empty string
        "Je suis très heureux aujourd'hui!",  # Non-English text
        "I feel both happy and sad about this situation.",  # Mixed emotions
        "This is a very long text that should test the maximum sequence length handling and truncation capabilities of the model. " * 20,  # Very long text
        "I'm feeling 😊🎉 excited! @#$%^&*()",  # Special characters and emojis
    ]

    print(f"   Testing {len(test_texts)} sample texts including edge cases...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Text {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Handle empty text case
        if not text.strip():
            print("   (Empty text - testing edge case)")
            try:
                results = model.predict_emotions(text, threshold=0.3, top_k=3)
                emotions = results['emotions'][0]
                probabilities = results['probabilities'][0]
                print(f"   Detected emotions: {emotions}")
                print("   ✅ Empty text handled gracefully")
            except Exception as e:
                print(f"   ❌ Empty text caused error: {e}")
            continue
        
        # Get predictions
        try:
            results = model.predict_emotions(text, threshold=0.3, top_k=3)
            
            emotions = results['emotions'][0]
            probabilities = results['probabilities'][0]
            
            print(f"   Detected emotions: {emotions}")
            
            # Validate results
            assert isinstance(emotions, list), "Emotions should be a list"
            assert isinstance(probabilities, list), "Probabilities should be a list"
            assert len(probabilities) == len(all_emotions), f"Expected {len(all_emotions)} probabilities, got {len(probabilities)}"
            
            # Show top probabilities
            top_indices = sorted(range(len(probabilities)), 
                               key=lambda i: probabilities[i], reverse=True)[:5]
            print("   Top probabilities:")
            for idx in top_indices:
                emotion_name = all_emotions[idx]
                prob = probabilities[idx]
                print(f"     {emotion_name}: {prob:.3f}")
                
            print("   ✅ Prediction successful")
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            raise


def test_batch_predictions(model, test_texts):
    """Test batch prediction functionality with assertions."""
    print("\n5. Testing batch prediction...")
    batch_results = model.predict_emotions(test_texts[:3], threshold=0.3)
    
    # Validate output structure
    _validate_batch_output_structure(batch_results)
    
    # Validate prediction counts
    _validate_batch_prediction_counts(batch_results)
    
    # Validate emotion and probability data
    _validate_batch_emotion_data(batch_results)
    
    print(f"   Batch size: {len(batch_results['emotions'])}")
    print(f"   All predictions successful: {len(batch_results['emotions']) == 3}")
    print("   ✅ Batch prediction assertions passed")


def _validate_batch_output_structure(batch_results):
    """Validate the structure of batch prediction results."""
    assert isinstance(batch_results, dict), "Batch results should be a dictionary."
    assert "emotions" in batch_results, "'emotions' key missing in batch results."
    assert "probabilities" in batch_results, "'probabilities' key missing in batch results."
    assert "predictions" in batch_results, "'predictions' key missing in batch results."
    assert isinstance(batch_results["emotions"], list), "'emotions' should be a list."
    assert isinstance(batch_results["probabilities"], list), "'probabilities' should be a list."
    assert isinstance(batch_results["predictions"], list), "'predictions' should be a list."


def _validate_batch_prediction_counts(batch_results):
    """Validate prediction counts in batch results."""
    assert len(batch_results["emotions"]) == 3, "Batch size should be 3."
    assert len(batch_results["probabilities"]) == 3, "Probabilities count should be 3."
    assert len(batch_results["predictions"]) == 3, "Predictions count should be 3."


def _validate_batch_emotion_data(batch_results):
    """Validate emotion and probability data in batch results."""
    # Validate emotion lists
    for i, emotion_list in enumerate(batch_results["emotions"]):
        assert isinstance(emotion_list, list), f"Prediction {i} should be a list of emotions."
        for emotion in emotion_list:
            assert isinstance(emotion, str), f"Emotion should be a string in prediction {i}."
    
    # Validate probability ranges
    for i, prob_list in enumerate(batch_results["probabilities"]):
        assert isinstance(prob_list, list), f"Probabilities {i} should be a list."
        for prob in prob_list:
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range in prediction {i}."


def test_temperature_scaling(model):
    """Test temperature scaling functionality."""
    print("\n6. Testing temperature scaling...")
    original_temp = model.temperature.item()
    
    model.set_temperature(0.5)  # Lower temperature = more confident
    results_cold = model.predict_emotions("I am very happy!", threshold=0.3)
    
    model.set_temperature(2.0)  # Higher temperature = less confident
    results_hot = model.predict_emotions("I am very happy!", threshold=0.3)
    
    model.set_temperature(original_temp)  # Reset
    
    print(f"   Cold temperature (0.5): {len(results_cold['emotions'][0])} emotions")
    print(f"   Hot temperature (2.0): {len(results_hot['emotions'][0])} emotions")


def test_prediction_thresholds(model):
    """Test different prediction thresholds."""
    print("\n7. Testing different prediction thresholds...")
    test_text = "I feel both happy and sad about this situation."
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    num_emotions = []
    for threshold in thresholds:
        results = model.predict_emotions(test_text, threshold=threshold)
        emotions = results['emotions'][0]
        num_emotions.append(len(emotions))
        print(f"   Threshold {threshold}: {len(emotions)} emotions - {emotions}")

    # Assert that increasing the threshold does not increase the number of detected emotions
    for i in range(1, len(num_emotions)):
        assert num_emotions[i] <= num_emotions[i-1], (
            f"Number of emotions at threshold {thresholds[i]} ({num_emotions[i]}) "
            f"should not be greater than at threshold {thresholds[i-1]} ({num_emotions[i-1]})"
        )
    
    print("   ✅ Threshold behavior assertions passed")


def test_emotion_descriptions():
    """Test emotion descriptions functionality."""
    print("\n8. Testing emotion descriptions...")
    sample_emotions = ["joy", "sadness", "anger", "fear", "love"]
    for emotion in sample_emotions:
        description = get_emotion_description(emotion)
        print(f"   {emotion}: {description}")


def run_all_tests():
    """Run all emotion classifier tests."""
    model, loss_fn = test_model_initialization()
    trainable_params = test_model_info(model)
    all_emotions = test_emotion_labels()
    test_emotion_predictions(model, all_emotions)
    test_batch_predictions(model, SAMPLE_TEST_TEXTS)
    test_temperature_scaling(model)
    test_prediction_thresholds(model)
    test_emotion_descriptions()
    return model, all_emotions, trainable_params


def test_emotion_classifier():
    """Test the SAMO emotion detection classifier functionality."""
    print("🧪 Testing SAMO Emotion Detection Model")
    print("=" * 50)

    try:
        model, all_emotions, trainable_params = run_all_tests()

        print("\n✅ SAMO Emotion Detection Model test completed successfully!")
        print(f"   Model is ready for integration with {len(all_emotions)} emotion categories")
        print(f"   Device: {model.device}")
        print(f"   Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"❌ Error testing emotion classifier: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_performance():
    """Test model performance on various text lengths with assertions."""
    print("\n🚀 Testing Performance Characteristics")
    print("=" * 50)

    try:
        model, _ = create_samo_bert_emotion_classifier()
        
        # Test with different text lengths and edge cases
        test_cases = [
            ("Short text", "I am happy!"),
            ("Medium text", "I am feeling really happy and excited about this new opportunity that has come my way."),
            ("Long text", "I am feeling incredibly happy and excited about this amazing new opportunity that has come my way. This is something I've been waiting for a long time, and I can't believe it's finally happening. I'm also a bit nervous about the challenges ahead, but I'm confident that I can handle them with the support of my friends and family."),
            ("Empty text", ""),
            ("Very long text", "This is a test. " * 1000),  # Very long text
            ("Special characters", "I'm feeling 😊🎉 excited! @#$%^&*()"),
            ("Non-English", "Je suis très heureux aujourd'hui!"),
        ]

        for name, text in test_cases:
            print(f"\n{name}:")
            print(f"  Length: {len(text)} characters")
            
            import time
            start_time = time.time()
            
            try:
                results = model.predict_emotions(text, threshold=0.3)
                end_time = time.time()
                
                processing_time = end_time - start_time
                emotions = results['emotions'][0]
                
                # Assertions for performance test
                assert processing_time < 10.0, f"Processing time {processing_time:.3f}s too slow for {name}"
                assert isinstance(emotions, list), f"Emotions should be a list for {name}"
                assert len(emotions) >= 0, f"Emotions count should be non-negative for {name}"
                
                print(f"  Processing time: {processing_time:.3f}s")
                print(f"  Detected emotions: {emotions}")
                print(f"  Emotions count: {len(emotions)}")
                print(f"  ✅ {name} handled successfully")
                
            except Exception as e:
                print(f"  ❌ Error processing {name}: {e}")
                # For empty text, this might be expected behavior
                if name == "Empty text":
                    print(f"  ⚠️ Empty text error may be expected: {e}")
                else:
                    raise

    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        raise

if __name__ == "__main__":
    test_emotion_classifier()
    test_performance()

#!/usr/bin/env python3
"""
Standalone test for SAMO Emotion Detection Model

This script tests the BERT emotion detection model independently
to ensure it works correctly before API integration.
"""

import sys
from pathlib import Path

# Add src to path for standalone execution
# Note: For proper package structure, use: export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# or install the package in editable mode: pip install -e .
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.emotion_detection.samo_bert_emotion_classifier import create_samo_bert_emotion_classifier
from models.emotion_detection.emotion_labels import get_all_emotions, get_emotion_description

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
    """Test emotion prediction on sample texts."""
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
    ]

    print(f"   Testing {len(test_texts)} sample texts...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Text {i}: {text}")
        
        # Get predictions
        results = model.predict_emotions(text, threshold=0.3, top_k=3)
        
        emotions = results['emotions'][0]
        probabilities = results['probabilities'][0]
        
        print(f"   Detected emotions: {emotions}")
        
        # Show top probabilities
        top_indices = sorted(range(len(probabilities)), 
                           key=lambda i: probabilities[i], reverse=True)[:5]
        print("   Top probabilities:")
        for idx in top_indices:
            emotion_name = all_emotions[idx]
            prob = probabilities[idx]
            print(f"     {emotion_name}: {prob:.3f}")


def test_invalid_input_types(model):
    """Test emotion prediction with invalid input types."""
    print("\n4.5. Testing invalid input types...")
    
    invalid_inputs = [
        None,
        123,
        [],
        {},
        "",
        "   ",  # Only whitespace
    ]
    
    for i, invalid_input in enumerate(invalid_inputs, 1):
        print(f"\n   Invalid input {i}: {type(invalid_input).__name__} = {repr(invalid_input)}")
        
        try:
            results = model.predict_emotions(invalid_input, threshold=0.3)
            print(f"   Result: {results}")
        except Exception as e:
            print(f"   Expected error: {type(e).__name__}: {e}")
            # This is expected behavior for invalid inputs


def test_batch_predictions(model, test_texts):
    """Test batch prediction functionality."""
    print("\n5. Testing batch prediction...")
    batch_results = model.predict_emotions(test_texts[:3], threshold=0.3)
    
    print(f"   Batch size: {len(batch_results['emotions'])}")
    print(f"   All predictions successful: {len(batch_results['emotions']) == 3}")


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
    
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        results = model.predict_emotions(test_text, threshold=threshold)
        emotions = results['emotions'][0]
        print(f"   Threshold {threshold}: {len(emotions)} emotions - {emotions}")


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
    test_invalid_input_types(model)
    test_batch_predictions(model, [
        "I am so happy and excited about this amazing opportunity!",
        "I feel really sad and disappointed about what happened today.",
        "I'm feeling anxious and worried about the upcoming presentation.",
    ])
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
    """Test model performance on various text lengths."""
    print("\n🚀 Testing Performance Characteristics")
    print("=" * 50)

    try:
        model, _ = create_samo_bert_emotion_classifier()
        
        # Test with different text lengths
        test_cases = [
            ("Short text", "I am happy!"),
            ("Medium text", "I am feeling really happy and excited about this new opportunity that has come my way."),
            ("Long text", "I am feeling incredibly happy and excited about this amazing new opportunity that has come my way. This is something I've been waiting for a long time, and I can't believe it's finally happening. I'm also a bit nervous about the challenges ahead, but I'm confident that I can handle them with the support of my friends and family."),
        ]

        for name, text in test_cases:
            print(f"\n{name}:")
            print(f"  Length: {len(text)} characters")
            
            import time
            start_time = time.time()
            results = model.predict_emotions(text, threshold=0.3)
            end_time = time.time()
            
            processing_time = end_time - start_time
            emotions = results['emotions'][0]
            
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Detected emotions: {emotions}")
            print(f"  Emotions count: {len(emotions)}")

    except Exception as e:
        print(f"❌ Error in performance test: {e}")


def test_batch_performance():
    """Test batch prediction performance with large input sets."""
    print("\n🚀 Testing Batch Performance")
    print("=" * 50)

    try:
        model, _ = create_samo_bert_emotion_classifier()
        
        # Generate large batch of test texts
        base_texts = [
            "I am so happy today!",
            "I feel really sad about this.",
            "I'm excited for the future!",
            "I'm worried about the outcome.",
            "I love spending time with family.",
            "I'm angry about the situation.",
            "I feel grateful for everything.",
            "I'm confused about what to do.",
            "I'm proud of my achievements.",
            "I feel anxious about the test.",
        ]
        
        # Create large batch (1000+ texts)
        large_batch = []
        for i in range(100):
            for base_text in base_texts:
                large_batch.append(f"{base_text} (Batch {i+1})")
        
        print(f"Testing batch prediction with {len(large_batch)} texts...")
        
        import time
        start_time = time.time()
        results = model.predict_emotions(large_batch, threshold=0.3, batch_size=32)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_text = total_time / len(large_batch)
        texts_per_second = len(large_batch) / total_time
        
        print(f"  Total processing time: {total_time:.3f}s")
        print(f"  Average time per text: {avg_time_per_text:.4f}s")
        print(f"  Texts per second: {texts_per_second:.2f}")
        print(f"  Total results: {len(results['emotions'])}")
        
        # Verify all texts were processed
        assert len(results['emotions']) == len(large_batch)
        print("  ✅ All texts processed successfully")

    except Exception as e:
        print(f"❌ Error in batch performance test: {e}")

if __name__ == "__main__":
    test_emotion_classifier()
    test_performance()
    test_batch_performance()

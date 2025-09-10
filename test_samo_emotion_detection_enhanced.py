#!/usr/bin/env python3
"""
Enhanced standalone test for SAMO Emotion Detection Model

This script tests the enhanced BERT emotion detection model with comprehensive
testing including edge cases, performance benchmarks, and error handling.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add src to path for standalone testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.emotion_detection.enhanced_bert_classifier import EnhancedBERTEmotionClassifier, EmotionPrediction
from models.emotion_detection.enhanced_config import create_enhanced_config_manager
from models.emotion_detection.emotion_labels import get_all_emotions, get_emotion_description

logger = logging.getLogger(__name__)


def test_model_initialization():
    """Test enhanced model initialization."""
    print("1. Initializing Enhanced SAMO BERT Emotion Classifier...")
    try:
        model = EnhancedBERTEmotionClassifier(
            model_name="bert-base-uncased",
            num_emotions=28,
            use_mixed_precision=True,
            cache_embeddings=True
        )
        print("✅ Enhanced classifier initialized successfully")
        return model
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return None


def test_model_info(model):
    """Test model information display."""
    print("\n2. Checking enhanced model information...")
    try:
        info = model.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Trainable parameters: {info['trainable_parameters']:,}")
        print(f"   Device: {info['device']}")
        print(f"   Mixed precision: {info['use_mixed_precision']}")
        print(f"   Max sequence length: {info['max_sequence_length']}")
        return info
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return None


def test_configuration_system():
    """Test enhanced configuration system."""
    print("\n3. Testing enhanced configuration system...")
    try:
        config_manager = create_enhanced_config_manager()
        config = config_manager.get_config()
        
        print(f"   Model name: {config.model.name}")
        print(f"   Num emotions: {config.emotion_detection.num_emotions}")
        print(f"   Prediction threshold: {config.emotion_detection.prediction_threshold}")
        print(f"   Mixed precision: {config.model.use_mixed_precision}")
        print("✅ Configuration system working")
        return config_manager
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None


def test_emotion_predictions(model):
    """Test emotion predictions with various inputs."""
    print("\n4. Testing emotion predictions...")
    
    test_texts = [
        "I am so happy and excited about this new opportunity!",
        "I feel really sad and disappointed about what happened.",
        "I'm feeling anxious and worried about the future.",
        "I'm grateful and thankful for all the support I've received.",
        "",  # Empty text
        "This is a very long text that should test the maximum sequence length handling and truncation capabilities of the model. " * 20,  # Very long text
    ]
    
    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            if not text.strip():
                print("   (Empty text - testing edge case)")
            
            prediction = model.predict_emotions(
                text, 
                top_k=3, 
                return_metadata=True
            )
            
            print(f"   Primary emotion: {prediction.primary_emotion}")
            print(f"   Confidence: {prediction.confidence:.3f}")
            print(f"   Intensity: {prediction.emotional_intensity}")
            print(f"   Top emotions: {prediction.top_k_emotions[:3]}")
            
            if prediction.prediction_metadata:
                print(f"   Text length: {prediction.prediction_metadata.get('text_length', 'N/A')}")
        
        print("✅ Emotion predictions working")
        return True
    except Exception as e:
        print(f"❌ Emotion prediction test failed: {e}")
        return False


def test_batch_predictions(model):
    """Test batch prediction capabilities."""
    print("\n5. Testing batch predictions...")
    
    batch_texts = [
        "I'm feeling great today!",
        "This is really frustrating.",
        "I'm so grateful for everything.",
        "I feel overwhelmed by all this work.",
        "I'm excited about the new project!"
    ]
    
    try:
        start_time = time.time()
        predictions = model.predict_emotions(
            batch_texts, 
            top_k=2, 
            return_metadata=True,
            batch_size=2
        )
        end_time = time.time()
        
        print(f"   Processed {len(batch_texts)} texts in {end_time - start_time:.3f}s")
        
        for i, prediction in enumerate(predictions):
            print(f"   Text {i+1}: {prediction.primary_emotion} ({prediction.confidence:.3f})")
        
        print("✅ Batch predictions working")
        return True
    except Exception as e:
        print(f"❌ Batch prediction test failed: {e}")
        return False


def test_performance_metrics(model):
    """Test performance tracking."""
    print("\n6. Testing performance metrics...")
    
    try:
        # Run some predictions to generate metrics
        test_texts = ["I'm happy", "I'm sad", "I'm excited"] * 5
        
        for text in test_texts:
            model.predict_emotions(text)
        
        metrics = model.get_performance_metrics()
        
        print(f"   Total inferences: {metrics['total_inferences']}")
        print(f"   Average inference time: {metrics['average_inference_time']:.3f}s")
        print(f"   Error count: {metrics['error_count']}")
        print(f"   Error rate: {metrics['error_rate']:.3f}")
        
        print("✅ Performance metrics working")
        return True
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False


def test_error_handling(model):
    """Test error handling capabilities."""
    print("\n7. Testing error handling...")
    
    try:
        # Test with None input
        try:
            model.predict_emotions(None)
        except Exception as e:
            print(f"   ✅ Handled None input: {type(e).__name__}")
        
        # Test with very long text
        very_long_text = "This is a test. " * 1000
        prediction = model.predict_emotions(very_long_text)
        print(f"   ✅ Handled very long text: {len(very_long_text)} chars")
        
        # Test with special characters
        special_text = "I'm feeling 😊🎉 excited! @#$%^&*()"
        prediction = model.predict_emotions(special_text)
        print(f"   ✅ Handled special characters: {prediction.primary_emotion}")
        
        print("✅ Error handling working")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_emotion_labels():
    """Test emotion labels functionality."""
    print("\n8. Testing emotion labels...")
    
    try:
        all_emotions = get_all_emotions()
        print(f"   Total emotions: {len(all_emotions)}")
        print(f"   Sample emotions: {all_emotions[:5]}")
        
        # Test emotion descriptions
        for emotion in all_emotions[:3]:
            description = get_emotion_description(emotion)
            print(f"   {emotion}: {description}")
        
        print("✅ Emotion labels working")
        return True
    except Exception as e:
        print(f"❌ Emotion labels test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧠 Testing Enhanced SAMO Emotion Detection Model")
    print("=" * 60)
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test model initialization
        model = test_model_initialization()
        if model is None:
            print("\n❌ Model initialization failed, cannot continue")
            return False
        
        # Test model information
        model_info = test_model_info(model)
        if model_info is None:
            print("\n❌ Model info test failed")
            return False
        
        # Test configuration system
        config_manager = test_configuration_system()
        if config_manager is None:
            print("\n❌ Configuration test failed")
            return False
        
        # Test emotion predictions
        if not test_emotion_predictions(model):
            print("\n❌ Emotion prediction test failed")
            return False
        
        # Test batch predictions
        if not test_batch_predictions(model):
            print("\n❌ Batch prediction test failed")
            return False
        
        # Test performance metrics
        if not test_performance_metrics(model):
            print("\n❌ Performance metrics test failed")
            return False
        
        # Test error handling
        if not test_error_handling(model):
            print("\n❌ Error handling test failed")
            return False
        
        # Test emotion labels
        if not test_emotion_labels():
            print("\n❌ Emotion labels test failed")
            return False
        
        print("\n🎉 All enhanced emotion detection tests passed!")
        print("✅ Enhanced model is ready for production use")
        
        return True
        
    except Exception as e:
        logger.exception("Test suite failed")
        print(f"\n❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

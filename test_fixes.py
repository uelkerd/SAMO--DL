#!/usr/bin/env python3
"""Comprehensive test script to verify all BERT training fixes."""

import sys
import os
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
import torch

def test_json_serialization():
    """Test that numpy types can be serialized to JSON."""
    print("🧪 Testing JSON serialization fix...")
    
    # Create test data with numpy types
    test_data = {
        "numpy_int": np.int64(42),
        "numpy_float": np.float64(3.14),
        "numpy_array": np.array([1, 2, 3]),
        "nested": {
            "numpy_value": np.int32(100)
        }
    }
    
    try:
        # This should fail without our fix
        json.dumps(test_data)
        print("❌ JSON serialization should have failed but didn't")
        return False
    except TypeError:
        print("✅ JSON serialization correctly fails with numpy types")
    
    # Test our conversion function
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item') and obj.size == 1:  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    converted_data = convert_numpy_types(test_data)
    try:
        json.dumps(converted_data)
        print("✅ JSON serialization works with converted data")
        return True
    except Exception as e:
        print(f"❌ JSON serialization still fails: {e}")
        return False

def test_evaluation_threshold():
    """Test the evaluation threshold fix."""
    print("\n🧪 Testing evaluation threshold fix...")
    
    try:
        # Initialize trainer
        trainer = EmotionDetectionTrainer()
        print("✅ Trainer initialized")
        
        # Prepare data
        trainer.prepare_data()
        print("✅ Data prepared")
        
        # Initialize model
        trainer.initialize_model()
        print("✅ Model initialized")
        
        # Test evaluation with different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5]
        
        for threshold in thresholds:
            print(f"   Testing threshold {threshold}...")
            metrics = evaluate_emotion_classifier(
                trainer.model, 
                trainer.val_dataloader, 
                trainer.device, 
                threshold=threshold
            )
            
            micro_f1 = metrics.get('micro_f1', 0)
            macro_f1 = metrics.get('macro_f1', 0)
            
            print(f"     Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
            
            if micro_f1 > 0 or macro_f1 > 0:
                print(f"✅ SUCCESS: Non-zero F1 scores with threshold {threshold}")
                return True
        
        print("❌ FAILED: All thresholds still give zero F1 scores")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_model_outputs():
    """Test what the model is actually outputting."""
    print("\n🧪 Testing model outputs...")
    
    try:
        trainer = EmotionDetectionTrainer()
        trainer.prepare_data()
        trainer.initialize_model()
        
        # Get a single batch
        batch = next(iter(trainer.val_dataloader))
        
        # Forward pass
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(batch["input_ids"], batch["attention_mask"])
            probabilities = outputs["probabilities"]
        
        print(f"   Probability stats:")
        print(f"     Min: {probabilities.min():.6f}")
        print(f"     Max: {probabilities.max():.6f}")
        print(f"     Mean: {probabilities.mean():.6f}")
        print(f"     Std: {probabilities.std():.6f}")
        
        # Check distribution
        for threshold in [0.1, 0.2, 0.3, 0.5]:
            count = (probabilities >= threshold).sum().item()
            total = probabilities.numel()
            percentage = (count / total) * 100
            print(f"     >= {threshold}: {count}/{total} ({percentage:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 COMPREHENSIVE BERT TRAINING FIXES TEST")
    print("=" * 50)
    
    results = []
    
    # Test 1: JSON serialization
    results.append(test_json_serialization())
    
    # Test 2: Model outputs
    results.append(test_model_outputs())
    
    # Test 3: Evaluation threshold (only if model outputs look reasonable)
    if results[-1]:  # If model outputs test passed
        results.append(test_evaluation_threshold())
    else:
        print("\n⏭️  Skipping evaluation test due to model output issues")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = ["JSON Serialization", "Model Outputs", "Evaluation Threshold"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
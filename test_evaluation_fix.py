#!/usr/bin/env python3
"""Test script to verify the evaluation threshold fix."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
import torch

def test_evaluation_fix():
    """Test the evaluation with the new threshold."""
    print("🔧 Testing evaluation threshold fix...")
    
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
        
        # Test evaluation with new threshold
        print("🧪 Testing evaluation with threshold=0.2...")
        metrics = evaluate_emotion_classifier(
            trainer.model, 
            trainer.val_dataloader, 
            trainer.device, 
            threshold=0.2
        )
        
        print(f"📊 New evaluation metrics:")
        print(f"   Micro F1: {metrics.get('micro_f1', 0):.4f}")
        print(f"   Macro F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"   Avg inference time: {metrics.get('avg_inference_time_ms', 0):.1f}ms")
        
        # Check if we got non-zero F1 scores
        if metrics.get('micro_f1', 0) > 0 or metrics.get('macro_f1', 0) > 0:
            print("🎉 SUCCESS: Evaluation threshold fix worked!")
            return True
        else:
            print("❌ FAILED: Still getting zero F1 scores")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_evaluation_fix()
    sys.exit(0 if success else 1) 
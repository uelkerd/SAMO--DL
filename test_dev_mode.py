#!/usr/bin/env python3
"""Test development mode for fast training."""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.emotion_detection.training_pipeline import train_emotion_detection_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_dev_mode():
    """Test development mode training."""
    print("🚀 Testing Development Mode Training...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Test with development mode enabled
        results = train_emotion_detection_model(
            batch_size=32,  # Larger batch size for dev mode
            num_epochs=1,   # Single epoch for testing
            output_dir="./test_checkpoints_dev",
            dev_mode=True   # Enable development mode
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ DEVELOPMENT MODE TEST COMPLETED!")
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"📊 Final test metrics: {results['final_test_metrics']}")
        print(f"🏆 Best validation score: {results['best_validation_score']:.4f}")
        
        # Check if training time is reasonable
        if training_time < 3600:  # Less than 1 hour
            print("✅ SUCCESS: Training completed in reasonable time!")
        else:
            print("⚠️  WARNING: Training still took too long")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_dev_mode()
    if success:
        print("\n🎉 Development mode is working correctly!")
    else:
        print("\n💥 Development mode needs more fixes!") 
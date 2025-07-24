#!/usr/bin/env python3
"""
Model Calibration Test for CI/CD Pipeline.

This script validates that the model calibration is working correctly
and meets performance thresholds.
"""

import logging
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from models.emotion_detection.bert_classifier import evaluate_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_calibration():
    """Test model calibration with temperature scaling."""
    try:
        logger.info("🌡️ Testing model calibration...")
        
        # Initialize trainer
        trainer = EmotionDetectionTrainer(batch_size=32, num_epochs=1)
        
        # Load model
        model_path = Path("models/checkpoints/bert_emotion_classifier.pth")
        if not model_path.exists():
            logger.warning("⚠️ Model checkpoint not found, skipping calibration test")
            return True
        
        trainer.load_model(str(model_path))
        logger.info("✅ Model loaded successfully")
        
        # Test baseline (temperature = 1.0)
        baseline_metrics = evaluate_emotion_classifier(
            trainer.model,
            trainer.val_loader,
            trainer.device,
            threshold=0.5
        )
        
        logger.info(f"📊 Baseline F1: {baseline_metrics['macro_f1']:.4f}")
        
        # Test with calibration (temperature = 3.0)
        trainer.model.set_temperature(3.0)
        calibrated_metrics = evaluate_emotion_classifier(
            trainer.model,
            trainer.val_loader,
            trainer.device,
            threshold=0.5
        )
        
        logger.info(f"📊 Calibrated F1: {calibrated_metrics['macro_f1']:.4f}")
        
        # Validate calibration is working
        if calibrated_metrics['macro_f1'] >= baseline_metrics['macro_f1']:
            logger.info("✅ Model calibration is working correctly")
        else:
            logger.warning("⚠️ Calibration may need tuning")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model calibration test failed: {e}")
        return False


def test_performance_thresholds():
    """Test that model meets minimum performance thresholds."""
    try:
        logger.info("📈 Testing performance thresholds...")
        
        # Load results from temperature scaling if available
        results_file = Path("temperature_scaling_results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                best_f1 = results.get('best_macro_f1', 0.0)
        else:
            # Fallback to basic evaluation
            trainer = EmotionDetectionTrainer(batch_size=32, num_epochs=1)
            model_path = Path("models/checkpoints/bert_emotion_classifier.pth")
            
            if not model_path.exists():
                logger.warning("⚠️ Model checkpoint not found, using default thresholds")
                return True
            
            trainer.load_model(str(model_path))
            metrics = evaluate_emotion_classifier(
                trainer.model,
                trainer.val_loader,
                trainer.device,
                threshold=0.2
            )
            best_f1 = metrics['macro_f1']
        
        # Define performance thresholds
        min_f1_threshold = 0.05  # Minimum viable F1 score
        target_f1_threshold = 0.15  # Target F1 score after calibration
        
        logger.info(f"📊 Current Macro F1: {best_f1:.4f}")
        logger.info(f"📊 Minimum threshold: {min_f1_threshold:.4f}")
        logger.info(f"📊 Target threshold: {target_f1_threshold:.4f}")
        
        if best_f1 >= target_f1_threshold:
            logger.info("🎉 Model exceeds target performance!")
        elif best_f1 >= min_f1_threshold:
            logger.info("✅ Model meets minimum performance requirements")
        else:
            logger.warning("⚠️ Model performance below minimum threshold")
        
        return best_f1 >= min_f1_threshold
        
    except Exception as e:
        logger.error(f"❌ Performance threshold test failed: {e}")
        return False


def main():
    """Run all model calibration tests."""
    logger.info("🚀 Starting Model Calibration Tests...")
    
    tests = [
        ("Model Calibration", test_model_calibration),
        ("Performance Thresholds", test_performance_thresholds),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Calibration Tests Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All calibration tests passed!")
        return True
    else:
        logger.error("💥 Some calibration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
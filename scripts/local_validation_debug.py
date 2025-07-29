#!/usr/bin/env python3
"""
Local Validation and Debug Script for SAMO Deep Learning.

This script performs targeted validation to identify the root cause of the 0.0000 loss issue.
It can be run locally to diagnose problems before deploying to GCP.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_environment():
    """Check basic environment setup."""
    logger.info("🔍 Checking environment...")
    
    try:
        import torch
        import transformers
        import numpy as np
        import pandas as pd
        
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ Transformers: {transformers.__version__}")
        logger.info(f"✅ NumPy: {np.__version__}")
        logger.info(f"✅ Pandas: {pd.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("✅ CPU mode available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        return False


def check_data_loading():
    """Check data loading functionality."""
    logger.info("🔍 Checking data loading...")
    
    try:
        from models.emotion_detection.dataset_loader import create_goemotions_loader
        
        # Create loader without dev_mode parameter
        logger.info("   Loading dataset...")
        loader = create_goemotions_loader()
        
        # Prepare datasets
        datasets = loader.prepare_datasets()
        
        # Check if we have the expected keys
        expected_keys = ["train", "validation", "test", "statistics", "class_weights"]
        for key in expected_keys:
            if key not in datasets:
                logger.error(f"❌ Missing key in datasets: {key}")
                return False
        
        logger.info(f"✅ Train set: {len(datasets['train'])} examples")
        logger.info(f"✅ Validation set: {len(datasets['validation'])} examples")
        logger.info(f"✅ Test set: {len(datasets['test'])} examples")
        
        # Check statistics
        stats = datasets["statistics"]
        logger.info(f"✅ Total examples: {stats.get('total_examples', 'N/A')}")
        logger.info(f"✅ Emotion distribution: {len(stats.get('emotion_counts', {}))} emotions")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False


def check_model_creation():
    """Check model creation and forward pass."""
    logger.info("🔍 Checking model creation...")
    
    try:
        from models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        import torch
        
        # Create model
        model, loss_fn = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=6,
        )
        
        logger.info(f"✅ Model created: {model.count_parameters():,} parameters")
        logger.info(f"✅ Loss function: {type(loss_fn).__name__}")
        
        # Test forward pass
        batch_size = 2
        seq_length = 64
        num_classes = 28
        
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        dummy_attention_mask = torch.ones(batch_size, seq_length)
        dummy_labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        # Ensure some positive labels
        dummy_labels[:, 0] = 1.0
        
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input_ids, dummy_attention_mask)
            loss = loss_fn(logits, dummy_labels)
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Loss value: {loss.item():.8f}")
        
        # Check for issues
        if loss.item() <= 0:
            logger.error(f"❌ CRITICAL: Loss is zero or negative: {loss.item()}")
            return False
        
        if torch.isnan(loss).any():
            logger.error("❌ CRITICAL: NaN loss!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return False


def check_loss_function():
    """Check loss function implementation."""
    logger.info("🔍 Checking loss function...")
    
    try:
        from models.emotion_detection.bert_classifier import WeightedBCELoss
        import torch
        import torch.nn.functional as F
        
        # Test different scenarios
        batch_size = 4
        num_classes = 28
        
        # Scenario 1: Mixed labels
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        labels[:, 0] = 1.0  # Ensure some positive labels
        
        loss_fn = WeightedBCELoss()
        loss1 = loss_fn(logits, labels)
        
        # Compare with manual BCE
        bce_manual = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        
        logger.info(f"✅ Mixed labels loss: {loss1.item():.8f}")
        logger.info(f"✅ All positive loss: {loss_fn(logits, torch.ones(batch_size, num_classes)).item():.8f}")
        logger.info(f"✅ All negative loss: {loss_fn(logits, torch.zeros(batch_size, num_classes)).item():.8f}")
        logger.info(f"✅ Manual BCE loss: {bce_manual.item():.8f}")
        
        # Check for issues
        if loss1.item() <= 0:
            logger.error("❌ CRITICAL: Loss function producing zero/negative values!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Loss function check failed: {e}")
        return False


def check_data_distribution():
    """Check data distribution to identify 0.0000 loss causes."""
    logger.info("🔍 Checking data distribution...")
    
    try:
        from models.emotion_detection.dataset_loader import create_goemotions_loader
        
        # Load data
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()
        
        # Get training data
        train_dataset = datasets["train"]
        
        # Analyze first few examples
        total_samples = min(100, len(train_dataset))
        total_positive_labels = 0
        label_distribution = {}
        
        for i in range(total_samples):
            example = train_dataset[i]
            labels = example["labels"]
            
            # Count positive labels
            positive_count = sum(labels)
            total_positive_labels += positive_count
            
            # Check per-class distribution
            for class_idx, label in enumerate(labels):
                if class_idx not in label_distribution:
                    label_distribution[class_idx] = 0
                if label == 1:
                    label_distribution[class_idx] += 1
        
        # Calculate statistics
        total_possible_labels = total_samples * 28  # 28 emotion classes
        positive_rate = total_positive_labels / total_possible_labels
        
        logger.info(f"✅ Total samples analyzed: {total_samples}")
        logger.info(f"✅ Total positive labels: {total_positive_labels}")
        logger.info(f"✅ Positive label rate: {positive_rate:.6f}")
        
        # Check for critical issues
        if positive_rate == 0:
            logger.error("❌ CRITICAL: No positive labels found!")
            logger.error("   This will cause 0.0000 loss with BCE")
            return False
        elif positive_rate == 1:
            logger.error("❌ CRITICAL: All labels are positive!")
            logger.error("   This will cause 0.0000 loss with BCE")
            return False
        elif positive_rate < 0.01:
            logger.warning("⚠️  Very low positive label rate")
            logger.warning("   Consider using focal loss or class weights")
        
        # Log class distribution
        logger.info("📊 Class distribution (first 10 classes):")
        for class_idx in range(min(10, len(label_distribution))):
            count = label_distribution.get(class_idx, 0)
            if count > 0:
                logger.info(f"   Class {class_idx}: {count} positive samples")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data distribution check failed: {e}")
        return False


def main():
    """Main function to run all validations."""
    logger.info("🚀 SAMO-DL Local Validation and Debug")
    logger.info("=" * 50)
    
    # Run all validations
    validations = [
        ("Environment", check_environment),
        ("Data Loading", check_data_loading),
        ("Model Creation", check_model_creation),
        ("Loss Function", check_loss_function),
        ("Data Distribution", check_data_distribution),
    ]
    
    # Run validations
    results = {}
    for name, validation_func in validations:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*40}")
        
        try:
            success = validation_func()
            results[name] = success
            
            if success:
                logger.info(f"✅ {name} PASSED")
            else:
                logger.error(f"❌ {name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {name} ERROR: {e}")
            results[name] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\n{'='*50}")
    logger.info("📊 VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total checks: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    
    if passed == total:
        logger.info("\n✅ ALL VALIDATIONS PASSED!")
        logger.info("   The 0.0000 loss issue is likely due to:")
        logger.info("   1. Learning rate too high (try 2e-6 instead of 2e-5)")
        logger.info("   2. Need focal loss for class imbalance")
        logger.info("   3. Need class weights for imbalanced data")
        logger.info("   Ready for training with optimized configuration!")
    else:
        logger.error("\n❌ SOME CHECKS FAILED!")
        logger.error("   Fix the issues above before proceeding")
        logger.error("   This will prevent the 0.0000 loss problem")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
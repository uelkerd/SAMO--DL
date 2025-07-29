#!/usr/bin/env python3
"""
Debug Training Loss Script for SAMO Deep Learning.

This script investigates why training loss is showing 0.0000 by examining:
- Data loading and label distribution
- Model outputs and predictions
- Loss function calculation
- Numerical precision issues
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_data_loading():
    """Debug data loading and label distribution."""
    logger.info("🔍 Debugging data loading...")
    
    try:
        from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
        
        # Initialize trainer
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            batch_size=4,  # Small batch for debugging
            num_epochs=1,
            dev_mode=True
        )
        
        # Prepare data
        datasets = trainer.prepare_data(dev_mode=True)
        
        # Check data statistics
        train_dataloader = datasets["train_dataloader"]
        val_dataloader = datasets["val_dataloader"]
        
        logger.info(f"✅ Train dataloader: {len(train_dataloader)} batches")
        logger.info(f"✅ Val dataloader: {len(val_dataloader)} batches")
        
        # Examine first batch
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= 2:  # Only check first 2 batches
                break
                
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            logger.info(f"📊 Batch {batch_idx + 1} Statistics:")
            logger.info(f"   Input shape: {input_ids.shape}")
            logger.info(f"   Labels shape: {labels.shape}")
            logger.info(f"   Labels dtype: {labels.dtype}")
            logger.info(f"   Labels min: {labels.min().item()}")
            logger.info(f"   Labels max: {labels.max().item()}")
            logger.info(f"   Labels mean: {labels.float().mean().item():.4f}")
            logger.info(f"   Non-zero labels: {(labels > 0).sum().item()}")
            logger.info(f"   Total labels: {labels.numel()}")
            
            # Check for all-zero or all-one labels
            if labels.sum() == 0:
                logger.error("❌ WARNING: All labels are zero!")
            elif labels.sum() == labels.numel():
                logger.error("❌ WARNING: All labels are one!")
            
            # Check label distribution per class
            for i in range(labels.shape[1]):
                class_count = labels[:, i].sum().item()
                if class_count > 0:
                    logger.info(f"   Class {i}: {class_count} positive samples")
        
        return datasets
        
    except Exception as e:
        logger.error(f"❌ Error in data loading: {e}")
        return None


def debug_model_outputs(datasets):
    """Debug model outputs and predictions."""
    logger.info("🔍 Debugging model outputs...")
    
    try:
        from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
        
        # Initialize trainer and model
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            batch_size=4,
            num_epochs=1,
            dev_mode=True
        )
        
        # Initialize model
        trainer.prepare_data(dev_mode=True)
        trainer.initialize_model()
        
        # Get first batch
        batch = next(iter(trainer.train_dataloader))
        input_ids = batch["input_ids"].to(trainer.device)
        attention_mask = batch["attention_mask"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        
        # Forward pass
        trainer.model.eval()
        with torch.no_grad():
            logits = trainer.model(input_ids, attention_mask)
            predictions = torch.sigmoid(logits)
        
        logger.info(f"📊 Model Output Statistics:")
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Predictions shape: {predictions.shape}")
        logger.info(f"   Logits min: {logits.min().item():.6f}")
        logger.info(f"   Logits max: {logits.max().item():.6f}")
        logger.info(f"   Logits mean: {logits.mean().item():.6f}")
        logger.info(f"   Logits std: {logits.std().item():.6f}")
        logger.info(f"   Predictions min: {predictions.min().item():.6f}")
        logger.info(f"   Predictions max: {predictions.max().item():.6f}")
        logger.info(f"   Predictions mean: {predictions.mean().item():.6f}")
        
        # Check for extreme values
        if torch.isnan(logits).any():
            logger.error("❌ WARNING: NaN values in logits!")
        if torch.isinf(logits).any():
            logger.error("❌ WARNING: Inf values in logits!")
        
        return logits, predictions, labels
        
    except Exception as e:
        logger.error(f"❌ Error in model outputs: {e}")
        return None, None, None


def debug_loss_calculation(logits, predictions, labels):
    """Debug loss function calculation."""
    logger.info("🔍 Debugging loss calculation...")
    
    try:
        from models.emotion_detection.bert_classifier import WeightedBCELoss
        
        # Test different loss configurations
        logger.info("📊 Testing different loss configurations:")
        
        # 1. Standard BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        loss_standard = bce_loss(logits, labels.float())
        logger.info(f"   Standard BCE Loss: {loss_standard.item():.6f}")
        
        # 2. Manual BCE calculation
        bce_manual = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction='none'
        )
        loss_manual = bce_manual.mean()
        logger.info(f"   Manual BCE Loss: {loss_manual.item():.6f}")
        
        # 3. Weighted BCE loss (no weights)
        weighted_bce = WeightedBCELoss(class_weights=None)
        loss_weighted = weighted_bce(logits, labels)
        logger.info(f"   Weighted BCE Loss (no weights): {loss_weighted.item():.6f}")
        
        # 4. Check individual components
        logger.info("📊 Individual loss components:")
        for i in range(min(5, logits.shape[1])):  # Check first 5 classes
            class_logits = logits[:, i]
            class_labels = labels[:, i].float()
            class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                class_logits, class_labels, reduction='mean'
            )
            logger.info(f"   Class {i}: loss={class_loss.item():.6f}, "
                       f"labels_sum={class_labels.sum().item()}, "
                       f"logits_mean={class_logits.mean().item():.6f}")
        
        # 5. Check for numerical precision issues
        logger.info("📊 Numerical precision check:")
        logger.info(f"   Logits precision: {logits.dtype}")
        logger.info(f"   Labels precision: {labels.dtype}")
        logger.info(f"   Device: {logits.device}")
        
        # 6. Test with small epsilon
        epsilon = 1e-8
        logits_eps = logits + epsilon
        loss_eps = bce_loss(logits_eps, labels.float())
        logger.info(f"   BCE Loss (with epsilon): {loss_eps.item():.6f}")
        
        return {
            'standard_bce': loss_standard.item(),
            'manual_bce': loss_manual.item(),
            'weighted_bce': loss_weighted.item(),
            'with_epsilon': loss_eps.item()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in loss calculation: {e}")
        return None


def debug_class_weights():
    """Debug class weights calculation."""
    logger.info("🔍 Debugging class weights...")
    
    try:
        from models.emotion_detection.dataset_loader import create_goemotions_loader
        
        # Get class weights
        datasets = create_goemotions_loader(dev_mode=True)
        class_weights = datasets.get('class_weights')
        
        if class_weights is not None:
            logger.info(f"📊 Class Weights Statistics:")
            logger.info(f"   Shape: {class_weights.shape}")
            logger.info(f"   Min: {class_weights.min():.6f}")
            logger.info(f"   Max: {class_weights.max():.6f}")
            logger.info(f"   Mean: {class_weights.mean():.6f}")
            logger.info(f"   Std: {class_weights.std():.6f}")
            
            # Check for extreme values
            if class_weights.min() <= 0:
                logger.error("❌ WARNING: Class weights contain zero or negative values!")
            if class_weights.max() > 100:
                logger.error("❌ WARNING: Class weights contain very large values!")
            
            # Show first 10 class weights
            logger.info("📊 First 10 class weights:")
            for i in range(min(10, len(class_weights))):
                logger.info(f"   Class {i}: {class_weights[i]:.6f}")
        
        return class_weights
        
    except Exception as e:
        logger.error(f"❌ Error in class weights: {e}")
        return None


def main():
    """Main debugging function."""
    logger.info("🚀 Starting training loss debugging...")
    
    # Debug data loading
    datasets = debug_data_loading()
    if datasets is None:
        logger.error("❌ Failed to load data, stopping debug")
        return
    
    # Debug class weights
    class_weights = debug_class_weights()
    
    # Debug model outputs
    logits, predictions, labels = debug_model_outputs(datasets)
    if logits is None:
        logger.error("❌ Failed to get model outputs, stopping debug")
        return
    
    # Debug loss calculation
    loss_results = debug_loss_calculation(logits, predictions, labels)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📋 DEBUG SUMMARY")
    logger.info("="*60)
    
    if loss_results:
        logger.info("Loss Results:")
        for loss_type, value in loss_results.items():
            logger.info(f"   {loss_type}: {value:.6f}")
            
        # Check if all losses are zero
        all_zero = all(abs(v) < 1e-10 for v in loss_results.values())
        if all_zero:
            logger.error("❌ CRITICAL: All loss values are effectively zero!")
            logger.error("   This indicates a serious issue with the training setup.")
        else:
            logger.info("✅ Loss values are non-zero, training should work correctly.")
    
    logger.info("🔍 Debugging complete!")


if __name__ == "__main__":
    main() 
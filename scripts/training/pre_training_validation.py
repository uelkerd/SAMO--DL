#!/usr/bin/env python3
"""
Pre-training Validation Script

This script validates the training setup before starting actual training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import create_goemotions_loader
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from torch.optim import AdamW
import pandas as pd
import shutil
import torch
import transformers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_training_setup():
    """Validate the training setup before starting actual training."""
    try:
        logger.info("🔍 Starting pre-training validation...")
        
        # Test data loading
        logger.info("📊 Testing data loading...")
        data_loader = create_goemotions_loader()
        datasets = data_loader.load_data()
        
        # Validate first batch
        train_loader = torch.utils.data.DataLoader(datasets["train"], batch_size=4, shuffle=True)
        batch = next(iter(train_loader))
        logger.info("✅ Data loading successful - batch shape: %s", batch[0].shape)
        
        # Validate labels
        logger.info("🏷️  Testing label encoding...")
        unique_labels = set(datasets["train"].labels)
        logger.info("✅ Found %s unique labels", len(unique_labels))
        
        # Validate outputs
        logger.info("🧠 Testing model creation...")
        model, loss_function = create_bert_emotion_classifier()
        logger.info("✅ Model creation successful")
        
        # Test learning rate
        logger.info("⚙️  Testing optimizer...")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        logger.info("✅ Optimizer creation successful")
        
        # Test loss function
        logger.info("📉 Testing loss function...")
        criterion = torch.nn.CrossEntropyLoss()
        logger.info("✅ Loss function creation successful")
        
        # Test one training step
        logger.info("🚀 Testing one training step...")
        model.train()
        outputs = model(batch[0])
        loss = criterion(outputs, batch[1])
        loss.backward()
        optimizer.step()
        logger.info("✅ Training step successful - loss: %.4f", loss.item())
        
        # Test scheduler
        logger.info("📈 Testing scheduler...")
        _scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        logger.info("✅ Scheduler creation successful")
        
        logger.info("🎉 All validation tests passed! Training setup is ready.")
        return True
        
    except Exception as e:
        logger.error("❌ Validation failed: %s", e)
        return False


def main():
    """Main function to run validation."""
    logger.info("Starting pre-training validation...")
    
    if validate_training_setup():
        logger.info("✅ Validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
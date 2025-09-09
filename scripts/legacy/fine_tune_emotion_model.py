#!/usr/bin/env python3
"""
Fine-tune Emotion Model Script

This script fine-tunes the emotion detection model.
"""

# Standard library imports
import logging
import sys
import traceback
from pathlib import Path

# Third-party imports
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Local imports
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import create_goemotions_loader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# Training loop
def train_model():
    """Train the emotion detection model."""
    try:
        logger.info("🚀 Starting model fine-tuning...")
        
        # Load dataset
        data_loader = create_goemotions_loader()
        datasets = data_loader.load_data()
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(datasets["train"], batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(datasets["val"], batch_size=16, shuffle=False)
        
        # Create model
        model, _ = create_bert_emotion_classifier()
        model.to(device)
        
        # Setup loss and optimizer
        criterion = CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        # Training phase
        model.train()
        for epoch in range(3):
            logger.info("Epoch %s/%s", epoch + 1, 3)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Log progress every 100 batches
                if batch_idx % 100 == 0:
                    logger.info("Batch %s, Loss: %.4f", batch_idx, loss.item())
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
            
            val_loss /= len(val_loader)
            logger.info("Validation Loss: %.4f", val_loss)
            
            # Update learning rate
            if epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
        
        # Save model
        torch.save(model.state_dict(), "fine_tuned_model.pth")
        logger.info("✅ Model fine-tuning completed!")
        
    except Exception as e:
        logger.error("❌ Training failed: %s", e)
        traceback.print_exc()


def main():
    """Main function."""
    train_model()


if __name__ == "__main__":
    main()
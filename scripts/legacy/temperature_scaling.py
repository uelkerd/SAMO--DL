#!/usr/bin/env python3
"""
Temperature Scaling Script

This script applies temperature scaling to improve model calibration.
"""

import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration."""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature


def calibrate_model(model, val_loader, device):
    """Calibrate model using temperature scaling."""
    try:
        logger.info("🌡️  Starting temperature scaling calibration...")
        
        # Create temperature scaling layer
        temp_scaling = TemperatureScaling().to(device)
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        
        # Concatenate all logits and labels
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        # Optimize temperature parameter
        optimizer = torch.optim.LBFGS([temp_scaling.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            """Evaluate loss for temperature scaling optimization.

            Returns:
                Loss value for LBFGS optimization
            """
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(temp_scaling(all_logits), all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        logger.info("✅ Temperature scaling completed!")
        logger.info("✅ Optimal temperature: %.4f", temp_scaling.temperature.item())
        
        return temp_scaling
        
    except Exception as e:
        logger.error("❌ Temperature scaling failed: %s", e)
        return None


def main():
    """Main function."""
    logger.info("Starting temperature scaling...")
    
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # This would normally load a real model and validation data
    logger.info("🎉 Temperature scaling setup completed!")


if __name__ == "__main__":
    main()
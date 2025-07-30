import os

#!/usr/bin/env python3
"""
Simple Test Script for Loss Debugging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_bce_loss():
    """Test BCE loss with different scenarios."""
    print("🧪 Testing BCE Loss Scenarios...")

    # Scenario 1: Normal case
    logits = torch.randn(4, 28)  # 4 samples, 28 classes
    labels = torch.randint(0, 2, (4, 28)).float()  # Random binary labels

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    print("Normal case - Loss: {loss.item():.6f}")

    # Scenario 2: All zeros
    logits = torch.randn(4, 28)
    labels = torch.zeros(4, 28)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    print("All zeros - Loss: {loss.item():.6f}")

    # Scenario 3: All ones
    logits = torch.randn(4, 28)
    labels = torch.ones(4, 28)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    print("All ones - Loss: {loss.item():.6f}")

    # Scenario 4: Perfect predictions
    logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]] * 4)  # Strong predictions
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)  # Perfect targets

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    print("Perfect predictions - Loss: {loss.item():.6f}")

    # Scenario 5: Very small logits
    logits = torch.tensor([[0.001, -0.001, 0.001, -0.001]] * 4)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    print("Small logits - Loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_bce_loss()

# Scenario 1: Normal case
# Scenario 2: All zeros
# Scenario 3: All ones
# Scenario 4: Perfect predictions
# Scenario 5: Very small logits
#!/usr/bin/env python3
import logging

import torch
import torch.nn.functional as F

"""
Simple Test Script for Loss Debugging
"""


def test_bce_loss():
    """Test BCE loss with different scenarios."""
    logging.info("🧪 Testing BCE Loss Scenarios...")

    logits = torch.randn(4, 28)  # 4 samples, 28 classes
    labels = torch.randint(0, 2, (4, 28)).float()  # Random binary labels

    F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Normal case - Loss: {loss.item():.6f}")

    logits = torch.randn(4, 28)
    labels = torch.zeros(4, 28)

    F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("All zeros - Loss: {loss.item():.6f}")

    logits = torch.randn(4, 28)
    labels = torch.ones(4, 28)

    F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("All ones - Loss: {loss.item():.6f}")

    logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]] * 4)  # Strong predictions
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)  # Perfect targets

    F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Perfect predictions - Loss: {loss.item():.6f}")

    logits = torch.tensor([[0.001, -0.001, 0.001, -0.001]] * 4)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)

    F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Small logits - Loss: {loss.item():.6f}")


if __name__ == "__main__":
    test_bce_loss()

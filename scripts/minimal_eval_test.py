import torch
import logging


#!/usr/bin/env python3
"""
Minimal test of evaluation logic to isolate the bug.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_logic():
    """Test the evaluation logic with synthetic data."""

    # Create synthetic data matching what we observed
    batch_size = 128  # From our debug output
    num_emotions = 28
    threshold = 0.2

    # Create probabilities similar to what we observed
    # min: 0.1150, max: 0.9119, mean: 0.4681
    torch.manual_seed(42)
    probabilities = torch.rand(batch_size, num_emotions) * 0.8 + 0.1

    print("🔍 Testing evaluation logic:")
    print("  Probabilities shape: {probabilities.shape}")
    print(
        "  Probabilities min/max/mean: {probabilities.min():.4f}/{probabilities.max():.4f}/{probabilities.mean():.4f}"
    )

    # Count how many should be above threshold
    expected_above_threshold = (probabilities >= threshold).sum().item()
    total_positions = batch_size * num_emotions

    print(
        "  Expected above threshold: {expected_above_threshold}/{total_positions} ({100*expected_above_threshold/total_positions:.1f}%)"
    )

    # Apply threshold (this is the exact line from our evaluation function)
    predictions = (probabilities >= threshold).float()

    print("  Predictions after threshold:")
    print("    - Sum: {predictions.sum().item()}")
    print("    - Mean: {predictions.mean().item():.4f}")
    print(
        "    - Match expected: {'✅' if predictions.sum().item() == expected_above_threshold else '❌'}"
    )

    # Check fallback logic
    samples_with_zero = (predictions.sum(dim=1) == 0).sum().item()
    batch_has_no_predictions = samples_with_zero > 0

    print("  Fallback check:")
    print("    - Samples with zero predictions: {samples_with_zero}")
    print("    - Needs fallback: {batch_has_no_predictions}")

    if batch_has_no_predictions:
        print("  Applying fallback...")
        samples_before = (predictions.sum(dim=1) == 0).sum().item()

        for sample_idx in range(predictions.shape[0]):
            if predictions[sample_idx].sum() == 0:
                top_idx = torch.topk(probabilities[sample_idx], k=1, dim=0)[1]
                predictions[sample_idx, top_idx] = 1.0

        samples_after = (predictions.sum(dim=1) == 0).sum().item()
        print("    - Applied fallback to {samples_before - samples_after} samples")
        print("    - Final predictions sum: {predictions.sum().item()}")
        print("    - Final predictions mean: {predictions.mean().item():.4f}")

    return predictions


if __name__ == "__main__":
    test_evaluation_logic()

#!/usr/bin/env python3
"""
Simple test to isolate the threshold application bug.
"""

import torch


def test_threshold_application():
    """Test threshold application with synthetic data."""

    print("🔍 Testing threshold application with synthetic data")

    # Create synthetic probability data that matches what we saw in debug output
    batch_size = 434
    num_emotions = 28

    # Create probabilities with similar distribution to what we observed
    # mean=0.4681, min=0.1150, max=0.9119
    torch.manual_seed(42)  # For reproducibility
    probabilities = torch.rand(batch_size, num_emotions) * 0.8 + 0.1  # Range 0.1 to 0.9

    print("📊 Synthetic probabilities:")
    print(f"  - Shape: {probabilities.shape}")
    print(f"  - Min: {probabilities.min():.4f}")
    print(f"  - Max: {probabilities.max():.4f}")
    print(f"  - Mean: {probabilities.mean():.4f}")

    # Test threshold application
    threshold = 0.2
    print(f"\n🎯 Applying threshold: {threshold}")

    # Count probabilities above threshold
    above_threshold = probabilities >= threshold
    num_above_threshold = above_threshold.sum().item()
    total_positions = batch_size * num_emotions

    print("📊 Threshold analysis:")
    print(f"  - Total positions: {total_positions}")
    print(f"  - Positions >= {threshold}: {num_above_threshold}")
    print(f"  - Percentage >= {threshold}: {100 * num_above_threshold / total_positions:.1f}%")

    # Apply threshold to get predictions
    predictions = (probabilities >= threshold).float()

    print("📊 Predictions after threshold:")
    print(f"  - Shape: {predictions.shape}")
    print(f"  - Sum: {predictions.sum().item()}")
    print(f"  - Mean: {predictions.mean().item():.4f}")
    print(f"  - Expected sum: {num_above_threshold}")
    print(f"  - Match: {'✅' if predictions.sum().item() == num_above_threshold else '❌'}")

    # Check for samples with no predictions
    samples_with_no_predictions = (predictions.sum(dim=1) == 0).sum().item()
    print(f"  - Samples with 0 predictions: {samples_with_no_predictions}")

    # Apply fallback logic
    if samples_with_no_predictions > 0:
        print(f"\n🔧 Applying fallback to {samples_with_no_predictions} samples...")

        predictions_with_fallback = predictions.clone()
        for sample_idx in range(predictions.shape[0]):
            if predictions[sample_idx].sum() == 0:
                top_idx = torch.topk(probabilities[sample_idx], k=1, dim=0)[1]
                predictions_with_fallback[sample_idx, top_idx] = 1.0

        print("📊 Predictions after fallback:")
        print(f"  - Sum: {predictions_with_fallback.sum().item()}")
        print(f"  - Mean: {predictions_with_fallback.mean().item():.4f}")
        print(
            f"  - Samples with 0 predictions: {(predictions_with_fallback.sum(dim=1) == 0).sum().item()}"
        )

    return predictions


if __name__ == "__main__":
    test_threshold_application()

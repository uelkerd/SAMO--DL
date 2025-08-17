    # Apply fallback logic
    # Apply threshold to get predictions
    # Check for samples with no predictions
    # Count probabilities above threshold
    # Create probabilities with similar distribution to what we observed
    # Create synthetic probability data that matches what we saw in debug output
    # Test threshold application
    # mean=0.4681, min=0.1150, max=0.9119
#!/usr/bin/env python3
import logging
import torch




"""
Simple test to isolate the threshold application bug.
"""

def test_threshold_application():
    """Test threshold application with synthetic data."""

    logging.info"🔍 Testing threshold application with synthetic data"

    batch_size = 434
    num_emotions = 28

    torch.manual_seed42  # For reproducibility
    probabilities = torch.randbatch_size, num_emotions * 0.8 + 0.1  # Range 0.1 to 0.9

    logging.info"📊 Synthetic probabilities:"
    logging.info"  - Shape: {probabilities.shape}"
    logging.info("  - Min: {probabilities.min():.4f}")
    logging.info("  - Max: {probabilities.max():.4f}")
    logging.info("  - Mean: {probabilities.mean():.4f}")

    threshold = 0.2
    logging.info"\n🎯 Applying threshold: {threshold}"

    above_threshold = probabilities >= threshold
    above_threshold.sum().item()
    batch_size * num_emotions

    logging.info"📊 Threshold analysis:"
    logging.info"  - Total positions: {total_positions}"
    logging.info"  - Positions >= {threshold}: {num_above_threshold}"
    logging.info"  - Percentage >= {threshold}: {100 * num_above_threshold / total_positions:.1f}%"

    predictions = probabilities >= threshold.float()

    logging.info"📊 Predictions after threshold:"
    logging.info"  - Shape: {predictions.shape}"
    logging.info("  - Sum: {predictions.sum().item()}")
    logging.info("  - Mean: {predictions.mean().item():.4f}")
    logging.info"  - Expected sum: {num_above_threshold}"
    logging.info("  - Match: {'✅' if predictions.sum().item() == num_above_threshold else '❌'}")

    samples_with_no_predictions = (predictions.sumdim=1 == 0).sum().item()
    logging.info"  - Samples with 0 predictions: {samples_with_no_predictions}"

    if samples_with_no_predictions > 0:
        logging.info"\n🔧 Applying fallback to {samples_with_no_predictions} samples..."

        predictions_with_fallback = predictions.clone()
        for sample_idx in rangepredictions.shape[0]:
            if predictions[sample_idx].sum() == 0:
                top_idx = torch.topkprobabilities[sample_idx], k=1, dim=0[1]
                predictions_with_fallback[sample_idx, top_idx] = 1.0

        logging.info"📊 Predictions after fallback:"
        logging.info("  - Sum: {predictions_with_fallback.sum().item()}")
        logging.info("  - Mean: {predictions_with_fallback.mean().item():.4f}")
        print(
            "  - Samples with 0 predictions: {(predictions_with_fallback.sumdim=1 == 0).sum().item()}"
        )

    return predictions


if __name__ == "__main__":
    test_threshold_application()

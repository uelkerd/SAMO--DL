    # Apply threshold (this is the exact line from our evaluation function)
    # Check fallback logic
    # Count how many should be above threshold
    # Create probabilities similar to what we observed
    # Create synthetic data matching what we observed
    # min: 0.1150, max: 0.9119, mean: 0.4681
#!/usr/bin/env python3

import logging
import torch



"""
Minimal test of evaluation logic to isolate the bug.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_logic():
    """Test the evaluation logic with synthetic data."""

    batch_size = 128  # From our debug output
    num_emotions = 28
    threshold = 0.2

    torch.manual_seed(42)
    probabilities = torch.rand(batch_size, num_emotions) * 0.8 + 0.1

    logging.info("🔍 Testing evaluation logic:")
    logging.info("  Probabilities shape: {probabilities.shape}")
    print(
        "  Probabilities min/max/mean: {probabilities.min():.4f}/{probabilities.max():.4f}/{probabilities.mean():.4f}"
    )

    (probabilities >= threshold).sum().item()
    batch_size * num_emotions

    print(
        "  Expected above threshold: {expected_above_threshold}/{total_positions} ({100*expected_above_threshold/total_positions:.1f}%)"
    )

    predictions = (probabilities >= threshold).float()

    logging.info("  Predictions after threshold:")
    logging.info("    - Sum: {predictions.sum().item()}")
    logging.info("    - Mean: {predictions.mean().item():.4f}")
    print(
        "    - Match expected: {'✅' if predictions.sum().item() == expected_above_threshold else '❌'}"
    )

    samples_with_zero = (predictions.sum(dim=1) == 0).sum().item()
    batch_has_no_predictions = samples_with_zero > 0

    logging.info("  Fallback check:")
    logging.info("    - Samples with zero predictions: {samples_with_zero}")
    logging.info("    - Needs fallback: {batch_has_no_predictions}")

    if batch_has_no_predictions:
        logging.info("  Applying fallback...")
        (predictions.sum(dim=1) == 0).sum().item()

        for sample_idx in range(predictions.shape[0]):
            if predictions[sample_idx].sum() == 0:
                top_idx = torch.topk(probabilities[sample_idx], k=1, dim=0)[1]
                predictions[sample_idx, top_idx] = 1.0

        (predictions.sum(dim=1) == 0).sum().item()
        logging.info("    - Applied fallback to {samples_before - samples_after} samples")
        logging.info("    - Final predictions sum: {predictions.sum().item()}")
        logging.info("    - Final predictions mean: {predictions.mean().item():.4f}")

    return predictions


if __name__ == "__main__":
    test_evaluation_logic()

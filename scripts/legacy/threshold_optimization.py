#!/usr/bin/env python3
"""
Threshold Optimization Script

This script optimizes classification thresholds for better performance.
"""

import sys
from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def optimize_thresholds(y_true, y_scores):
    """Optimize classification thresholds for better F1 score."""
    try:
        logger.info("🔍 Starting threshold optimization...")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        logger.info("✅ Optimal threshold: %.4f", optimal_threshold)
        logger.info("✅ Optimal F1 score: %.4f", optimal_f1)
        
        return optimal_threshold, optimal_f1
        
    except Exception as e:
        logger.error("❌ Threshold optimization failed: %s", e)
        return None, None


def main():
    """Main function."""
    # Example usage
    logger.info("Starting threshold optimization...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.random(1000)
    
    threshold, f1 = optimize_thresholds(y_true, y_scores)
    
    if threshold is not None:
        logger.info("🎉 Threshold optimization completed!")
    else:
        logger.error("💥 Threshold optimization failed!")


if __name__ == "__main__":
    main()
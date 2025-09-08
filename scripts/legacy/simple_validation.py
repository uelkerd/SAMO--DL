#!/usr/bin/env python3
"""Simple Validation Script for SAMO Emotion Detection Model"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    import sklearn
    import transformers
except ImportError as e:
    logger.error("Missing required dependencies: %s", e)
    logger.info("Please install: pip install torch scikit-learn transformers")
    sys.exit(1)


def validate_environment() -> bool:
    """Validate that all required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    try:
        # Test PyTorch
        logger.info("Testing PyTorch...")
        x = torch.randn(1, 10)
        y = F.relu(x)
        logger.info("✅ PyTorch working")
        
        # Test scikit-learn
        logger.info("Testing scikit-learn...")
        from sklearn.metrics import accuracy_score
        logger.info("✅ scikit-learn working")
        
        # Test transformers
        logger.info("Testing transformers...")
        from transformers import AutoTokenizer
        logger.info("✅ transformers working")
        
        return True
        
    except Exception as e:
        logger.error("❌ Validation failed: %s", e)
        return False


def main():
    """Main function to validate environment."""
    logger.info("🔍 Validating environment...")
    
    if validate_environment():
        logger.info("✅ All dependencies validated successfully")
        sys.exit(0)
    else:
        logger.error("❌ Environment validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
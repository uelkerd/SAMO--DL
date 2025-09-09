#!/usr/bin/env python3
"""
Minimal Validation Script

This script performs minimal validation checks.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def minimal_validation():
    """Perform minimal validation checks."""
    try:
        logger.info("🔍 Starting minimal validation...")
        
        # Check Python version
        logger.info("Python version: %s", sys.version)
        
        # Check imports
        logger.info("Testing critical imports...")
        import torch
        import transformers
        import sklearn
        
        logger.info("✅ All minimal validation checks passed!")
        return True
        
    except Exception as e:
        logger.error("❌ Minimal validation failed: %s", e)
        return False


def main():
    """Main function."""
    if minimal_validation():
        logger.info("🎉 Minimal validation completed!")
    else:
        logger.error("💥 Minimal validation failed!")


if __name__ == "__main__":
    main()
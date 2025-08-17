#!/usr/bin/env python3
"""
Test Vertex AI Setup Script

This script tests the Vertex AI setup and configuration.
"""


import logging
import sys
from pathlib import Path


# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_vertex_setup():
    """Test Vertex AI setup and configuration."""
    logger.info("🚀 Starting Vertex AI Setup Test")

    try:
        # Test configuration files
        config_dir = Path("configs/vertex_ai")
        if config_dir.exists():
            logger.info(f"✅ Configuration directory exists: {config_dir}")
            
            config_files = list(config_dir.glob("*.json"))
            logger.info(f"✅ Found {len(config_files)} configuration files")
            
            for config_file in config_files:
                logger.info(f"   - {config_file.name}")
        else:
            logger.warning(f"⚠️  Configuration directory not found: {config_dir}")

        # Test data directory
        data_dir = Path("data/vertex_ai")
        if data_dir.exists():
            logger.info(f"✅ Data directory exists: {data_dir}")
            
            data_files = list(data_dir.glob("*.json"))
            logger.info(f"✅ Found {len(data_files)} data files")
            
            for data_file in data_files:
                logger.info(f"   - {data_file.name}")
        else:
            logger.warning(f"⚠️  Data directory not found: {data_dir}")

        # Test model directory
        model_dir = Path("models/checkpoints")
        if model_dir.exists():
            logger.info(f"✅ Model directory exists: {model_dir}")
        else:
            logger.warning(f"⚠️  Model directory not found: {model_dir}")

        logger.info("✅ Vertex AI setup test completed!")

    except Exception as e:
        logger.error(f"❌ Setup test failed: {e}")
        raise


if __name__ == "__main__":
    test_vertex_setup()

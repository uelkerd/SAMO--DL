#!/usr/bin/env python3
"""
Simple Vertex AI Training Script

This script provides a simple interface for training on Google Cloud Vertex AI.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__


def simple_vertex_training():
    """Run simple Vertex AI training setup."""
    logger.info"🚀 Starting Simple Vertex AI Training Setup"

    try:
        # Create training configuration
        config = {
            "project_id": "your-project-id",
            "region": "us-central1",
            "model_name": "bert-emotion-classifier",
            "training_data_path": "gs://your-bucket/data/train.csv",
            "validation_data_path": "gs://your-bucket/data/val.csv",
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
        }

        # Save configuration
        config_dir = Path"configs/vertex_ai"
        config_dir.mkdirparents=True, exist_ok=True

        import json
        with openconfig_dir / "training_config.json", "w" as f:
            json.dumpconfig, f, indent=2

        logger.infof"✅ Configuration saved to {config_dir / 'training_config.json'}"
        logger.info"✅ Simple Vertex AI training setup completed!"

    except Exception as e:
        logger.errorf"❌ Training setup failed: {e}"
        raise


if __name__ == "__main__":
    simple_vertex_training()

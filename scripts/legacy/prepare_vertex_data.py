#!/usr/bin/env python3
"""Prepare Vertex AI Data Script.

This script prepares data for training on Google Cloud Vertex AI.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_vertex_data():
    """Prepare data for Vertex AI training."""
    logger.info("🚀 Starting Vertex AI Data Preparation")

    try:
        # Create data directory structure
        data_dir = Path("data/vertex_ai")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample training data
        sample_data = [
            {"text": "I am feeling happy today!", "labels": [1, 0, 0, 0]},
            {"text": "This makes me sad.", "labels": [0, 1, 0, 0]},
            {"text": "I'm really angry about this!", "labels": [0, 0, 1, 0]},
            {"text": "I'm scared of what might happen.", "labels": [0, 0, 0, 1]},
        ]

        # Save training data
        import json
        with open(data_dir / "training_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)

        logger.info("✅ Training data saved to {data_dir / "training_data.json'}")
        logger.info(f"✅ Created {len(sample_data)} training samples")

        # Create configuration file
        config = {
            "model_name": "bert-base-uncased",
            "num_epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "max_length": 128,
        }

        with open(data_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Configuration saved to {data_dir / "config.json'}")
        logger.info("✅ Vertex AI data preparation completed!")

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    prepare_vertex_data()

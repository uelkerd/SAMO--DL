#!/usr/bin/env python3
"""
Debug script to understand the dataset structure.
"""

import logging
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.emotion_detection.dataset_loader import create_goemotions_loader


def main():
    logging.info("🔍 Debugging dataset structure...")

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        train_data = datasets["train"]
        first_example = train_data[0]

        logging.info(f"✅ First example keys: {list(first_example.keys())}")
        logging.info(f"✅ First example: {first_example}")

        if "labels" in first_example:
            labels = first_example["labels"]
            logging.info(f"✅ Labels type: {type(labels)}")
            logging.info(f"✅ Labels value: {labels}")
            logging.info(f"✅ Labels length: {len(labels)}")

            if len(labels) > 0:
                logging.info(f"✅ First label: {labels[0]}")
                logging.info(f"✅ First label type: {type(labels[0])}")

        for i in range(1, 5):
            example = train_data[i]
            if "labels" in example:
                logging.info(f"✅ Example {i} labels: {example['labels']}")

        return True

    except Exception as e:
        logging.info(f"❌ Error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

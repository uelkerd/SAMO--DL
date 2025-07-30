            # Try to access first label
        # Check a few more examples
        # Check labels specifically
        # Create loader
        # Get first example
        from models.emotion_detection.dataset_loader import create_goemotions_loader
        import traceback
# Add src to path
#!/usr/bin/env python3
from pathlib import Path
import logging
import sys
import traceback





"""
Debug script to understand the dataset structure.
"""

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    logging.info("🔍 Debugging dataset structure...")

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        train_data = datasets["train"]
        first_example = train_data[0]

        logging.info("✅ First example keys: {list(first_example.keys())}")
        logging.info("✅ First example: {first_example}")

        if "labels" in first_example:
            labels = first_example["labels"]
            logging.info("✅ Labels type: {type(labels)}")
            logging.info("✅ Labels value: {labels}")
            logging.info("✅ Labels length: {len(labels)}")

            if len(labels) > 0:
                logging.info("✅ First label: {labels[0]}")
                logging.info("✅ First label type: {type(labels[0])}")

        for i in range(1, 5):
            example = train_data[i]
            if "labels" in example:
                logging.info("✅ Example {i} labels: {example['labels']}")

        return True

    except Exception as e:
        logging.info("❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

        # Create loader
        # Get first example
        # Try different ways to access
import logging
import sys
        import traceback
import traceback
# Add src to path
#!/usr/bin/env python3
from pathlib import Path
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader





"""
Simple test to understand the dataset object type.
"""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def main():
    logging.info("🔍 Simple test...")

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        train_data = datasets["train"]
        first_example = train_data[0]

        logging.info("✅ Type of first_example: {type(first_example)}")
        logging.info("✅ Dir of first_example: {dir(first_example)}")

        try:
            logging.info("✅ As dict: {dict(first_example)}")
        except:
            logging.info("❌ Cannot convert to dict")

        try:
            logging.info("✅ Keys: {first_example.keys()}")
        except:
            logging.info("❌ No keys method")

        try:
            logging.info("✅ Labels: {first_example['labels']}")
        except Exception as e:
            logging.info("❌ Cannot access labels: {e}")

        try:
            logging.info("✅ Labels attr: {getattr(first_example, 'labels', 'No labels attr')}")
        except Exception as e:
            logging.info("❌ Cannot get labels attr: {e}")

        return True

    except Exception as e:
        logging.info("❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

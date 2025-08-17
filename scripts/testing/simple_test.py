        # Create loader
        # Get first example
        # Try different ways to access
        from src.models.emotion_detection.dataset_loader import create_goemotions_loader
        import traceback
# Add src to path
#!/usr/bin/env python3
from pathlib import Path
import logging
import sys
import traceback





"""
Simple test to understand the dataset object type.
"""

sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))

def main():
    logging.info"🔍 Simple test..."

    try:
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        train_data = datasets["train"]
        first_example = train_data[0]

        logging.info("✅ Type of first_example: {typefirst_example}")
        logging.info("✅ Dir of first_example: {dirfirst_example}")

        try:
            logging.info("✅ As dict: {dictfirst_example}")
        except:
            logging.info"❌ Cannot convert to dict"

        try:
            logging.info("✅ Keys: {first_example.keys()}")
        except:
            logging.info"❌ No keys method"

        try:
            logging.info"✅ Labels: {first_example['labels']}"
        except Exception as e:
            logging.info"❌ Cannot access labels: {e}"

        try:
            logging.info("✅ Labels attr: {getattrfirst_example, 'labels', 'No labels attr'}")
        except Exception as e:
            logging.info"❌ Cannot get labels attr: {e}"

        return True

    except Exception as e:
        logging.info"❌ Error: {e}"
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit1

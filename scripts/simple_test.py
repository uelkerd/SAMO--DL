import sys
import traceback

#!/usr/bin/env python3
"""
Simple test to understand the dataset object type.
"""

from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("🔍 Simple test...")

    try:
        from models.emotion_detection.dataset_loader import create_goemotions_loader

        # Create loader
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        # Get first example
        train_data = datasets["train"]
        first_example = train_data[0]

        print("✅ Type of first_example: {type(first_example)}")
        print("✅ Dir of first_example: {dir(first_example)}")

        # Try different ways to access
        try:
            print("✅ As dict: {dict(first_example)}")
        except:
            print("❌ Cannot convert to dict")

        try:
            print("✅ Keys: {first_example.keys()}")
        except:
            print("❌ No keys method")

        try:
            print("✅ Labels: {first_example['labels']}")
        except Exception as e:
            print("❌ Cannot access labels: {e}")

        try:
            print("✅ Labels attr: {getattr(first_example, 'labels', 'No labels attr')}")
        except Exception as e:
            print("❌ Cannot get labels attr: {e}")

        return True

    except Exception as e:
        print("❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

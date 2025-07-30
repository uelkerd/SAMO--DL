import sys
import traceback

#!/usr/bin/env python3
"""
Debug script to understand the dataset structure.
"""

from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("🔍 Debugging dataset structure...")

    try:
        from models.emotion_detection.dataset_loader import create_goemotions_loader

        # Create loader
        loader = create_goemotions_loader()
        datasets = loader.prepare_datasets()

        # Get first example
        train_data = datasets["train"]
        first_example = train_data[0]

        print("✅ First example keys: {list(first_example.keys())}")
        print("✅ First example: {first_example}")

        # Check labels specifically
        if "labels" in first_example:
            labels = first_example["labels"]
            print("✅ Labels type: {type(labels)}")
            print("✅ Labels value: {labels}")
            print("✅ Labels length: {len(labels)}")

            # Try to access first label
            if len(labels) > 0:
                print("✅ First label: {labels[0]}")
                print("✅ First label type: {type(labels[0])}")

        # Check a few more examples
        for i in range(1, 5):
            example = train_data[i]
            if "labels" in example:
                print("✅ Example {i} labels: {example['labels']}")

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

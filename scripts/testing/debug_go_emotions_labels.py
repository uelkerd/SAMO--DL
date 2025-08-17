#!/usr/bin/env python3
"""
Debug the actual GoEmotions label structure to understand the mapping.
"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies."""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pandas"])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

# Install dependencies first
if not install_dependencies():
    print("❌ Cannot proceed without dependencies")
    sys.exit(1)

from datasets import load_dataset

def debug_go_emotions():
    """Debug the actual GoEmotions dataset structure."""
    print("🔍 Debugging GoEmotions dataset structure...")

    # Load the dataset
    go_emotions = load_dataset("go_emotions", "simplified")

    print("\n📊 Dataset structure:")
    print(f"Keys: {list(go_emotions.keys())}")
    print("Train size: {len(go_emotions["train'])}")
    print("Validation size: {len(go_emotions["validation'])}")
    print("Test size: {len(go_emotions["test'])}")

    # Check first few examples
    print("\n📊 First 5 examples:")
    for i in range(min(5, len(go_emotions['train']))):
        example = go_emotions['train'][i]
        print(f"Example {i}:")
        print("  Text: {example["text'][:100]}...")
        print("  Labels: {example["labels']}")
        print("  Label types: {[type(label) for label in example["labels']]}")
        print()

    # Check if there's a label mapping
    print("\n🔍 Checking for label mapping...")

    # Try to get the dataset info
    try:
        dataset_info = go_emotions['train'].info
        print(f"Dataset info: {dataset_info}")
    except:
        print("No dataset info available")

    # Check if there are features
    try:
        features = go_emotions['train'].features
        print(f"Features: {features}")
    except:
        print("No features available")

    # Look for label names in the dataset
    print("\n🔍 Looking for label names...")

    # Check if there's a label_names field
    if hasattr(go_emotions, 'label_names'):
        print(f"Label names: {go_emotions.label_names}")
    else:
        print("No label_names attribute")

    # Check if there's a features attribute with label names
    if hasattr(go_emotions['train'], 'features'):
        features = go_emotions['train'].features
        print(f"Features: {features}")
        if 'labels' in features:
            print("Labels feature: {features["labels']}")

    # Try to get the original dataset
    print("\n🔍 Trying original dataset...")
    try:
        original_go_emotions = load_dataset("go_emotions")
        print(f"Original dataset keys: {list(original_go_emotions.keys())}")

        if 'train' in original_go_emotions:
            print("Original train size: {len(original_go_emotions["train'])}")
            example = original_go_emotions['train'][0]
            print(f"Original example: {example}")
    except Exception as e:
        print(f"Could not load original dataset: {e}")

    # Check the dataset card
    print("\n🔍 Checking dataset documentation...")
    print("GoEmotions dataset should have emotion names like:")
    print("['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grie", "joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relie", "remorse', 'sadness', 'surprise', 'neutral']")

    return go_emotions

if __name__ == "__main__":
    debug_go_emotions()

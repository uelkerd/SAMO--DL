#!/usr/bin/env python3
"""
Setup script for testing the emotion detection model.
"""

import os
import json
import shutil

def check_model_files():
    """Check if required model files exist."""
    print("🔍 Checking for model files...")

    required_files = {
        'model': 'best_simple_model.pth',
        'results': 'simple_training_results.json'
    }

    missing_files = []
    existing_files = {}

    for file_type, filename in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            existing_files[file_type] = (filename, size)
            print(f"✅ {file_type.capitalize()}: {filename} ({size:,} bytes)")
        else:
            missing_files.append(file_type)
            print(f"❌ {file_type.capitalize()}: {filename} - MISSING")

    return existing_files, missing_files

def create_mock_results():
    """Create mock results file for testing if missing."""
    print("\n🔧 Creating mock results file for testing...")

    # Mock results based on our training
    mock_results = {
        "best_f1": 0.6692,
        "target_achieved": False,
        "num_labels": 12,
        "go_samples": 43410,
        "journal_samples": 150,
        "all_emotions": [
            "anxious", "calm", "content", "excited", "frustrated",
            "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"
        ],
        "emotion_mapping": {
            "joy": "happy",
            "gratitude": "grateful",
            "pride": "proud",
            "excitement": "excited",
            "optimism": "hopeful",
            "sadness": "sad",
            "fear": "anxious",
            "anger": "frustrated",
            "disgust": "frustrated",
            "surprise": "excited",
            "love": "content",
            "caring": "content",
            "approval": "proud",
            "admiration": "proud",
            "amusement": "happy",
            "confusion": "anxious",
            "curiosity": "excited",
            "desire": "excited",
            "disappointment": "sad",
            "disapproval": "frustrated",
            "embarrassment": "anxious",
            "grief": "sad",
            "nervousness": "anxious",
            "realization": "content",
            "relief": "calm",
            "remorse": "sad",
            "neutral": "calm"
        }
    }

    with open('simple_training_results.json', 'w') as f:
        json.dump(mock_results, f, indent=2)

    print("✅ Created mock results file: simple_training_results.json")

def find_model_file(min_size_bytes: int = 0):
    """Find the model file in common locations."""
    print("\n🔍 Searching for model file...")

    search_locations = [
        "best_simple_model.pth",
        "best_focal_model.pth",  # Fallback
        os.path.expanduser("~/Downloads/best_simple_model.pth"),
        os.path.expanduser("~/Desktop/best_simple_model.pth"),
        os.path.expanduser("~/best_simple_model.pth")
    ]

    for location in search_locations:
        if os.path.exists(location):
            size = os.path.getsize(location)
            if size < min_size_bytes:
                print(f"⚠️  Skipping {location} - too small ({size:,} bytes < {min_size_bytes:,} bytes)")
                continue

            print(f"✅ Found model: {location} ({size:,} bytes)")

            # Copy to current directory if not already here
            if location != "best_simple_model.pth":
                shutil.copy2(location, "best_simple_model.pth")
                print("✅ Copied to: best_simple_model.pth")

            return True

    print("❌ Model file not found in common locations")
    return False

def setup_testing():
    """Main setup function."""
    print("🚀 SETTING UP MODEL TESTING")
    print("=" * 50)

    # Check existing files
    existing_files, missing_files = check_model_files()

    # Define minimum size for model files (10KB)
    min_size_bytes = 10 * 1024

    # Check if model file is missing or too small
    model_missing = 'model' in missing_files
    model_exists = os.path.exists('best_simple_model.pth')

    if model_exists:
        model_size = os.path.getsize('best_simple_model.pth')
        model_too_small = model_size < min_size_bytes
    else:
        model_too_small = False

    needs_model = model_missing or model_too_small

    if needs_model and not find_model_file(min_size_bytes=min_size_bytes):
        print("\n❌ Cannot proceed without model file!")
        print("📋 Please download best_simple_model.pth and place it in this directory")
        return False

    # Create mock results if missing
    if 'results' in missing_files:
        create_mock_results()

    print("\n✅ Setup complete! Ready for testing.")
    return True

def test_model_loading():
    """Test loading the model file to verify it's valid."""
    if not os.path.exists('best_simple_model.pth'):
        return False

    print("✅ Model file exists")

    # Try to load safely with weights_only when supported
    import torch
    try:
        # Try with weights_only=True for security (PyTorch 1.13+)
        checkpoint = torch.load('best_simple_model.pth', weights_only=True, map_location='cpu')
        print("✅ Model loaded with weights_only=True (secure)")
    except TypeError:
        # Fall back to map_location-only if weights_only not supported
        checkpoint = torch.load('best_simple_model.pth', map_location='cpu')
        print("✅ Model loaded with map_location only (fallback)")

    # Safely inspect the loaded object
    try:
        if isinstance(checkpoint, dict):
            # Handle dictionary checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                layer_count = len(state_dict.keys())
                print(f"✅ Model checkpoint loaded with {layer_count} layers (from state_dict)")
            else:
                layer_count = len(checkpoint.keys())
                print(f"✅ Model checkpoint loaded with {layer_count} layers (from dict keys)")
        elif hasattr(checkpoint, 'state_dict'):
            # Handle torch.nn.Module
            state_dict = checkpoint.state_dict()
            layer_count = len(state_dict.keys())
            print(f"✅ Model checkpoint loaded with {layer_count} layers (from Module.state_dict)")
        elif hasattr(checkpoint, 'parameters'):
            # Fallback to parameters count
            param_count = len(list(checkpoint.parameters()))
            print(f"✅ Model checkpoint loaded with {param_count} parameters")
        else:
            print("✅ Model checkpoint loaded (unknown structure)")
    except Exception as e:
        print(f"⚠️  Could not determine model structure: {e}")
        print("✅ Model checkpoint loaded (structure inspection failed)")

    return True

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")

    try:
        import torch
        import transformers
        from sklearn.preprocessing import LabelEncoder

        print("✅ All required libraries available")
        return test_model_loading()

    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("📋 Install with: pip install torch transformers scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if setup_testing():
        run_quick_test()
        print("\n🎉 Ready to test the model!")
        print("📋 Run: python scripts/test_emotion_model.py")
    else:
        print("\n❌ Setup failed. Please check the issues above.")

#!/usr/bin/env python3
"""
Local Inference Test Script for Emotion Detection Model
Tests the downloaded model files directly without API server
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add the deployment directory to the path
DEPLOYMENT_DIR = PROJECT_ROOT / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


def test_local_inference():
    """Test the local inference with the downloaded model"""
    print("🧪 LOCAL INFERENCE TEST")
    print("=" * 50)

    # Check if model files exist
    model_dir = DEPLOYMENT_DIR / 'model'
    required_files = ['config.json', 'model.safetensors', 'training_args.bin']

    print(f"📁 Checking model directory: {model_dir}")

    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("Please download the model files from Colab first!")
        return False

    print(f"\n✅ All model files found!")

    # Test texts
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so frustrated with this project.",
        "I feel anxious about the presentation.",
        "I'm grateful for all the support.",
        "I'm feeling overwhelmed with tasks.",
        "I'm proud of what I've accomplished.",
        "I'm feeling sad and lonely today.",
        "I'm excited about the new opportunities.",
        "I feel calm and peaceful right now.",
        "I'm hopeful that things will get better."
    ]

    try:
        # Import the inference module
        from inference import EmotionDetector

        print(f"\n🔧 Loading model...")
        detector = EmotionDetector(model_path=str(model_dir))
        print("✅ Model loaded successfully!")

        print("\n🧪 Running inference tests...")
        for i, text in enumerate(test_texts, 1):
            result = detector.predict(text)
            print(f"{i:02d}. {text} -> {result}")

        print("\n✅ Local inference test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Local inference test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_local_inference()
    sys.exit(0 if success else 1) 
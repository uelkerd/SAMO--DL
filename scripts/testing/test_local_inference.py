#!/usr/bin/env python3
"""
Local Inference Test Script for Emotion Detection Model
Tests the downloaded model files directly without API server
"""

import sys

from scripts.testing._bootstrap import ensure_project_root_on_sys_path, ensure_path, configure_basic_logging

# Ensure project root and logging
PROJECT_ROOT = ensure_project_root_on_sys_path()
logger = configure_basic_logging()

# Add the deployment directory to the path
DEPLOYMENT_DIR = PROJECT_ROOT / 'deployment'
ensure_path(DEPLOYMENT_DIR)


def test_local_inference():
    """Test the local inference with the downloaded model"""
    logger.info("🧪 LOCAL INFERENCE TEST")
    logger.info("=" * 50)

    # Check if model files exist
    model_dir = DEPLOYMENT_DIR / 'model'
    required_files = ['config.json', 'model.safetensors', 'training_args.bin']

    logger.info("📁 Checking model directory: %s", model_dir)

    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            logger.info("✅ Found: %s", file)
        else:
            logger.error("❌ Missing: %s", file)
            missing_files.append(file)

    if missing_files:
        logger.error("❌ Missing required files: %s", missing_files)
        logger.info("Please download the model files from Colab first!")
        return False

    logger.info("✅ All model files found!")

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

        logger.info("🔧 Loading model...")
        detector = EmotionDetector(model_path=str(model_dir))
        logger.info("✅ Model loaded successfully!")

        logger.info("🧪 Running inference tests...")
        for i, text in enumerate(test_texts, 1):
            result = detector.predict(text)
            logger.info("%02d. %s -> %s", i, text, result)

        logger.info("✅ Local inference test completed successfully!")
        return True

    except Exception as e:
        logger.error("❌ Local inference test failed for model_dir=%s: %s", model_dir, e)
        return False


if __name__ == "__main__":
    success = test_local_inference()
    sys.exit(0 if success else 1)

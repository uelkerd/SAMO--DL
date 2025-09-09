#!/usr/bin/env python3
"""
Simple Model Finalization Script

This script finalizes the trained model for deployment.
"""

from pathlib import Path
import json
import shutil
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("deployment_ready")


def create_output_directory():
    """Create output directory for finalized model."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Created output directory: %s", OUTPUT_DIR)


def save_metadata():
    """Save model metadata."""
    # Save metadata
    metadata = {
        "model_type": "emotion_detection",
        "version": "1.0.0",
        "framework": "pytorch",
        "created_at": "2024-01-01"
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Saved model metadata")


def verify_requirements():
    """Verify all requirements are met."""
    # Verify requirements
    required_files = ["model.pth", "config.json", "labels.json"]
    
    for file in required_files:
        if not (MODEL_DIR / file).exists():
            logger.error("Missing required file: %s", file)
            return False
    
    logger.info("All requirements verified")
    return True


def finalize_model():
    """Main function to finalize the model."""
    try:
        logger.info("🚀 Starting model finalization...")
        
        # Create output directory
        create_output_directory()
        
        # Verify requirements
        if not verify_requirements():
            logger.error("❌ Requirements verification failed")
            return False
        
        # Save metadata
        save_metadata()
        
        # Copy model files
        logger.info("📁 Copying model files...")
        for file in ["model.pth", "config.json", "labels.json"]:
            shutil.copy2(MODEL_DIR / file, OUTPUT_DIR / file)
        
        logger.info("✅ Model finalization completed successfully!")
        logger.info("📦 Deployment-ready model saved to: %s", OUTPUT_DIR)
        return True
        
    except Exception as e:
        logger.error("❌ Model finalization failed: %s", e)
        return False


def main():
    """Main function."""
    if finalize_model():
        logger.info("🎉 Model finalization completed!")
        sys.exit(0)
    else:
        logger.error("💥 Model finalization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
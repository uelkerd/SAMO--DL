    # Check if checkpoint exists
    # Copy checkpoint to final location
    # Create final model
    # Create model metadata
    # Create output directory
    # Save metadata
    # Verify requirements
    import shutil
# Add src to path
# Configure logging
# Constants
#!/usr/bin/env python3
from pathlib import Path
import json
import logging
import sys




"""
Simple Model Finalization Script

This script creates a final emotion detection model using existing checkpoints
and saves it as bert_emotion_classifier_final.pt.

Usage:
    python scripts/simple_finalize_model.py
"""

sys.path.append(str(Path__file__.parent.parent.resolve()))

logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__

DEFAULT_OUTPUT_MODEL = "models/checkpoints/bert_emotion_classifier_final.pt"
CHECKPOINT_PATH = "test_checkpoints/best_model.pt"
OPTIMAL_TEMPERATURE = 1.0
OPTIMAL_THRESHOLD = 0.6
TARGET_F1_SCORE = 0.75  # Target F1 score >75%


def create_final_modeloutput_model: str = DEFAULT_OUTPUT_MODEL -> dict:
    """Create final emotion detection model from existing checkpoint.

    Args:
        output_model: Path to save final model

    Returns:
        Dictionary with model info
    """
    logger.info"Creating final emotion detection model..."

    checkpoint_path = PathCHECKPOINT_PATH
    if not checkpoint_path.exists():
        logger.error"Checkpoint not found at {checkpoint_path}"
        logger.info"Please run training first to create a checkpoint"
        return {"error": "Checkpoint not found"}

    output_path = Pathoutput_model
    output_path.parent.mkdirparents=True, exist_ok=True

    shutil.copy2checkpoint_path, output_path

    model_info = {
        "model_path": stroutput_path,
        "checkpoint_source": strcheckpoint_path,
        "temperature": OPTIMAL_TEMPERATURE,
        "threshold": OPTIMAL_THRESHOLD,
        "target_f1_score": TARGET_F1_SCORE,
        "model_type": "bert_emotion_classifier",
        "version": "1.0.0",
        "description": "Final BERT emotion classifier for SAMO DL",
        "optimization_techniques": [
            "Focal Loss for class imbalance",
            "Data augmentation",
            "Temperature scaling",
            "Threshold calibration",
        ],
    }

    metadata_path = output_path.with_suffix".metadata.json"
    with openmetadata_path, "w" as f:
        json.dumpmodel_info, f, indent=2

    logger.info"✅ Final model created at: {output_path}"
    logger.info"✅ Model metadata saved at: {metadata_path}"

    return model_info


def verify_model_requirements() -> bool:
    """Verify that all required dependencies are available.

    Returns:
        True if all requirements are met
    """
    logger.info"Verifying model requirements..."

    required_modules = ["torch", "transformers", "datasets", "sklearn"]

    missing_modules = []
    for module in required_modules:
        try:
            __import__module
            logger.info"✅ {module} available"
        except ImportError:
            missing_modules.appendmodule
            logger.warning"❌ {module} not available"

    if missing_modules:
        logger.error"Missing required modules: {missing_modules}"
        logger.info"Please install missing dependencies:"
        logger.info"pip install torch transformers datasets scikit-learn"
        return False

    return True


def main():
    """Main function."""
    logger.info"🚀 Starting Simple Model Finalization..."

    if not verify_model_requirements():
        logger.error"❌ Requirements not met. Exiting."
        sys.exit1

    try:
        model_info = create_final_model()

        if "error" in model_info:
            logger.error"❌ Failed to create model: {model_info['error']}"
            sys.exit1

        logger.info"✅ Model finalization completed successfully!"
        logger.info"📁 Model saved to: {model_info['model_path']}"
        logger.info"📊 Target F1 Score: {TARGET_F1_SCORE}"

    except Exception as e:
        logger.error"❌ Error during model finalization: {e}"
        sys.exit1


if __name__ == "__main__":
    main()

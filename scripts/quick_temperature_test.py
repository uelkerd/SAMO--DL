import logging

import sys

#!/usr/bin/env python3
from pathlib import Path

from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from models.emotion_detection.bert_classifier import evaluate_emotion_classifier



"""
Quick Temperature Scaling Test.
"""

sys.path.append(str(Path.cwd() / "src"))

def quick_temperature_test():
    logging.info("🌡️ Quick Temperature Scaling Test")

    # Initialize trainer with dev_mode
    trainer = EmotionDetectionTrainer()

    # Load model
    model_path = Path("test_checkpoints/best_model.pt")
    if not model_path.exists():
        logging.info("❌ Model not found")
        return

    trainer.load_model(str(model_path))
    logging.info("✅ Model loaded")

    # Test temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]
    threshold = 0.5

    logging.info("\n🎯 Testing temperatures with threshold {threshold}")
    logging.info("-" * 50)

    for temp in temperatures:
        logging.info("\n🌡️ Temperature: {temp}")

        # Update temperature
        trainer.model.set_temperature(temp)

        # Quick evaluation
        metrics = evaluate_emotion_classifier(
            trainer.model, trainer.val_loader, trainer.device, threshold=threshold
        )

        logging.info("  📊 Macro F1: {metrics['macro_f1']:.4f}")
        logging.info("  📊 Micro F1: {metrics['micro_f1']:.4f}")

    logging.info("\n🎉 Temperature scaling test complete!")


if __name__ == "__main__":
    quick_temperature_test()

        # Quick evaluation
        # Update temperature
    # Initialize trainer with dev_mode
    # Load model
    # Test temperatures
#!/usr/bin/env python3
from src.models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from pathlib import Path
import logging
import sys

"""
Quick Temperature Scaling Test.
"""

sys.path.append(str(Path.cwd() / "src"))

def quick_temperature_test():
    logging.info("🌡️ Quick Temperature Scaling Test")

    trainer = EmotionDetectionTrainer()

    model_path = Path("test_checkpoints/best_model.pt")
    if not model_path.exists():
        logging.info("❌ Model not found")
        return

    trainer.load_model(str(model_path))
    logging.info("✅ Model loaded")

    temperatures = [1.0, 2.0, 3.0, 4.0]
    threshold = 0.5

    logging.info("\n🎯 Testing temperatures with threshold {threshold}")
    logging.info("-" * 50)

    for temp in temperatures:
        logging.info("\n🌡️ Temperature: {temp}")

        trainer.model.set_temperature(temp)

        evaluate_emotion_classifier(
            trainer.model, trainer.val_loader, trainer.device, threshold=threshold
        )

        logging.info("  📊 Macro F1: {metrics['macro_f1']:.4f}")
        logging.info("  📊 Micro F1: {metrics['micro_f1']:.4f}")

    logging.info("\n🎉 Temperature scaling test complete!")


if __name__ == "__main__":
    quick_temperature_test()

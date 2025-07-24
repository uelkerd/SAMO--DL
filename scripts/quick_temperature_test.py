#!/usr/bin/env python3
"""
Quick Temperature Scaling Test.
"""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from models.emotion_detection.bert_classifier import evaluate_emotion_classifier


def quick_temperature_test():
    print("🌡️ Quick Temperature Scaling Test")

    # Initialize trainer with dev_mode
    trainer = EmotionDetectionTrainer()

    # Load model
    model_path = Path("test_checkpoints/best_model.pt")
    if not model_path.exists():
        print("❌ Model not found")
        return

    trainer.load_model(str(model_path))
    print("✅ Model loaded")

    # Test temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]
    threshold = 0.5

    print(f"\n🎯 Testing temperatures with threshold {threshold}")
    print("-" * 50)

    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")

        # Update temperature
        trainer.model.set_temperature(temp)

        # Quick evaluation
        metrics = evaluate_emotion_classifier(
            trainer.model, trainer.val_loader, trainer.device, threshold=threshold
        )

        print(f"  📊 Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  📊 Micro F1: {metrics['micro_f1']:.4f}")

    print("\n🎉 Temperature scaling test complete!")


if __name__ == "__main__":
    quick_temperature_test()

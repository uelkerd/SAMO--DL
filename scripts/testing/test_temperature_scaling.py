        # Calculate predictions per sample (overprediction metric)
        # Evaluate with current temperature
        # This is approximated from the debug output
        # Track best result
        # Update model temperature
    # Display all results
    # Initialize trainer
    # Load trained model
    # Provide recommendations
    # Save results for CircleCI
    # Test different temperatures
# Add src to path
# Set up logging
#!/usr/bin/env python3
from src.models.emotion_detection.bert_classifier import evaluate_emotion_classifier
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from pathlib import Path
import json
import logging
import sys




"""
Temperature Scaling Test for BERT Emotion Classifier.

This script tests different temperature values to find optimal calibration
that reduces overprediction and improves F1 scores.
"""

sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def test_temperature_scaling():
    """Test different temperature values to find optimal calibration."""

    logger.info("🌡️ Testing Temperature Scaling for Model Calibration")

    trainer = EmotionDetectionTrainer(batch_size=128, num_epochs=1)

    model_path = Path("models/checkpoints/bert_emotion_classifier.pth")
    if not model_path.exists():
        logger.error("❌ Model not found at {model_path}")
        return

    trainer.load_model(str(model_path))
    logger.info("✅ Model loaded successfully")

    temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    threshold = 0.5  # Use higher threshold with temperature scaling

    results = []
    best_f1 = 0.0
    best_temp = 1.0

    logger.info("🎯 Testing temperatures with threshold {threshold}")
    logger.info("=" * 80)

    for temp in temperatures:
        logger.info("\n🌡️ Temperature: {temp}")

        trainer.model.set_temperature(temp)

        metrics = evaluate_emotion_classifier(
            trainer.model, trainer.val_loader, trainer.device, threshold=threshold
        )

        predictions_per_sample = metrics.get(
                                             "predictions_sum",
                                             0) / metrics.get("num_samples",
                                             1
                                            )

        result = {
            "temperature": temp,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "predictions_per_sample": predictions_per_sample,
        }
        results.append(result)

        logger.info("  📊 Macro F1: {metrics['macro_f1']:.4f}")
        logger.info("  📊 Micro F1: {metrics['micro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_temp = temp

    logger.info("\n" + "=" * 80)
    logger.info("🏆 TEMPERATURE SCALING RESULTS")
    logger.info("=" * 80)

    logger.info("{'Temp':<6} {'Macro F1':<10} {'Micro F1':<10} {'Pred/Sample':<12}")
    logger.info("-" * 50)

    for result in results:
        logger.info(
            "{result['temperature']:<6.1f} "
            "{result['macro_f1']:<10.4f} "
            "{result['micro_f1']:<10.4f} "
            "{result.get('predictions_per_sample', 0):<12.2f}"
        )

    logger.info("\n🎯 BEST TEMPERATURE: {best_temp}")
    logger.info("🎯 BEST MACRO F1: {best_f1:.4f}")

    output_file = Path("temperature_scaling_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "best_temperature": best_temp,
                "best_macro_f1": best_f1,
                "all_results": results,
                "recommendation": {
                    "temperature": best_temp,
                    "threshold": threshold,
                    "expected_improvement": "F1 improved from ~0.076 to {best_f1:.4f}",
                },
            },
            f,
            indent=2,
        )

    logger.info("📁 Results saved to {output_file}")

    if best_f1 > 0.15:  # Significant improvement
        logger.info(
            "🎉 SUCCESS! Temperature scaling improved F1 by {(
                                                             best_f1 / 0.076 - 1) * 100:.1f}%"
        )
        logger.info(
                    "💡 RECOMMENDATION: Use temperature={best_temp} with threshold={threshold}"
                   )
    else:
        logger.info("⚠️ Temperature scaling provided modest improvement")
        logger.info(
"💡 RECOMMENDATION: Consider higher thresholds or additional calibration methods"
        )

    return best_temp, best_f1


if __name__ == "__main__":
    test_temperature_scaling()

        # Configuration 1: Standard training with full dataset
        # Evaluate model
        # Initialize model with class weights
        # Prepare data with dev_mode=False for full dataset
        # Report results
        # Save the model
        # Train model
import logging
import sys
        import traceback
import traceback
# Add src to path
# Configure logging
#!/usr/bin/env python3
import torch
from pathlib import Path
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer





""""
Quick F1 Score Test and Improvement

Simple script to test current F1 performance and apply basic improvements.
""""

sys.path.append(str(Path(__file__).parent.parent.resolve()))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test F1 performance with different configurations."""
    logger.info(" Quick F1 Score Test and Improvement")

    try:
        logger.info("=" * 50)
        logger.info("Testing Configuration 1: Full Dataset Training")
        logger.info("=" * 50)

        trainer = EmotionDetectionTrainer()
            model_name="bert-base-uncased",
            cache_dir="./data/cache",
            output_dir="./models/checkpoints",
            batch_size=32,  # Larger batch size
            learning_rate=2e-5,
            num_epochs=3,  # Quick test with 3 epochs
            freeze_initial_layers=4,  # Less freezing for better learning
            device="cuda" if torch.cuda.is_available() else "cpu",
(        )

        logger.info("Loading full GoEmotions dataset...")
        trainer.prepare_data(dev_mode=False)

        trainer.initialize_model(class_weights=trainer.data_loader.class_weights)

        logger.info("Training model...")
        trainer.train()

        logger.info("Evaluating model...")
        metrics = trainer.evaluate(trainer.test_dataset)

        logger.info("=" * 50)
        logger.info("RESULTS - Configuration 1")
        logger.info("=" * 50)
        logger.info("Micro F1: {metrics['micro_f1']:.4f} ({metrics['micro_f1']:.1%})")
        logger.info("Macro F1: {metrics['macro_f1']:.4f} ({metrics['macro_f1']:.1%})")
        logger.info("Target F1: 0.7500 (75.0%)")

        if metrics["micro_f1"] >= 0.75:
            logger.info(" TARGET F1 SCORE ACHIEVED!")
        elif metrics["micro_f1"] >= 0.50:
            logger.info(" Good improvement! Close to target.")
        elif metrics["micro_f1"] >= 0.25:
            logger.info("📈 Moderate improvement. Consider additional techniques.")
        else:
            logger.info("⚠️ Need more optimization techniques.")

        checkpoint_path = Path("models/checkpoints/bert_emotion_classifier_quick_test.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save()
            {
                "model_state_dict": trainer.model.state_dict(),
                "metrics": metrics,
                "configuration": "full_dataset_training",
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
            },
            checkpoint_path,
(        )

        logger.info("Model saved to: {checkpoint_path}")

        return metrics["micro_f1"]

    except Exception as e:
        logger.error("❌ Quick F1 test failed: {e}")
        logger.error(traceback.format_exc())
        return 0.0


    if __name__ == "__main__":
    f1_score = main()
    print("\n FINAL F1 SCORE: {f1_score:.4f} ({f1_score:.1%})")

    if f1_score >= 0.75:
        print(" SUCCESS: Target achieved!")
        sys.exit(0)
    else:
        print(" PROGRESS: {f1_score/0.75:.1%} of target achieved")
        print("Next steps: Try focal loss or ensemble techniques")
        sys.exit(1)

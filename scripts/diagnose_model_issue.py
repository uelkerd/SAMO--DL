import sys

#!/usr/bin/env python3
import logging
from pathlib import Path

# Add src to path
import torch
from models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Configure logging

"""Diagnose Model Issue - Why is the model predicting all emotions?

This script investigates why the BERT model is predicting all emotions
as positive instead of learning proper discrimination.
"""

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_model_outputs():
    """Diagnose what the model is actually outputting."""
    logger.info("🔍 Diagnosing Model Output Issue")

    try:
        # Create trainer
        trainer = EmotionDetectionTrainer(
            model_name="bert-base-uncased",
            cache_dir="./data/cache",
            output_dir="./test_checkpoints_dev",
            batch_size=8,  # Small batch for debugging
            device="cpu",
        )

        # Prepare data
        trainer.prepare_data(dev_mode=True)
        trainer.initialize_model(class_weights=trainer.data_loader.class_weights)

        # Load trained model
        model_path = Path("./test_checkpoints_dev/best_model.pt")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("✅ Loaded trained model")
        else:
            logger.info("⚠️  No trained model found, using fresh model")

        # Get a few samples from validation set
        trainer.model.eval()

        # Get one batch for detailed analysis
        for batch_idx, batch in enumerate(trainer.val_dataloader):
            if batch_idx > 0:  # Only analyze first batch
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            logger.info("📊 Batch Analysis:")
            logger.info("  Input shape: {input_ids.shape}")
            logger.info("  Labels shape: {labels.shape}")
            logger.info("  Labels sum per sample: {labels.sum(dim=1).tolist()}")
            logger.info("  Labels mean: {labels.float().mean():.4f}")

            # Forward pass
            with torch.no_grad():
                logits = trainer.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.sigmoid(logits)

            logger.info("📈 Model Output Analysis:")
            logger.info("  Logits shape: {logits.shape}")
            logger.info("  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            logger.info("  Logits mean: {logits.mean():.4f}")
            logger.info("  Logits std: {logits.std():.4f}")

            logger.info(
                "  Probabilities range: [{probabilities.min():.4f}, {probabilities.max():.4f}]"
            )
            logger.info("  Probabilities mean: {probabilities.mean():.4f}")
            logger.info("  Probabilities std: {probabilities.std():.4f}")

            # Check if all probabilities are high
            high_prob_count = (probabilities > 0.5).sum()
            total_predictions = probabilities.numel()

            logger.info(
                "  Predictions > 0.5: {high_prob_count}/{total_predictions} ({100*high_prob_count/total_predictions:.1f}%)"
            )

            # Sample analysis
            for i in range(min(3, input_ids.shape[0])):
                sample_probs = probabilities[i]
                sample_labels = labels[i]

                top_emotions_idx = torch.topk(sample_probs, 5).indices
                true_emotions_idx = torch.where(sample_labels == 1)[0]

                logger.info("  Sample {i}:")
                logger.info("    True emotions: {true_emotions_idx.tolist()}")
                logger.info("    Top predicted: {top_emotions_idx.tolist()}")
                logger.info("    Top probs: {sample_probs[top_emotions_idx].tolist()}")

            break

        return True

    except Exception as _:
        logger.error("❌ Diagnosis failed: {e}")
        return False


def diagnose_loss_function():
    """Check if the loss function is working correctly."""
    logger.info("🔍 Diagnosing Loss Function")

    try:
        # Create simple test case
        batch_size, num_emotions = 4, 28

        # Create fake logits and labels
        logits = torch.randn(batch_size, num_emotions) * 2  # Random logits
        labels = torch.zeros(batch_size, num_emotions)

        # Set some emotions as positive
        labels[0, [1, 5, 10]] = 1  # Sample 0 has emotions 1, 5, 10
        labels[1, [2, 7]] = 1  # Sample 1 has emotions 2, 7

        logger.info("Test logits shape: {logits.shape}")
        logger.info("Test labels shape: {labels.shape}")
        logger.info("Labels per sample: {labels.sum(dim=1).tolist()}")

        # Test BCE loss
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(logits, labels)

        logger.info("BCE Loss: {loss.item():.4f}")

        # Test with class weights
        pos_weight = torch.ones(num_emotions) * 2.0  # Give more weight to positive class
        weighted_bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        weighted_loss = weighted_bce(logits, labels)

        logger.info("Weighted BCE Loss: {weighted_loss.item():.4f}")

        # Check gradients
        logits.requires_grad_(True)
        loss.backward()

        logger.info("Gradient magnitude: {logits.grad.abs().mean():.6f}")

        return True

    except Exception as _:
        logger.error("❌ Loss function diagnosis failed: {e}")
        return False


def main():
    """Run all diagnostics."""
    logger.info("🧪 SAMO Model Diagnosis Suite")
    logger.info("=" * 50)

    tests = [
        ("Model Output Analysis", diagnose_model_outputs),
        ("Loss Function Analysis", diagnose_loss_function),
    ]

    passed = 0
    for test_name, test_func in tests:
        logger.info("\n🔍 {test_name}")
        if test_func():
            passed += 1
            logger.info("✅ {test_name} completed")
        else:
            logger.error("❌ {test_name} failed")

    logger.info("=" * 50)
    logger.info("📊 Diagnostics completed: {passed}/{len(tests)} tests passed")

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

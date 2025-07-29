#!/usr/bin/env python3
"""
Threshold Optimization for Emotion Detection

This script optimizes prediction thresholds for each emotion class to maximize F1 scores.
It performs grid search over different threshold values and finds the optimal threshold per class.

Usage:
    python scripts/threshold_optimization.py [--model_path PATH] [--threshold_range 0.1 0.9]
"""

import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(
    model_path: str, device: str = "cpu"
) -> tuple[torch.nn.Module, torch.utils.data.DataLoader]:
    """Load trained model and validation data.

    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, validation_dataloader)
    """
    logger.info(f"Loading model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model, _ = create_bert_emotion_classifier(
        model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=4
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load validation data
    logger.info("Loading validation dataset...")
    data_loader = GoEmotionsDataLoader()
    datasets = data_loader.prepare_data(dev_mode=False)
    val_dataset = datasets["val_dataset"]

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    logger.info("Model and data loaded successfully")
    logger.info(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   • Validation examples: {len(val_dataset)}")

    return model, val_loader


def get_predictions_and_labels(
    model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions and true labels for validation set.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run inference on

    Returns:
        Tuple of (predictions, true_labels) as numpy arrays
    """
    logger.info("Generating predictions...")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float()

            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs)

            all_predictions.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    logger.info("Generated predictions:")
    logger.info(f"   • Shape: {predictions.shape}")
    logger.info(f"   • Range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    return predictions, true_labels


def optimize_thresholds(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    threshold_range: tuple[float, float] = (0.1, 0.9),
    num_thresholds: int = 20,
) -> dict:
    """Optimize thresholds for each emotion class.

    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels
        threshold_range: Range of thresholds to test
        num_thresholds: Number of threshold values to test

    Returns:
        Dictionary with optimization results
    """
    logger.info("Optimizing thresholds...")
    logger.info(f"   • Range: {threshold_range}")
    logger.info(f"   • Steps: {num_thresholds}")

    # Generate threshold values
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

    # Emotion labels (from GoEmotions dataset)
    emotion_labels = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]

    # Results storage
    best_thresholds = []
    best_f1_scores = []
    class_results = []

    # Optimize each class separately
    for class_idx in range(predictions.shape[1]):
        logger.info(
            f"Optimizing class {class_idx + 1}/{predictions.shape[1]}: {emotion_labels[class_idx]}"
        )

        class_predictions = predictions[:, class_idx]
        class_labels = true_labels[:, class_idx]

        best_f1 = 0.0
        best_threshold = 0.5
        class_f1_scores = []

        # Test each threshold
        for threshold in thresholds:
            # Apply threshold
            binary_predictions = (class_predictions >= threshold).astype(int)

            # Calculate F1 score
            if np.sum(class_labels) > 0:  # Only if class has positive examples
                f1 = f1_score(class_labels, binary_predictions, zero_division=0)
                class_f1_scores.append(f1)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            else:
                class_f1_scores.append(0.0)

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

        class_results.append(
            {
                "class_idx": class_idx,
                "emotion": emotion_labels[class_idx],
                "best_threshold": best_threshold,
                "best_f1": best_f1,
                "f1_scores": class_f1_scores,
            }
        )

        logger.info(f"   • Best threshold: {best_threshold:.3f}")
        logger.info(f"   • Best F1: {best_f1:.3f}")

    # Calculate overall metrics with optimized thresholds
    optimized_predictions = np.zeros_like(predictions)
    for class_idx, threshold in enumerate(best_thresholds):
        optimized_predictions[:, class_idx] = (predictions[:, class_idx] >= threshold).astype(int)

    # Overall metrics
    macro_f1 = f1_score(true_labels, optimized_predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, optimized_predictions, average="micro", zero_division=0)
    weighted_f1 = f1_score(true_labels, optimized_predictions, average="weighted", zero_division=0)

    # Compare with default threshold (0.5)
    default_predictions = (predictions >= 0.5).astype(int)
    default_macro_f1 = f1_score(true_labels, default_predictions, average="macro", zero_division=0)
    default_micro_f1 = f1_score(true_labels, default_predictions, average="micro", zero_division=0)

    results = {
        "best_thresholds": best_thresholds,
        "best_f1_scores": best_f1_scores,
        "class_results": class_results,
        "overall_metrics": {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "default_macro_f1": default_macro_f1,
            "default_micro_f1": default_micro_f1,
            "improvement_macro": macro_f1 - default_macro_f1,
            "improvement_micro": micro_f1 - default_micro_f1,
        },
    }

    return results


def save_optimization_results(results: dict, output_path: str):
    """Save optimization results to file.

    Args:
        results: Optimization results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays for easy loading
    np.savez(
        output_path,
        best_thresholds=np.array(results["best_thresholds"]),
        best_f1_scores=np.array(results["best_f1_scores"]),
        overall_metrics=results["overall_metrics"],
    )

    # Save detailed results as text
    text_path = output_path.with_suffix(".txt")
    with open(text_path, "w") as f:
        f.write("Threshold Optimization Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"  Macro F1 (optimized): {results['overall_metrics']['macro_f1']:.4f}\n")
        f.write(f"  Micro F1 (optimized): {results['overall_metrics']['micro_f1']:.4f}\n")
        f.write(f"  Weighted F1 (optimized): {results['overall_metrics']['weighted_f1']:.4f}\n")
        f.write(f"  Macro F1 (default): {results['overall_metrics']['default_macro_f1']:.4f}\n")
        f.write(f"  Micro F1 (default): {results['overall_metrics']['default_micro_f1']:.4f}\n")
        f.write(f"  Improvement (macro): {results['overall_metrics']['improvement_macro']:.4f}\n")
        f.write(f"  Improvement (micro): {results['overall_metrics']['improvement_micro']:.4f}\n\n")

        f.write("Per-Class Results:\n")
        f.write("-" * 30 + "\n")
        for class_result in results["class_results"]:
            f.write(
                f"{class_result['emotion']:15s}: "
                f"threshold={class_result['best_threshold']:.3f}, "
                f"F1={class_result['best_f1']:.3f}\n"
            )

    logger.info("Results saved to:")
    logger.info(f"   • {output_path}")
    logger.info(f"   • {text_path}")


def main():
    """Main function to run threshold optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize prediction thresholds for emotion detection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/checkpoints/focal_loss_best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--threshold_range",
        type=float,
        nargs=2,
        default=[0.1, 0.9],
        help="Range of thresholds to test",
    )
    parser.add_argument(
        "--num_thresholds", type=int, default=20, help="Number of threshold values to test"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/optimized/thresholds.npz",
        help="Path to save optimization results",
    )

    args = parser.parse_args()

    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model and data
        model, val_loader = load_model_and_data(args.model_path, device)

        # Get predictions
        predictions, true_labels = get_predictions_and_labels(model, val_loader, device)

        # Optimize thresholds
        results = optimize_thresholds(
            predictions,
            true_labels,
            threshold_range=tuple(args.threshold_range),
            num_thresholds=args.num_thresholds,
        )

        # Save results
        save_optimization_results(results, args.output_path)

        # Print summary
        logger.info("\n🎯 Threshold Optimization Complete!")
        logger.info(f"   • Macro F1 (optimized): {results['overall_metrics']['macro_f1']:.4f}")
        logger.info(
            f"   • Macro F1 (default): {results['overall_metrics']['default_macro_f1']:.4f}"
        )
        logger.info(f"   • Improvement: {results['overall_metrics']['improvement_macro']:.4f}")
        logger.info(f"   • Results saved: {args.output_path}")

        if results["overall_metrics"]["improvement_macro"] > 0.05:
            logger.info("   ✅ Significant improvement achieved!")
        else:
            logger.info("   ⚠️  Modest improvement - consider other techniques")

    except Exception as e:
        logger.error(f"❌ Threshold optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

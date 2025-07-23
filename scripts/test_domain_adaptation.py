#!/usr/bin/env python3
"""Domain Adaptation Testing for SAMO Deep Learning.

This script tests how well the GoEmotions-trained model performs on
journal entries and provides domain adaptation strategies.

Usage:
    python scripts/test_domain_adaptation.py --model-path ./test_checkpoints/best_model.pt
    python scripts/test_domain_adaptation.py --create-journal-samples
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_journal_test_samples() -> list[dict[str, any]]:
    """Create realistic journal entry samples for domain adaptation testing."""
    journal_samples = [
        # Positive emotions
        {
            "text": "Today was absolutely wonderful. I finally got the promotion I've been working towards for months. I feel so proud and accomplished.",
            "expected_emotions": ["joy", "pride", "gratitude"],
        },
        {
            "text": "Had the most amazing dinner with Sarah tonight. We laughed until our stomachs hurt. I'm so grateful for our friendship.",
            "expected_emotions": ["joy", "gratitude", "love"],
        },
        {
            "text": "The meditation session this morning left me feeling so peaceful and centered. I love these quiet moments of reflection.",
            "expected_emotions": ["relief", "gratitude", "love"],
        },
        # Negative emotions
        {
            "text": "Another rejection email today. I'm starting to doubt whether I'll ever find a job that's right for me. This whole process is exhausting.",
            "expected_emotions": ["disappointment", "sadness", "nervousness"],
        },
        {
            "text": "Mom called upset about dad's health again. I feel so helpless being so far away. Why does life have to be so complicated?",
            "expected_emotions": ["sadness", "fear", "caring"],
        },
        {
            "text": "Traffic was terrible, I was late to the meeting, and my boss was not happy. Everything that could go wrong did go wrong today.",
            "expected_emotions": ["annoyance", "disappointment", "anger"],
        },
        # Mixed emotions
        {
            "text": "Graduation was bittersweet. I'm excited about the future but sad to leave all my friends behind. Change is scary but necessary.",
            "expected_emotions": ["joy", "sadness", "nervousness", "excitement"],
        },
        {
            "text": "Finished reading that book about climate change. It was eye-opening but also terrifying. I want to help but don't know where to start.",
            "expected_emotions": ["fear", "caring", "curiosity", "nervousness"],
        },
        # Neutral/complex emotions
        {
            "text": "Spent most of the day organizing my closet. It's funny how decluttering physical space can make your mind feel clearer too.",
            "expected_emotions": ["neutral", "realization"],
        },
        {
            "text": "Watched an old movie with my roommate. We didn't talk much, but it was nice to just be together. Simple moments like these matter.",
            "expected_emotions": ["love", "gratitude", "neutral"],
        },
        # Emotional complexity
        {
            "text": "Had a panic attack during the presentation. My heart was racing and I could barely speak. I'm embarrassed but also proud that I didn't give up.",
            "expected_emotions": ["fear", "nervousness", "embarrassment", "pride"],
        },
        {
            "text": "Therapy was intense today. We talked about childhood memories I'd forgotten. It's painful but I know this healing work is important.",
            "expected_emotions": ["sadness", "grief", "caring", "optimism"],
        },
    ]

    # Save samples for testing
    output_path = Path("data/processed/journal_domain_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(journal_samples, f, indent=2)

    logger.info(f"✅ Created {len(journal_samples)} journal test samples: {output_path}")
    return journal_samples


def load_emotion_mapping() -> dict[str, int]:
    """Load GoEmotions emotion mapping."""
    # GoEmotions emotion labels (28 emotions including neutral)
    goemotions_emotions = [
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

    return {emotion: idx for idx, emotion in enumerate(goemotions_emotions)}


def predict_emotions(
    model_path: str,
    texts: list[str],
    model_name: str = "bert-base-uncased",
    threshold: float = 0.3,
) -> list[dict[str, any]]:
    """Predict emotions for given texts using trained model."""
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Import and initialize model
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier

    model = BERTEmotionClassifier(model_name=model_name, num_labels=28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Emotion mapping
    emotion_mapping = load_emotion_mapping()
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_mapping.items()}

    predictions = []

    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Predict
            outputs = model(encoding["input_ids"], encoding["attention_mask"])
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            # Apply threshold and get predicted emotions
            predicted_emotions = []
            emotion_scores = {}

            for idx, prob in enumerate(probabilities):
                emotion = idx_to_emotion[idx]
                emotion_scores[emotion] = float(prob)

                if prob > threshold:
                    predicted_emotions.append({"emotion": emotion, "confidence": float(prob)})

            # Sort by confidence
            predicted_emotions.sort(key=lambda x: x["confidence"], reverse=True)

            predictions.append(
                {
                    "text": text,
                    "predicted_emotions": predicted_emotions,
                    "all_scores": emotion_scores,
                }
            )

    return predictions


def analyze_domain_adaptation(
    model_path: str, test_samples: list[dict[str, any]] | None = None
) -> dict[str, any]:
    """Analyze how well the model performs on journal entries vs Reddit comments."""
    if test_samples is None:
        test_samples = create_journal_test_samples()

    logger.info("🔍 Analyzing domain adaptation performance...")

    # Extract texts for prediction
    texts = [sample["text"] for sample in test_samples]

    # Get predictions
    predictions = predict_emotions(model_path, texts)

    # Analyze results
    analysis = {
        "total_samples": len(test_samples),
        "predictions": predictions,
        "domain_analysis": {},
        "recommendations": [],
    }

    # Performance analysis
    correct_predictions = 0
    partial_matches = 0

    for i, (sample, pred) in enumerate(zip(test_samples, predictions, strict=False)):
        expected = set(sample["expected_emotions"])
        predicted = {e["emotion"] for e in pred["predicted_emotions"]}

        # Exact match
        if expected == predicted:
            correct_predictions += 1
        # Partial match (at least one emotion correct)
        elif expected.intersection(predicted):
            partial_matches += 1

        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Text: {sample['text'][:100]}...")
        logger.info(f"Expected: {expected}")
        logger.info(f"Predicted: {predicted}")
        logger.info(
            f"Match: {'✅ Exact' if expected == predicted else '🟡 Partial' if expected.intersection(predicted) else '❌ None'}"
        )

    # Calculate metrics
    exact_accuracy = correct_predictions / len(test_samples)
    partial_accuracy = (correct_predictions + partial_matches) / len(test_samples)

    analysis["domain_analysis"] = {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "exact_matches": correct_predictions,
        "partial_matches": partial_matches,
        "no_matches": len(test_samples) - correct_predictions - partial_matches,
    }

    # Generate recommendations
    if exact_accuracy < 0.3:
        analysis["recommendations"].append(
            "❌ Strong domain shift detected - consider domain adaptation"
        )
        analysis["recommendations"].append("• Collect journal entry dataset with emotion labels")
        analysis["recommendations"].append("• Fine-tune model on journal entries")
        analysis["recommendations"].append("• Use data augmentation techniques")
    elif exact_accuracy < 0.6:
        analysis["recommendations"].append("⚠️  Moderate domain adaptation needed")
        analysis["recommendations"].append("• Consider few-shot learning with journal examples")
        analysis["recommendations"].append("• Implement confidence thresholding")
        analysis["recommendations"].append("• Monitor performance on real user data")
    else:
        analysis["recommendations"].append("✅ Good cross-domain performance")
        analysis["recommendations"].append("• Current model should work well for journal entries")
        analysis["recommendations"].append("• Monitor performance and collect feedback")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="SAMO Domain Adaptation Testing")
    parser.add_argument("--model-path", type=str, default="./test_checkpoints/best_model.pt")
    parser.add_argument(
        "--create-journal-samples",
        action="store_true",
        help="Create journal test samples",
    )
    parser.add_argument("--test-adaptation", action="store_true", help="Test domain adaptation")
    parser.add_argument("--threshold", type=float, default=0.3, help="Emotion prediction threshold")

    args = parser.parse_args()

    if args.create_journal_samples or not any([args.test_adaptation]):
        samples = create_journal_test_samples()
        print(f"\n✅ Created {len(samples)} journal test samples")

    if args.test_adaptation:
        if not Path(args.model_path).exists():
            logger.error(f"Model not found: {args.model_path}")
            return

        analysis = analyze_domain_adaptation(args.model_path)

        print("\n" + "=" * 60)
        print("📊 DOMAIN ADAPTATION ANALYSIS")
        print("=" * 60)

        metrics = analysis["domain_analysis"]
        print(f"\nExact Accuracy: {metrics['exact_accuracy']:.2%}")
        print(f"Partial Accuracy: {metrics['partial_accuracy']:.2%}")
        print(f"Exact Matches: {metrics['exact_matches']}/{analysis['total_samples']}")
        print(f"Partial Matches: {metrics['partial_matches']}/{analysis['total_samples']}")
        print(f"No Matches: {metrics['no_matches']}/{analysis['total_samples']}")

        print("\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"   {rec}")

        # Save detailed results
        results_path = Path("domain_adaptation_results.json")
        with open(results_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n📄 Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()

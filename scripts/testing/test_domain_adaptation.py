            # Apply threshold and get predicted emotions
            # Predict
            # Sort by confidence
            # Tokenize
        # Emotional complexity
        # Exact match
        # Mixed emotions
        # Negative emotions
        # Neutral/complex emotions
        # Partial match (at least one emotion correct)
        # Positive emotions
        # Save detailed results
    # Analyze results
    # Calculate metrics
    # Emotion mapping
    # Extract texts for prediction
    # Generate recommendations
    # Get predictions
    # GoEmotions emotion labels (28 emotions including neutral)
    # Import and initialize model
    # Initialize tokenizer
    # Load model
    # Performance analysis
    # Save samples for testing
import argparse
import json
import logging
# Set up logging
#!/usr/bin/env python3
import torch
from pathlib import Path
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
from transformers import AutoTokenizer





"""Domain Adaptation Testing for SAMO Deep Learning.

This script tests how well the GoEmotions-trained model performs on
journal entries and provides domain adaptation strategies.

Usage:
    python scripts/test_domain_adaptation.py --model-path ./test_checkpoints/best_model.pt
    python scripts/test_domain_adaptation.py --create-journal-samples
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_journal_test_samples() -> list[dict[str, any]]:
    """Create realistic journal entry samples for domain adaptation testing."""
    journal_samples = [
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
            "expected_emotions": ["relie", "gratitude", "love"],
        },
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
        {
            "text": "Graduation was bittersweet. I'm excited about the future but sad to leave all my friends behind. Change is scary but necessary.",
            "expected_emotions": ["joy", "sadness", "nervousness", "excitement"],
        },
        {
            "text": "Finished reading that book about climate change. It was eye-opening but also terrifying. I want to help but don't know where to start.",
            "expected_emotions": ["fear", "caring", "curiosity", "nervousness"],
        },
        {
            "text": "Spent most of the day organizing my closet. It's funny how decluttering physical space can make your mind feel clearer too.",
            "expected_emotions": ["neutral", "realization"],
        },
        {
            "text": "Watched an old movie with my roommate. We didn't talk much, but it was nice to just be together. Simple moments like these matter.",
            "expected_emotions": ["love", "gratitude", "neutral"],
        },
        {
            "text": "Had a panic attack during the presentation. My heart was racing and I could barely speak. I'm embarrassed but also proud that I didn't give up.",
            "expected_emotions": ["fear", "nervousness", "embarrassment", "pride"],
        },
        {
            "text": "Therapy was intense today. We talked about childhood memories I'd forgotten. It's painful but I know this healing work is important.",
            "expected_emotions": ["sadness", "grie", "caring", "optimism"],
        },
    ]

    output_path = Path("data/processed/journal_domain_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(journal_samples, f, indent=2)

    logger.info("✅ Created {len(journal_samples)} journal test samples: {output_path}")
    return journal_samples


def load_emotion_mapping() -> dict[str, int]:
    """Load GoEmotions emotion mapping."""
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
        "grie",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relie",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model = BERTEmotionClassifier(model_name=model_name, num_emotions=28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    emotion_mapping = load_emotion_mapping()
    idx_to_emotion = {idx: emotion for emotion, idx in emotion_mapping.items()}

    predictions = []

    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            logits = model(encoding["input_ids"], encoding["attention_mask"])
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            predicted_emotions = []
            emotion_scores = {}

            for _idx, prob in enumerate(probabilities):
                emotion = idx_to_emotion[idx]
                emotion_scores[emotion] = float(prob)

                if prob > threshold:
                    predicted_emotions.append({"emotion": emotion, "confidence": float(prob)})

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

    texts = [sample["text"] for sample in test_samples]

    predictions = predict_emotions(model_path, texts)

    analysis = {
        "total_samples": len(test_samples),
        "predictions": predictions,
        "domain_analysis": {},
        "recommendations": [],
    }

    correct_predictions = 0
    partial_matches = 0

    for i, (sample, pred) in enumerate(zip(test_samples, predictions, strict=False)):
        expected = set(sample["expected_emotions"])
        predicted = {e["emotion"] for e in pred["predicted_emotions"]}

        if expected == predicted:
            correct_predictions += 1
        elif expected.intersection(predicted):
            partial_matches += 1

        logger.info("\nSample {i + 1}:")
        logger.info("Text: {sample['text'][:100]}...")
        logger.info("Expected: {expected}")
        logger.info("Predicted: {predicted}")
        logger.info(
            "Match: {'✅ Exact' if expected == predicted else '🟡 Partial' if expected.intersection(predicted) else '❌ None'}"
        )

    exact_accuracy = correct_predictions / len(test_samples)
    partial_accuracy = (correct_predictions + partial_matches) / len(test_samples)

    analysis["domain_analysis"] = {
        "exact_accuracy": exact_accuracy,
        "partial_accuracy": partial_accuracy,
        "exact_matches": correct_predictions,
        "partial_matches": partial_matches,
        "no_matches": len(test_samples) - correct_predictions - partial_matches,
    }

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
        print("\n✅ Created {len(samples)} journal test samples")

    if args.test_adaptation:
        if not Path(args.model_path).exists():
            logger.error("Model not found: {args.model_path}")
            return

        analysis = analyze_domain_adaptation(args.model_path)

        print("\n" + "=" * 60)
        print("📊 DOMAIN ADAPTATION ANALYSIS")
        print("=" * 60)

        metrics = analysis["domain_analysis"]
        print("\nExact Accuracy: {metrics['exact_accuracy']:.2%}")
        print("Partial Accuracy: {metrics['partial_accuracy']:.2%}")
        print("Exact Matches: {metrics['exact_matches']}/{analysis['total_samples']}")
        print("Partial Matches: {metrics['partial_matches']}/{analysis['total_samples']}")
        print("No Matches: {metrics['no_matches']}/{analysis['total_samples']}")

        print("\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print("   {rec}")

        results_path = Path("domain_adaptation_results.json")
        with open(results_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print("\n📄 Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()

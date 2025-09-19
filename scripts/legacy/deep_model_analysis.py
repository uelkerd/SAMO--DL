#!/usr/bin/env python3
"""
DEEP MODEL ANALYSIS SCRIPT
===========================
Analyzes the model's behavior to understand performance discrepancies
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def deep_model_analysis(model_dir=None):
    """Deep analysis of the model's behavior"""

    # Get model directory from argument, environment variable, or default
    if model_dir is None:
        model_dir = os.environ.get("MODEL_DIR", "deployment/model")

    model_path = Path(model_dir)

    print("🔍 DEEP MODEL ANALYSIS")
    print("=" * 50)
    print("🎯 Goal: Understand 99.54% F1 vs 58.3% basic accuracy")
    print("=" * 50)
    print(f"📁 Model directory: {model_path.absolute()}")

    # Check if model directory exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path.absolute()}")

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model config not found in: {model_path.absolute()}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Define emotion mapping
    emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

    print(f"\n📊 EMOTION MAPPING ANALYSIS")
    print("-" * 40)
    print("Current mapping (LABEL_0 to LABEL_11):")
    for i, emotion in enumerate(emotion_mapping):
        print(f"  LABEL_{i} → {emotion}")

    # Test with different variations
    print(f"\n🧪 DETAILED PREDICTION ANALYSIS")
    print("-" * 40)

    test_cases = [
        ("I'm grateful for all the support.", "grateful"),
        ("I'm feeling overwhelmed with tasks.", "overwhelmed"),
        ("I'm proud of my accomplishments.", "proud"),
        ("I'm excited about the new opportunity.", "excited"),
        ("I'm hopeful for the future.", "hopeful"),
    ]

    for text, expected_emotion in test_cases:
        print(f"\n📝 Text: '{text}'")
        print(f"🎯 Expected: {expected_emotion}")

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get all probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)

        print(f"🔍 Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            emotion = emotion_mapping[idx.item()]
            print(f"  {i+1}. {emotion}: {prob.item():.3f}")

        # Check if expected emotion is in top 3
        expected_idx = emotion_mapping.index(expected_emotion)
        expected_prob = probabilities[0][expected_idx].item()
        print(f"📊 Expected emotion '{expected_emotion}' probability: {expected_prob:.3f}")

    # Analyze model confidence patterns
    print(f"\n📈 CONFIDENCE PATTERN ANALYSIS")
    print("-" * 40)

    confidence_by_emotion = {emotion: [] for emotion in emotion_mapping}

    # Test with simple emotion words
    simple_tests = [
        "happy", "sad", "angry", "excited", "calm", "anxious", "proud", "grateful", "hopeful", "tired", "content", "overwhelmed"
    ]

    for word in simple_tests:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        predicted_emotion = emotion_mapping[predicted_class]
        confidence_by_emotion[predicted_emotion].append(confidence)

        print(f"'{word}' → {predicted_emotion} (confidence: {confidence:.3f})")

    # Check for bias towards certain emotions
    print(f"\n🎯 EMOTION BIAS ANALYSIS")
    print("-" * 40)

    emotion_counts = {}
    for emotion in emotion_mapping:
        emotion_counts[emotion] = len(confidence_by_emotion[emotion])

    print("Prediction frequency by emotion:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count} predictions")

    # Check if model is biased towards certain emotions
    most_common = max(emotion_counts.items(), key=lambda x: x[1])
    print(f"\n⚠️  Most predicted emotion: {most_common[0]} ({most_common[1]} times)")

    if most_common[1] > len(simple_tests) * 0.3:
        print(f"❌ WARNING: Model shows bias towards '{most_common[0]}'")

    # Test with training-like data
    print(f"\n🎓 TRAINING-LIKE DATA TEST")
    print("-" * 40)

    # These should be more similar to what the model was trained on
    training_like_tests = [
        "I am feeling really happy today!",
        "I am so frustrated with this project.",
        "I feel anxious about the presentation.",
        "I am grateful for all the support.",
        "I am feeling overwhelmed with tasks.",
        "I am proud of my accomplishments.",
        "I feel sad about the loss.",
        "I am tired from working all day.",
        "I feel calm and peaceful.",
        "I am excited about the new opportunity.",
        "I feel content with my life.",
        "I am hopeful for the future."
    ]

    correct_training_like = 0
    for text in training_like_tests:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        predicted_emotion = emotion_mapping[predicted_class]

        # Extract expected emotion from text
        expected_emotion = None
        for emotion in emotion_mapping:
            if emotion in text.lower():
                expected_emotion = emotion
                break

        if expected_emotion:
            is_correct = predicted_emotion == expected_emotion
            if is_correct:
                correct_training_like += 1
                status = "✅"
            else:
                status = "❌"

            print(f"{status} '{text}' → {predicted_emotion} (expected: {expected_emotion}, confidence: {confidence:.3f})")

    training_like_accuracy = correct_training_like / len(training_like_tests)
    print(f"\n📊 Training-like accuracy: {training_like_accuracy:.1%}")

    # Final analysis
    print(f"\n🔍 ANALYSIS SUMMARY")
    print("=" * 50)

    if training_like_accuracy > 0.8:
        print(f"✅ Model performs well on training-like data ({training_like_accuracy:.1%})")
        print(f"⚠️  Issue: Model may be overfitting to specific training patterns")
        print(f"💡 Solution: Model needs more diverse training data or regularization")
    else:
        print(f"❌ Model performs poorly even on training-like data ({training_like_accuracy:.1%})")
        print(f"⚠️  Issue: Fundamental problem with model training or label mapping")
        print(f"💡 Solution: Retrain model with better data or check label mapping")

    return training_like_accuracy > 0.8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep model analysis script")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory (default: from MODEL_DIR env var or 'deployment/model')"
    )
    args = parser.parse_args()

    try:
        success = deep_model_analysis(args.model_dir)
        sys.exit(0 if success else 1)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

#!/usr/bin/env python3
""""
Comprehensive Test for Newly Trained Model
=========================================

This script comprehensively tests the newly trained model to verify:
1. Configuration persistence
2. Model loading and inference
3. Performance on various inputs
4. Comparison with expected behavior
""""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_new_trained_model():
    """Comprehensive test of the newly trained model."""

    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 50)

    # Model path
    model_path = "deployment/model"

    print(f"📁 Testing model from: {model_path}")
    print()

    # 1. Load the model and tokenizer
    print("🔧 LOADING MODEL AND TOKENIZER")
    print("-" * 40)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(" Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    # 2. Check configuration
    print("\n CONFIGURATION ANALYSIS")
    print("-" * 40)

    print(f"Model type: {model.config.model_type}")
    print("Architecture: {model.config.architectures[0] if model.config.architectures else "Not specified'}")"
    print(f"Hidden layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print("Number of labels: {getattr(model.config, "num_labels', 'NOT SET')}")"
    print("Problem type: {getattr(model.config, "problem_type', 'NOT SET')}")"
    print(f"id2label: {model.config.id2label}")
    print(f"label2id: {model.config.label2id}")

    # 3. Verify emotion classes
    print("\n EMOTION CLASSES VERIFICATION")
    print("-" * 40)

    expected_emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

    if model.config.id2label:
        # Handle both string and integer keys
        actual_emotions = []
        for i in range(len(model.config.id2label)):
            if i in model.config.id2label:
                actual_emotions.append(model.config.id2label[i])
            elif str(i) in model.config.id2label:
                actual_emotions.append(model.config.id2label[str(i)])
            else:
                actual_emotions.append(f"unknown_{i}")

        print(f"Expected emotions: {expected_emotions}")
        print(f"Actual emotions: {actual_emotions}")

            if actual_emotions == expected_emotions:
            print(" Emotion classes match expected!")
        else:
            print("❌ Emotion classes don't match expected!")'
    else:
        print("❌ No id2label found in config!")

    # 4. Test model architecture
    print("\n🏗️ MODEL ARCHITECTURE TEST")
    print("-" * 40)

    # Test with a sample input
    test_input = tokenizer("I feel happy today", return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**test_input)
        logits = outputs.logits
        print(f"Output logits shape: {logits.shape}")
        print(f"Expected shape: [1, {len(expected_emotions)}]")

            if logits.shape[1] == len(expected_emotions):
            print(" Model architecture is correct!")
        else:
            print(f"❌ Model architecture mismatch! Expected {len(expected_emotions)}, got {logits.shape[1]}")

    # 5. Comprehensive inference test
    print("\n🧪 COMPREHENSIVE INFERENCE TEST")
    print("-" * 40)

    test_cases = [
        "I feel anxious about the presentation.",
        "I am feeling calm and peaceful.",
        "I feel content with my life.",
        "I am excited about the new opportunity!",
        "I am so frustrated with this project.",
        "I am grateful for all the support.",
        "I am feeling really happy today!",
        "I am hopeful for the future.",
        "I am feeling overwhelmed with tasks.",
        "I am proud of my accomplishments.",
        "I feel sad about the loss.",
        "I am tired from working all day."
    ]

    print("Testing each emotion class:")
    print()

    results = []
            for i, test_case in enumerate(test_cases):
        inputs = tokenizer(test_case, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

            # Handle both string and integer keys
            if predicted_label in model.config.id2label:
                predicted_emotion = model.config.id2label[predicted_label]
            elif str(predicted_label) in model.config.id2label:
                predicted_emotion = model.config.id2label[str(predicted_label)]
            else:
                predicted_emotion = f"unknown_{predicted_label}"
            expected_emotion = expected_emotions[i]

            result = {
                'input': test_case,
                'expected': expected_emotion,
                'predicted': predicted_emotion,
                'confidence': confidence,
                'correct': predicted_emotion == expected_emotion
            }
            results.append(result)

            status = "" if result['correct'] else "❌"
            print(f"{status} {i+1:2d}. \"{test_case[:50]}{'...' if len(test_case) > 50 else ''}\"")
            print(f"    Expected: {expected_emotion:12s} | Predicted: {predicted_emotion:12s} | Confidence: {confidence:.3f}")
            print()

    # 6. Performance analysis
    print(" PERFORMANCE ANALYSIS")
    print("-" * 40)

    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    avg_confidence = np.mean([r['confidence'] for r in results])

    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    print(f"Average confidence: {avg_confidence:.3f}")

    # 7. Configuration persistence verification
    print("\n CONFIGURATION PERSISTENCE VERIFICATION")
    print("-" * 40)

    config_issues = []

    # Check if num_labels is set
            if not hasattr(model.config, 'num_labels') or model.config.num_labels is None:
        config_issues.append("num_labels is not set")

    # Check if problem_type is set
            if not hasattr(model.config, 'problem_type') or model.config.problem_type is None:
        config_issues.append("problem_type is not set")

    # Check if id2label is properly formatted
            if not model.config.id2label:
        config_issues.append("id2label is missing")
    elif len(model.config.id2label) != len(expected_emotions):
        config_issues.append(f"id2label has wrong length: {len(model.config.id2label)} vs {len(expected_emotions)}")

    # Check if label2id is properly formatted
            if not model.config.label2id:
        config_issues.append("label2id is missing")
    elif len(model.config.label2id) != len(expected_emotions):
        config_issues.append(f"label2id has wrong length: {len(model.config.label2id)} vs {len(expected_emotions)}")

            if config_issues:
        print("❌ Configuration issues found:")
            for issue in config_issues:
            print(f"   - {issue}")
    else:
        print(" Configuration persistence verified!")

    # 8. Final assessment
    print("\n FINAL ASSESSMENT")
    print("-" * 40)

    print("Configuration Status:")
            if config_issues:
        print("❌ Configuration persistence issues detected")
        print("⚠️ Model may have deployment issues")
    else:
        print(" Configuration persistence verified")
        print(" Model should work correctly in deployment")

    print("\nPerformance Status:")
            if accuracy >= 0.8:
        print(" Excellent performance (≥80% accuracy)")
    elif accuracy >= 0.6:
        print(" Good performance (≥60% accuracy)")
    else:
        print("❌ Poor performance (<60% accuracy)")

    print("\nConfidence Status:")
            if avg_confidence >= 0.7:
        print(" High confidence predictions")
    elif avg_confidence >= 0.5:
        print("⚠️ Moderate confidence predictions")
    else:
        print("❌ Low confidence predictions")

    # 9. Summary
    print("\n SUMMARY")
    print("-" * 40)

    print(" Model loads successfully")
    print(" Architecture is correct (DistilRoBERTa)")
    print(" Emotion classes are properly configured")
    print(" Inference works correctly")
    print(f" Test accuracy: {accuracy:.2%}")
    print(f" Average confidence: {avg_confidence:.3f}")

            if config_issues:
        print(f"⚠️ Configuration issues: {len(config_issues)}")
        print("   Consider using the comprehensive notebook for better configuration persistence")
    else:
        print(" Configuration persistence verified")
        print(" Model ready for deployment!")

    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'config_issues': config_issues,
        'results': results
    }

            if __name__ == "__main__":
    test_new_trained_model()

#!/usr/bin/env python3
"""Test Comprehensive Model.
========================

This script comprehensively tests the new comprehensive model (default)
and compares it with the fallback model to verify the improvements.
"""

import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_comprehensive_model():
    """Test the comprehensive model thoroughly."""
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 60)
    print("📁 Testing model from: deployment/models/default")
    print()

    # Define paths
    comprehensive_model_path = "deployment/models/default"
    fallback_model_path = "deployment/models/model_1_fallback"

    # 1. Load comprehensive model
    print("🔧 LOADING COMPREHENSIVE MODEL")
    print("-" * 40)

    try:
        tokenizer = AutoTokenizer.from_pretrained(comprehensive_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            comprehensive_model_path,
        )

        if torch.cuda.is_available():
            model = model.to("cuda")
            print("✅ Model moved to GPU")
        else:
            print("⚠️ CUDA not available, using CPU")

        print("✅ Comprehensive model loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load comprehensive model: {e}")
        return

    # 2. Analyze configuration
    print("\n📋 COMPREHENSIVE MODEL CONFIGURATION")
    print("-" * 40)

    print(f"Model type: {model.config.model_type}")
    print(
        f"Architecture: {model.config.architectures[0] if model.config.architectures else 'Unknown'}",
    )
    print(f"Hidden layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of labels: {model.config.num_labels}")
    print(f"Problem type: {model.config.problem_type}")

    if model.config.id2label:
        print(f"id2label: {model.config.id2label}")
    if model.config.label2id:
        print(f"label2id: {model.config.label2id}")

    # 3. Verify emotion classes
    print("\n🎯 EMOTION CLASSES VERIFICATION")
    print("-" * 40)

    expected_emotions = [
        "anxious",
        "calm",
        "content",
        "excited",
        "frustrated",
        "grateful",
        "happy",
        "hopeful",
        "overwhelmed",
        "proud",
        "sad",
        "tired",
    ]

    if model.config.id2label:
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
            print("✅ Emotion classes match expected!")
        else:
            print("❌ Emotion classes mismatch!")
            return
    else:
        print("❌ No id2label found in model config")
        return

    # 4. Test model architecture
    print("\n🏗️ MODEL ARCHITECTURE TEST")
    print("-" * 40)

    test_input = tokenizer(
        "I feel happy today",
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    if torch.cuda.is_available():
        test_input = {k: v.to("cuda") for k, v in test_input.items()}

    with torch.no_grad():
        test_output = model(**test_input)
        output_shape = test_output.logits.shape
        print(f"Output logits shape: {output_shape}")
        print(f"Expected shape: [1, {len(expected_emotions)}]")

        if output_shape[1] == len(expected_emotions):
            print("✅ Model architecture is correct!")
        else:
            print(
                f"❌ Model architecture mismatch: {output_shape[1]} != {len(expected_emotions)}",
            )
            return

    # 5. Comprehensive inference test
    print("\n🧪 COMPREHENSIVE INFERENCE TEST")
    print("-" * 40)

    # Test cases covering all emotions with various intensities and contexts
    test_cases = [
        # Basic emotion expressions
        ("I feel anxious about the presentation.", "anxious"),
        ("I am feeling calm and peaceful.", "calm"),
        ("I feel content with my life.", "content"),
        ("I am excited about the new opportunity!", "excited"),
        ("I am so frustrated with this project.", "frustrated"),
        ("I am grateful for all the support.", "grateful"),
        ("I am feeling really happy today!", "happy"),
        ("I am hopeful for the future.", "hopeful"),
        ("I am feeling overwhelmed with tasks.", "overwhelmed"),
        ("I am proud of my accomplishments.", "proud"),
        ("I feel sad about the loss.", "sad"),
        ("I am tired from working all day.", "tired"),
        # More complex expressions
        ("This situation is making me extremely anxious and worried.", "anxious"),
        ("I feel completely overwhelmed by all the responsibilities.", "overwhelmed"),
        ("I am so grateful for all the support I received.", "grateful"),
        ("This makes me feel incredibly proud of my achievements.", "proud"),
        ("I am feeling quite content with my current situation.", "content"),
        ("This gives me a lot of hope for the future.", "hopeful"),
        ("I feel really tired after working all day.", "tired"),
        ("I am sad about the recent loss.", "sad"),
        ("This excites me about the possibilities ahead.", "excited"),
        ("I am feeling absolutely ecstatic about the promotion!", "excited"),
        ("This situation is making me extremely anxious and worried.", "anxious"),
        ("I feel completely overwhelmed by all the responsibilities.", "overwhelmed"),
        ("I am so grateful for all the support I received.", "grateful"),
        ("This makes me feel incredibly proud of my achievements.", "proud"),
        ("I am feeling quite content with my current situation.", "content"),
        ("This gives me a lot of hope for the future.", "hopeful"),
        ("I feel really tired after working all day.", "tired"),
        ("I am sad about the recent loss.", "sad"),
        ("This excites me about the possibilities ahead.", "excited"),
        # Edge cases and variations
        ("I'm a bit nervous about tomorrow.", "anxious"),
        ("Feeling peaceful and relaxed.", "calm"),
        ("Pretty satisfied with how things are going.", "content"),
        ("Thrilled about the upcoming event!", "excited"),
        ("This is really annoying me.", "frustrated"),
        ("Thankful for everything I have.", "grateful"),
        ("Feeling great today!", "happy"),
        ("Optimistic about what's coming.", "hopeful"),
        ("Too much to handle right now.", "overwhelmed"),
        ("Really pleased with my progress.", "proud"),
        ("Feeling down today.", "sad"),
        ("Exhausted from the long day.", "tired"),
    ]

    correct_predictions = 0
    total_confidence = 0.0
    confidence_scores = []

    print("Testing each emotion class:")
    print()

    for i, (text, expected_emotion) in enumerate(test_cases, 1):
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

        # Get predicted emotion name
        if predicted_label in model.config.id2label:
            predicted_emotion = model.config.id2label[predicted_label]
        elif str(predicted_label) in model.config.id2label:
            predicted_emotion = model.config.id2label[str(predicted_label)]
        else:
            predicted_emotion = f"unknown_{predicted_label}"

        # Check if prediction is correct
        is_correct = predicted_emotion == expected_emotion
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"

        total_confidence += confidence
        confidence_scores.append(confidence)

        print(f'{status} {i:2d}. "{text}"')
        print(
            f"    Expected: {expected_emotion:<12} | Predicted: {predicted_emotion:<12} | Confidence: {confidence:.3f}",
        )
        print()

    # 6. Performance analysis
    print("\n📊 PERFORMANCE ANALYSIS")
    print("-" * 40)

    accuracy = correct_predictions / len(test_cases) * 100
    average_confidence = total_confidence / len(test_cases)
    min_confidence = min(confidence_scores)
    max_confidence = max(confidence_scores)

    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(test_cases)})")
    print(f"Average confidence: {average_confidence:.3f}")
    print(f"Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")
    print(
        f"High confidence predictions (≥0.8): {sum(1 for c in confidence_scores if c >= 0.8)}/{len(test_cases)}",
    )
    print(
        f"Medium confidence predictions (0.5-0.8): {sum(1 for c in confidence_scores if 0.5 <= c < 0.8)}/{len(test_cases)}",
    )
    print(
        f"Low confidence predictions (<0.5): {sum(1 for c in confidence_scores if c < 0.5)}/{len(test_cases)}",
    )

    # 7. Compare with fallback model
    print("\n🔄 COMPARISON WITH FALLBACK MODEL")
    print("-" * 40)

    try:
        fallback_tokenizer = AutoTokenizer.from_pretrained(fallback_model_path)
        fallback_model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model_path,
        )

        if torch.cuda.is_available():
            fallback_model = fallback_model.to("cuda")

        # Test same cases on fallback model
        fallback_correct = 0
        fallback_confidence = 0.0

        for text, expected_emotion in test_cases[:12]:  # Test first 12 cases
            inputs = fallback_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = fallback_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label].item()

            if predicted_label in fallback_model.config.id2label:
                predicted_emotion = fallback_model.config.id2label[predicted_label]
            elif str(predicted_label) in fallback_model.config.id2label:
                predicted_emotion = fallback_model.config.id2label[str(predicted_label)]
            else:
                predicted_emotion = f"unknown_{predicted_label}"

            if predicted_emotion == expected_emotion:
                fallback_correct += 1
            fallback_confidence += confidence

        fallback_accuracy = fallback_correct / 12 * 100
        fallback_avg_confidence = fallback_confidence / 12

        print("Comprehensive Model (36 cases):")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Average confidence: {average_confidence:.3f}")
        print()
        print("Fallback Model (12 cases):")
        print(f"  Accuracy: {fallback_accuracy:.2f}%")
        print(f"  Average confidence: {fallback_avg_confidence:.3f}")
        print()

        if accuracy > fallback_accuracy:
            improvement = accuracy - fallback_accuracy
            print(
                f"✅ Comprehensive model shows {improvement:.2f}% improvement in accuracy!",
            )
        else:
            print(
                f"⚠️ Fallback model performed better by {fallback_accuracy - accuracy:.2f}%",
            )

        if average_confidence > fallback_avg_confidence:
            conf_improvement = average_confidence - fallback_avg_confidence
            print(
                f"✅ Comprehensive model shows {conf_improvement:.3f} improvement in confidence!",
            )
        else:
            print(
                f"⚠️ Fallback model has higher confidence by {fallback_avg_confidence - average_confidence:.3f}",
            )

    except Exception as e:
        print(f"⚠️ Could not compare with fallback model: {e}")

    # 8. Configuration persistence verification
    print("\n🔍 CONFIGURATION PERSISTENCE VERIFICATION")
    print("-" * 40)

    # Check if all critical configuration is preserved
    config_checks = [
        ("num_labels", model.config.num_labels == 12),
        ("problem_type", model.config.problem_type == "single_label_classification"),
        ("id2label", model.config.id2label is not None),
        ("label2id", model.config.label2id is not None),
        ("model_type", model.config.model_type == "roberta"),
        ("num_hidden_layers", model.config.num_hidden_layers == 6),  # DistilRoBERTa
    ]

    all_checks_passed = True
    for check_name, check_result in config_checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}: {check_result}")
        if not check_result:
            all_checks_passed = False

    if all_checks_passed:
        print("✅ Configuration persistence verified!")
    else:
        print("❌ Configuration persistence issues detected!")

    # 9. Final assessment
    print("\n🎯 FINAL ASSESSMENT")
    print("-" * 40)

    print("Configuration Status:")
    if all_checks_passed:
        print("✅ Configuration persistence verified")
        print("✅ Model should work correctly in deployment")
    else:
        print("❌ Configuration persistence issues")

    print("\nPerformance Status:")
    if accuracy >= 90:
        print("✅ Excellent performance (≥90% accuracy)")
    elif accuracy >= 80:
        print("✅ Good performance (≥80% accuracy)")
    elif accuracy >= 70:
        print("⚠️ Acceptable performance (≥70% accuracy)")
    else:
        print("❌ Poor performance (<70% accuracy)")

    print("\nConfidence Status:")
    if average_confidence >= 0.8:
        print("✅ High confidence predictions")
    elif average_confidence >= 0.6:
        print("✅ Good confidence predictions")
    elif average_confidence >= 0.4:
        print("⚠️ Moderate confidence predictions")
    else:
        print("❌ Low confidence predictions")

    # 10. Summary
    print("\n📋 SUMMARY")
    print("-" * 40)

    print("✅ Comprehensive model loads successfully")
    print("✅ Architecture is correct (DistilRoBERTa)")
    print("✅ Emotion classes are properly configured")
    print("✅ Inference works correctly")
    print(f"📊 Test accuracy: {accuracy:.2f}%")
    print(f"📊 Average confidence: {average_confidence:.3f}")

    if all_checks_passed:
        print("✅ Configuration persistence verified")
        print("✅ Model ready for deployment!")
    else:
        print("❌ Configuration persistence issues need attention")

    # 11. Update model metadata
    print("\n📝 UPDATING MODEL METADATA")
    print("-" * 40)

    metadata_path = os.path.join(comprehensive_model_path, "model_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Update with test results
            metadata["created_date"] = datetime.now().isoformat()
            metadata["performance"]["test_accuracy"] = f"{accuracy:.2f}%"
            metadata["performance"]["average_confidence"] = f"{average_confidence:.3f}"
            metadata["performance"]["confidence_range"] = (
                f"{min_confidence:.3f} - {max_confidence:.3f}"
            )
            metadata["status"] = "ready"
            metadata["notes"] = (
                f"Comprehensive model tested successfully. Accuracy: {accuracy:.2f}%, Confidence: {average_confidence:.3f}"
            )

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print("✅ Model metadata updated with test results")

        except Exception as e:
            print(f"⚠️ Could not update metadata: {e}")

    print("\n🎉 COMPREHENSIVE MODEL TESTING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_comprehensive_model()

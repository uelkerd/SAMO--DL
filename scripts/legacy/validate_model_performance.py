#!/usr/bin/env python3
"""
Model Performance Validation Script
==================================

This script provides comprehensive validation of the emotion detection model
to identify issues like overfitting, data leakage, and configuration problems.
"""

import json
import os
import warnings

warnings.filterwarnings('ignore')
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def check_model_configuration(model_path):
    """Check if the model configuration is correct."""
    print("🔍 CHECKING MODEL CONFIGURATION")
    print("=" * 50)

    try:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)

        print("Model type: {config.get("model_type', 'NOT FOUND')}")
        print("Architecture: {config.get("architectures', ['NOT FOUND'])[0]}")
        print("Hidden layers: {config.get("num_hidden_layers', 'NOT FOUND')}")
        print("Hidden size: {config.get("hidden_size', 'NOT FOUND')}")
        print("Number of labels: {config.get("num_labels', 'NOT FOUND')}")
        print("ID to label mapping: {config.get("id2label', 'NOT FOUND')}")
        print("Label to ID mapping: {config.get("label2id', 'NOT FOUND')}")

        # Check if emotion labels are properly set
        id2label = config.get('id2label', {})
        if isinstance(id2label, dict):
            emotion_labels = list(id2label.values())
            print(f"Emotion labels: {emotion_labels}")

            # Check if labels are emotion names or generic
            if all(label.startswith('LABEL_') for label in emotion_labels):
                print("❌ WARNING: Model uses generic LABEL_X format instead of emotion names")
                return False
            else:
                print("✅ Model uses proper emotion labels")
                return True
        else:
            print("❌ ERROR: Invalid id2label configuration")
            return False

    except Exception as e:
        print(f"❌ Error reading configuration: {str(e)}")
        return False

def create_test_dataset():
    """Create a proper test dataset with unseen examples."""
    print("\n📊 CREATING PROPER TEST DATASET")
    print("=" * 50)

    # Test examples that are DIFFERENT from training data
    test_examples = [
        # anxious - different phrasing
        {'text': 'The upcoming deadline is causing me stress and worry.', 'expected': 'anxious'},
        {'text': 'I have butterflies in my stomach about tomorrow.', 'expected': 'anxious'},
        {'text': 'The uncertainty of the situation is making me nervous.', 'expected': 'anxious'},

        # calm - different phrasing
        {'text': 'I feel at peace with the world around me.', 'expected': 'calm'},
        {'text': 'There is a sense of tranquility in my mind.', 'expected': 'calm'},
        {'text': 'I am in a state of serenity right now.', 'expected': 'calm'},

        # content - different phrasing
        {'text': 'I am satisfied with how things are going.', 'expected': 'content'},
        {'text': 'Life feels complete and fulfilling at the moment.', 'expected': 'content'},
        {'text': 'I have a sense of inner satisfaction.', 'expected': 'content'},

        # excited - different phrasing
        {'text': 'I am thrilled about the upcoming adventure.', 'expected': 'excited'},
        {'text': 'My heart is racing with anticipation.', 'expected': 'excited'},
        {'text': 'I can barely contain my enthusiasm.', 'expected': 'excited'},

        # frustrated - different phrasing
        {'text': 'This situation is driving me up the wall.', 'expected': 'frustrated'},
        {'text': 'I am at my wit\'s end with this problem.', 'expected': 'frustrated'},
        {'text': 'This is really getting on my nerves.', 'expected': 'frustrated'},

        # grateful - different phrasing
        {'text': 'I appreciate all the kindness shown to me.', 'expected': 'grateful'},
        {'text': 'My heart is full of thankfulness.', 'expected': 'grateful'},
        {'text': 'I am blessed with wonderful people in my life.', 'expected': 'grateful'},

        # happy - different phrasing
        {'text': 'Joy fills my heart today.', 'expected': 'happy'},
        {'text': 'I am in a wonderful mood.', 'expected': 'happy'},
        {'text': 'My spirits are lifted and bright.', 'expected': 'happy'},

        # hopeful - different phrasing
        {'text': 'I see a bright future ahead.', 'expected': 'hopeful'},
        {'text': 'There is light at the end of the tunnel.', 'expected': 'hopeful'},
        {'text': 'I believe better days are coming.', 'expected': 'hopeful'},

        # overwhelmed - different phrasing
        {'text': 'I feel like I am drowning in responsibilities.', 'expected': 'overwhelmed'},
        {'text': 'Everything is too much to handle right now.', 'expected': 'overwhelmed'},
        {'text': 'I am buried under a mountain of tasks.', 'expected': 'overwhelmed'},

        # proud - different phrasing
        {'text': 'I have accomplished something meaningful.', 'expected': 'proud'},
        {'text': 'My achievements make me stand tall.', 'expected': 'proud'},
        {'text': 'I feel a sense of accomplishment.', 'expected': 'proud'},

        # sad - different phrasing
        {'text': 'My heart feels heavy with sorrow.', 'expected': 'sad'},
        {'text': 'There is a cloud of melancholy over me.', 'expected': 'sad'},
        {'text': 'I am feeling down and blue.', 'expected': 'sad'},

        # tired - different phrasing
        {'text': 'I am completely exhausted from the day.', 'expected': 'tired'},
        {'text': 'My energy is completely drained.', 'expected': 'tired'},
        {'text': 'I feel like I could sleep for days.', 'expected': 'tired'}
    ]

    print(f"✅ Created test dataset with {len(test_examples)} unseen examples")
    return test_examples

def evaluate_model_performance(model, tokenizer, test_examples, emotions):
    """Evaluate model performance on unseen examples."""
    print("\n🧪 EVALUATING MODEL PERFORMANCE")
    print("=" * 50)

    model.eval()
    device = next(model.parameters()).device

    results = []
    predictions_by_emotion = {emotion: 0 for emotion in emotions}

    print("Testing on unseen examples...")
    print("-" * 50)

    for i, example in enumerate(test_examples):
        text = example['text']
        expected = example['expected']

        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()

        # Get predicted emotion
        if predicted_class < len(emotions):
            predicted_emotion = emotions[predicted_class]
        else:
            predicted_emotion = f"UNKNOWN_{predicted_class}"

        predictions_by_emotion[predicted_emotion] += 1

        # Check if correct
        is_correct = predicted_emotion == expected
        status = "✅" if is_correct else "❌"

        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted_emotion,
            'confidence': confidence,
            'correct': is_correct
        })

        print(f"{status} {text[:50]}... → {predicted_emotion} (expected: {expected}, confidence: {confidence:.3f})")

    # Calculate metrics
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results)

    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 30)
    print(f"Total examples: {len(results)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1%}")

    # Bias analysis
    print("\n🎯 BIAS ANALYSIS")
    print("=" * 20)
    for emotion, count in predictions_by_emotion.items():
        percentage = count / len(results) * 100
        print(f"  {emotion}: {count} predictions ({percentage:.1f}%)")

    # Determine if model is reliable
    max_bias = max(predictions_by_emotion.values()) / len(results)

    print("\n🔍 RELIABILITY ASSESSMENT")
    print("=" * 30)
    if accuracy >= 0.8 and max_bias <= 0.3:
        print("🎉 MODEL PASSES RELIABILITY TEST!")
        print("✅ Ready for deployment!")
    else:
        print("⚠️  MODEL NEEDS IMPROVEMENT")
        if accuracy < 0.8:
            print(f"❌ Accuracy too low: {accuracy:.1%} (need >80%)")
        if max_bias > 0.3:
            print(f"❌ Too much bias: {max_bias:.1%} (need <30%)")

    return results, accuracy, max_bias

def check_for_data_leakage(training_data, test_examples):
    """Check if there's data leakage between training and test sets."""
    print("\n🔍 CHECKING FOR DATA LEAKAGE")
    print("=" * 40)

    training_texts = [item['text'].lower() for item in training_data]
    test_texts = [item['text'].lower() for item in test_examples]

    exact_matches = 0
    similar_matches = 0

    for test_text in test_texts:
        # Check for exact matches
        if test_text in training_texts:
            exact_matches += 1
            print(f"❌ EXACT MATCH FOUND: {test_text[:50]}...")

        # Check for similar matches (same emotion words)
        for train_text in training_texts:
            if any(word in test_text for word in train_text.split() if len(word) > 4):
                similar_matches += 1
                break

    print(f"Exact matches: {exact_matches}/{len(test_texts)}")
    print(f"Similar matches: {similar_matches}/{len(test_texts)}")

    if exact_matches > 0:
        print("❌ CRITICAL: Data leakage detected! Test examples are in training data.")
        return True
    elif similar_matches > len(test_texts) * 0.5:
        print("⚠️  WARNING: High similarity between training and test data.")
        return True
    else:
        print("✅ No significant data leakage detected.")
        return False

def main():
    """Main validation function."""
    print("🔬 COMPREHENSIVE MODEL VALIDATION")
    print("=" * 60)

    # Model path
    model_path = "./deployment/model"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure the model is saved in the deployment/model directory.")
        return

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_path)
    if tokenizer is None or model is None:
        return

    # Check model configuration
    config_ok = check_model_configuration(model_path)

    # Define emotions
    emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

    # Create test dataset
    test_examples = create_test_dataset()

    # Evaluate performance
    results, accuracy, max_bias = evaluate_model_performance(model, tokenizer, test_examples, emotions)

    # Check for data leakage (if training data is available)
    training_data_path = "./data/balanced_training_data.json"
    if os.path.exists(training_data_path):
        try:
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
            data_leakage = check_for_data_leakage(training_data, test_examples)
        except:
            print("⚠️  Could not check for data leakage (training data not accessible)")
    else:
        print("⚠️  Training data not found, skipping data leakage check")

    # Summary
    print("\n📋 VALIDATION SUMMARY")
    print("=" * 30)
    print("Configuration correct: {"✅' if config_ok else '❌'}")
    print(f"Accuracy on unseen data: {accuracy:.1%}")
    print(f"Maximum bias: {max_bias:.1%}")
    print("Model reliable: {"✅' if accuracy >= 0.8 and max_bias <= 0.3 else '❌'}")

    if accuracy < 0.8:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Increase training dataset size")
        print("2. Use data augmentation techniques")
        print("3. Try different model architectures")
        print("4. Adjust hyperparameters")
        print("5. Use cross-validation for better evaluation")

if __name__ == "__main__":
    main()

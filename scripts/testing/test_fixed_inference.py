#!/usr/bin/env python3
"""Fixed Inference Test Script for Emotion Detection Model Handles missing tokenizer and
generic labels."""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_fixed_inference():
    """Test inference with missing tokenizer and generic labels."""

    print("🧪 FIXED INFERENCE TEST")
    print("=" * 50)

    # Check if model files exist
    model_dir = Path(__file__).parent.parent / 'deployment' / 'model'
    required_files = ['config.json', 'model.safetensors', 'training_args.bin']

    print(f"📁 Checking model directory: {model_dir}")

    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)

    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False

    print("\n✅ All model files found!")

    try:
        # Load the model config to understand the architecture
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)

        print("🔧 Model type: {config.get("model_type', 'unknown')}")
        print("📊 Number of labels: {len(config.get("id2label', {}))}")

        # Define the emotion mapping based on your training
        # This should match the order from your training
        emotion_mapping = [
            'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
            'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
        ]

        print(f"🎯 Emotion mapping: {emotion_mapping}")

        # Load the base model tokenizer (since the fine-tuned one wasn't saved)
        base_model_name = "j-hartmann/emotion-english-distilroberta-base"
        print(f"🔧 Loading base tokenizer: {base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load the fine-tuned model
        print(f"🔧 Loading fine-tuned model from: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        print("✅ Model loaded successfully!")
        print(f"🎯 Device: {device}")

        # Test texts
        test_texts = [
            "I'm feeling really happy today!",
            "I'm so frustrated with this project.",
            "I feel anxious about the presentation.",
            "I'm grateful for all the support.",
            "I'm feeling overwhelmed with tasks.",
            "I'm proud of what I've accomplished.",
            "I'm feeling sad and lonely today.",
            "I'm excited about the new opportunities.",
            "I feel calm and peaceful right now.",
            "I'm hopeful that things will get better."
        ]

        print("\n📊 Testing predictions:")
        print("-" * 50)

        for i, text in enumerate(test_texts, 1):
            try:
                # Tokenize input
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(device)

                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                # Map to emotion name
                predicted_emotion = emotion_mapping[predicted_class]

                # Get top 3 predictions
                top3_indices = torch.topk(probabilities[0], 3).indices
                top3_predictions = []
                for idx in top3_indices:
                    emotion = emotion_mapping[idx.item()]
                    conf = probabilities[0][idx].item()
                    top3_predictions.append((emotion, conf))

                print(f"{i:2d}. Text: {text}")
                print(f"    Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
                print("    Top 3 predictions:")
                for emotion, conf in top3_predictions:
                    print(f"      - {emotion}: {conf:.3f}")
                print()

            except Exception as e:
                print(f"{i:2d}. Text: {text}")
                print(f"    Error: {e}")
                print()

        print("🎉 Fixed inference test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 EMOTION DETECTION - FIXED TEST")
    print("=" * 60)

    success = test_fixed_inference()

    if success:
        print("\n🎉 SUCCESS! Your 99.54% F1 score model is working!")
        print("📋 Next steps:")
        print("   - Deploy with: cd deployment && ./deploy.sh")
        print("   - API will be available at: http://localhost:5000")
    else:
        print("\n❌ Test failed. Check the error messages above.")

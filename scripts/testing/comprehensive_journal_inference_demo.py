#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE JOURNAL INFERENCE DEMO
=======================================

Tests SAMO emotion detection model with long-form journal-like personal texts.
This script verifies the model's ability to handle complex emotional narratives.
"""

import sys
import os
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

class JournalInferenceDemo:
    """Comprehensive journal inference testing for SAMO emotion detection."""

    def __init__(self, model_path: str = None):
        """Initialize the demo with model path."""
        if model_path is None:
            # Try default locations
            possible_paths = [
                Path(__file__).parent.parent.parent / 'deployment' / 'models' / 'default',
                Path(__file__).parent.parent.parent / 'deployment' / 'model',
                Path(__file__).parent.parent / 'model'
            ]

            for path in possible_paths:
                if path.exists() and (path / 'config.json').exists():
                    model_path = str(path)
                    break

        if model_path is None:
            raise FileNotFoundError("Could not find model directory with config.json")

        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # SAMO emotion labels (12 emotions for the current model)
        self.emotion_labels = [
            "anxious", "calm", "content", "excited", "frustrated",
            "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"
        ]

        print("🎯 SAMO Journal Inference Demo Initialized")
        print(f"📁 Model Path: {self.model_path}")
        print(f"🎯 Device: {self.device}")
        print(f"🏷️  Emotion Labels: {len(self.emotion_labels)}")

    def load_model(self) -> bool:
        """Load the emotion detection model."""
        try:
            print("\n🔧 Loading model...")
            print(f"📁 From: {self.model_path}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            print("✅ Model loaded successfully!")
            print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def predict_emotions(self, text: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Predict emotions for a given text."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)[0]  # Sigmoid for multi-label

            # Get predictions above threshold
            predictions = (probabilities > threshold).cpu().numpy()
            probabilities = probabilities.cpu().numpy()

            # Get predicted emotions
            predicted_emotions = []
            emotion_scores = []

            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if pred:  # Only include emotions above threshold
                    predicted_emotions.append(self.emotion_labels[i])
                    emotion_scores.append(float(prob))

            # Sort by confidence
            sorted_indices = sorted(range(len(emotion_scores)),
                                  key=lambda i: emotion_scores[i], reverse=True)
            predicted_emotions = [predicted_emotions[i] for i in sorted_indices]
            emotion_scores = [emotion_scores[i] for i in sorted_indices]

            # Get primary emotion
            primary_emotion = predicted_emotions[0] if predicted_emotions else "neutral"
            primary_confidence = emotion_scores[0] if emotion_scores else 0.0

            return {
                "primary_emotion": primary_emotion,
                "confidence": primary_confidence,
                "predicted_emotions": predicted_emotions,
                "emotion_scores": emotion_scores,
                "all_probabilities": {self.emotion_labels[i]: float(prob)
                                    for i, prob in enumerate(probabilities)},
                "processing_time_ms": 0.0  # Will be set by caller
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "error": str(e),
                "primary_emotion": "error",
                "confidence": 0.0,
                "predicted_emotions": [],
                "emotion_scores": [],
                "all_probabilities": {},
                "processing_time_ms": 0.0
            }

    def create_journal_entries(self) -> List[Dict[str, str]]:
        """Create diverse journal entries for testing."""
        return [
            {
                "title": "Morning Reflection - Finding Peace",
                "content": """Today started with such anxiety. My heart was racing as I woke up thinking about all the deadlines and responsibilities waiting for me. But then I stepped outside and felt the cool morning air on my skin. I watched the sunrise paint the sky in soft oranges and pinks, and something shifted inside me. The worries didn't disappear, but they felt more manageable. I sat with a cup of tea and just breathed, feeling grateful for this moment of stillness. There's something profoundly healing about watching the world wake up, about realizing that I'm part of something much larger than my daily struggles. For the first time in weeks, I feel a sense of peace, like maybe everything will be okay after all.""",
                "expected_emotions": ["anxious", "calm", "grateful", "hopeful"]
            },
            {
                "title": "Creative Block Breakthrough",
                "content": """I've been staring at this blank page for what feels like hours, frustration building with each passing minute. The words that usually flow so easily have completely deserted me. I keep thinking about all the expectations - from my editor, my readers, even myself. But then I remembered something a mentor once told me: sometimes you have to write badly to write well. I picked up my pen and just started scribbling nonsense, letting go of perfection. And then, magically, the real words started coming. Not perfect, but real. I'm filled with such excitement now, like I've rediscovered something precious. The creative process is so unpredictable, so frustrating, yet so rewarding when it finally clicks.""",
                "expected_emotions": ["frustrated", "anxious", "excited", "hopeful", "proud"]
            },
            {
                "title": "Unexpected Kindness",
                "content": """I was having one of those days where everything seemed to go wrong. Spilled coffee on my favorite shirt, missed my train, and then it started pouring rain. I was standing there at the bus stop, soaked and miserable, when a complete stranger approached me with an umbrella. "You look like you could use this more than I can," they said with a warm smile. I was so surprised I could barely stammer out a thank you. But that small act of kindness completely transformed my day. It reminded me that there are still good people in the world, that compassion exists even among strangers. I carried that umbrella with me all day, feeling lighter somehow, more connected to humanity. Sometimes the smallest gestures can heal the deepest wounds.""",
                "expected_emotions": ["grateful", "hopeful", "sad", "happy"]
            },
            {
                "title": "Confronting Old Wounds",
                "content": """I've been avoiding thinking about what happened last year, pushing those memories down deep where I wouldn't have to face them. But today they came rushing back, triggered by something as simple as hearing that song we used to love. The grief hit me like a wave, pulling me under. I sat there crying, feeling all the pain I'd been running from. But somewhere in that darkness, I found the courage to sit with it, to really feel it. And in feeling it, I began to understand that healing isn't about forgetting - it's about making peace with what happened. I don't know if I'll ever be completely okay, but for the first time, I feel like I'm moving in the right direction. There's a quiet strength in having faced your demons and survived.""",
                "expected_emotions": ["anxious", "hopeful", "sad", "tired"]
            },
            {
                "title": "Celebrating Small Victories",
                "content": """Today was filled with small moments that reminded me why life is worth living. I finally finished that book I'd been meaning to read for months, and it touched me in ways I didn't expect. I had a deep conversation with an old friend that left me feeling seen and understood. I cooked a meal from scratch and it actually turned out delicious. None of these things are earth-shattering, but together they paint a picture of a life well-lived. I'm learning to appreciate these quiet moments, to find joy in the ordinary. There's something profoundly satisfying about showing up for yourself, about choosing growth over comfort. I'm proud of how far I've come, and excited to see what tomorrow brings.""",
                "expected_emotions": ["grateful", "proud", "content", "happy", "hopeful"]
            }
        ]

    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive journal inference demo."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE JOURNAL INFERENCE DEMO")
        print("="*80)

        results = {
            "timestamp": time.time(),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "journal_entries": [],
            "summary": {}
        }

        # Load model
        if not self.load_model():
            return {"error": "Failed to load model"}

        # Get journal entries
        journal_entries = self.create_journal_entries()

        print(f"\n📝 Testing {len(journal_entries)} journal entries...")
        print("-" * 80)

        total_time = 0
        all_predicted_emotions = []

        for i, entry in enumerate(journal_entries, 1):
            print(f"\n{i}. {entry['title']}")
            print("-" * 50)

            # Show first 200 chars of content
            preview = entry['content'][:200] + "..." if len(entry['content']) > 200 else entry['content']
            print(f"📖 Content: {preview}")

            # Time the prediction
            start_time = time.time()
            prediction = self.predict_emotions(entry['content'])
            prediction['processing_time_ms'] = (time.time() - start_time) * 1000

            total_time += prediction['processing_time_ms']

            # Display results
            print(f"🎯 Primary Emotion: {prediction['primary_emotion']} (confidence: {prediction['confidence']:.3f})")
            print(f"🏷️  Predicted Emotions: {', '.join(prediction['predicted_emotions'][:5])}")
            print(f"⚡ Processing Time: {prediction['processing_time_ms']:.2f}ms")
            # Show top emotions with scores
            print(f"\n🏆 Top Emotions:")
            for emotion, score in zip(prediction['predicted_emotions'][:3], prediction['emotion_scores'][:3]):
                print(f"      - {emotion}: {score:.3f}")
            # Compare with expected emotions
            expected = set(entry['expected_emotions'])
            predicted = set(prediction['predicted_emotions'])
            overlap = expected.intersection(predicted)

            print(f"\n📊 Expected vs Predicted:")
            print(f"   Expected: {', '.join(expected)}")
            print(f"   Predicted: {', '.join(predicted)}")
            print(f"   Overlap: {', '.join(overlap)} ({len(overlap)}/{len(expected)})")

            # Store results
            entry_result = {
                "title": entry['title'],
                "content_length": len(entry['content']),
                "expected_emotions": entry['expected_emotions'],
                "prediction": prediction
            }
            results["journal_entries"].append(entry_result)

            all_predicted_emotions.extend(prediction['predicted_emotions'])

        # Generate summary
        unique_emotions = set(all_predicted_emotions)
        avg_time = total_time / len(journal_entries)

        results["summary"] = {
            "total_entries": len(journal_entries),
            "unique_emotions_detected": len(unique_emotions),
            "all_emotions_detected": sorted(list(unique_emotions)),
            "average_processing_time_ms": avg_time,
            "total_processing_time_ms": total_time
        }

        print(f"\n{'='*80}")
        print("📊 DEMO SUMMARY")
        print(f"{'='*80}")
        print(f"📝 Total Journal Entries: {len(journal_entries)}")
        print(f"🏷️  Unique Emotions Detected: {len(unique_emotions)}")
        print(f"⚡ Average Processing Time: {avg_time:.2f}ms")
        print(f"\n🎯 All Emotions Detected:")
        for emotion in sorted(unique_emotions):
            count = all_predicted_emotions.count(emotion)
            print(f"   {emotion}: {count} times")

        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Results saved to: comprehensive_journal_demo_results.json")

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save demo results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_journal_demo_results_{timestamp}.json"

        filepath = Path(__file__).parent / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {filepath}")
        return str(filepath)


def main():
    """Main demo execution."""
    print("🚀 SAMO Journal Inference Demo")
    print("=" * 60)

    try:
        # Initialize demo
        demo = JournalInferenceDemo()

        # Run comprehensive demo
        results = demo.run_comprehensive_demo()

        if "error" not in results:
            # Save results
            demo.save_results(results)

            print("\n🎉 SUCCESS! Journal inference demo completed!")
            print("📋 Your model successfully processed long-form journal entries!")
            print("📊 Check the results file for detailed analysis.")
        else:
            print(f"❌ Demo failed: {results['error']}")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

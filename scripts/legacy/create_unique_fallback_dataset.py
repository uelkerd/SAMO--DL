#!/usr/bin/env python3
"""
🚀 CREATE UNIQUE FALLBACK DATASET
=================================
Generate a diverse, unique fallback dataset with NO DUPLICATES.
Each sample must be completely different from all others.
"""

import json
import random


def create_unique_fallback_dataset():
    """Create a unique fallback dataset with no duplicates."""

    # Define unique templates for each emotion with variations
    emotion_templates = {
        "happy": [
            "I'm feeling really happy today! Everything is going well.",
            "I'm so happy with how things turned out.",
            "I feel joyful and content right now.",
            "I'm thrilled about the good news I received.",
            "I'm feeling wonderful and optimistic.",
            "I'm happy with my progress so far.",
            "I feel great about the positive changes.",
            "I'm delighted with the outcome.",
            "I'm feeling cheerful and upbeat.",
            "I'm happy about the opportunities ahead.",
            "I feel blessed and grateful for today.",
            "I'm excited and happy about the future.",
        ],
        "frustrated": [
            "I'm so frustrated with this project. Nothing is working.",
            "I'm getting really annoyed with these constant issues.",
            "I feel irritated by the lack of progress.",
            "I'm frustrated with the repeated failures.",
            "I'm annoyed by the constant setbacks.",
            "I feel exasperated with this situation.",
            "I'm getting tired of these problems.",
            "I'm frustrated with the slow progress.",
            "I feel irritated by the lack of support.",
            "I'm annoyed with the constant delays.",
            "I'm frustrated with the unclear instructions.",
            "I feel exasperated with these obstacles.",
        ],
        "anxious": [
            "I feel anxious about the upcoming presentation.",
            "I'm worried about the meeting tomorrow.",
            "I feel nervous about the interview.",
            "I'm anxious about the test results.",
            "I feel uneasy about the decision.",
            "I'm concerned about the deadline.",
            "I feel stressed about the workload.",
            "I'm anxious about the unknown outcome.",
            "I feel worried about the future.",
            "I'm nervous about the performance review.",
            "I feel uneasy about the changes.",
            "I'm anxious about the responsibilities.",
        ],
        "grateful": [
            "I'm grateful for all the support I've received.",
            "I feel thankful for the opportunities given to me.",
            "I'm grateful for the help from my friends.",
            "I feel blessed for the good things in my life.",
            "I'm thankful for the positive experiences.",
            "I feel grateful for the lessons learned.",
            "I'm thankful for the second chances.",
            "I feel blessed for the guidance received.",
            "I'm grateful for the kindness shown to me.",
            "I feel thankful for the understanding.",
            "I'm grateful for the patience of others.",
            "I feel blessed for the love and support.",
        ],
        "overwhelmed": [
            "I'm feeling overwhelmed with all these tasks.",
            "I feel swamped with the amount of work.",
            "I'm overwhelmed by the responsibilities.",
            "I feel buried under all these deadlines.",
            "I'm overwhelmed by the complexity of this.",
            "I feel swamped with the information.",
            "I'm overwhelmed by the expectations.",
            "I feel buried under the pressure.",
            "I'm overwhelmed by the changes.",
            "I feel swamped with the demands.",
            "I'm overwhelmed by the uncertainty.",
            "I feel buried under the workload.",
        ],
        "proud": [
            "I'm proud of what I've accomplished so far.",
            "I feel proud of my achievements.",
            "I'm proud of how far I've come.",
            "I feel proud of the progress made.",
            "I'm proud of the work I've done.",
            "I feel proud of my growth.",
            "I'm proud of the impact I've made.",
            "I feel proud of my contributions.",
            "I'm proud of the skills I've developed.",
            "I feel proud of my resilience.",
            "I'm proud of the challenges I've overcome.",
            "I feel proud of my determination.",
        ],
        "sad": [
            "I'm feeling sad and lonely today.",
            "I feel down about the recent events.",
            "I'm sad about the loss I experienced.",
            "I feel melancholy about the situation.",
            "I'm saddened by the disappointing news.",
            "I feel blue about the outcome.",
            "I'm sad about the missed opportunities.",
            "I feel downhearted about the results.",
            "I'm saddened by the lack of progress.",
            "I feel melancholy about the changes.",
            "I'm sad about the broken promises.",
            "I feel down about the setbacks.",
        ],
        "excited": [
            "I'm excited about the new opportunities ahead.",
            "I feel thrilled about the upcoming adventure.",
            "I'm excited about the possibilities.",
            "I feel enthusiastic about the future.",
            "I'm excited about the new project.",
            "I feel thrilled about the changes.",
            "I'm excited about the learning opportunities.",
            "I feel enthusiastic about the challenges.",
            "I'm excited about the potential outcomes.",
            "I feel thrilled about the new experiences.",
            "I'm excited about the growth opportunities.",
            "I feel enthusiastic about the journey ahead.",
        ],
        "calm": [
            "I feel calm and peaceful right now.",
            "I'm feeling serene and relaxed.",
            "I feel tranquil about the situation.",
            "I'm calm about the current state of things.",
            "I feel peaceful and content.",
            "I'm feeling relaxed and at ease.",
            "I feel serene about the outcome.",
            "I'm calm about the decisions made.",
            "I feel tranquil and centered.",
            "I'm feeling peaceful and balanced.",
            "I feel calm about the future.",
            "I'm serene about the present moment.",
        ],
        "hopeful": [
            "I'm hopeful that things will get better.",
            "I feel optimistic about the future.",
            "I'm hopeful about the possibilities ahead.",
            "I feel optimistic about the changes.",
            "I'm hopeful that the situation will improve.",
            "I feel optimistic about the outcomes.",
            "I'm hopeful about the new opportunities.",
            "I feel optimistic about the progress.",
            "I'm hopeful that we'll find solutions.",
            "I feel optimistic about the results.",
            "I'm hopeful about the positive changes.",
            "I feel optimistic about the journey ahead.",
        ],
        "tired": [
            "I'm tired and need some rest.",
            "I feel exhausted from the long day.",
            "I'm tired of dealing with these issues.",
            "I feel worn out from the stress.",
            "I'm tired of the constant challenges.",
            "I feel exhausted from the workload.",
            "I'm tired of the repetitive tasks.",
            "I feel worn out from the pressure.",
            "I'm tired of the ongoing problems.",
            "I feel exhausted from the demands.",
            "I'm tired of the uncertainty.",
            "I feel worn out from the responsibilities.",
        ],
        "content": [
            "I'm content with how things are going.",
            "I feel satisfied with the current situation.",
            "I'm content with my progress.",
            "I feel satisfied with the results.",
            "I'm content with the decisions made.",
            "I feel satisfied with the outcomes.",
            "I'm content with the current state.",
            "I feel satisfied with the work done.",
            "I'm content with the direction.",
            "I feel satisfied with the achievements.",
            "I'm content with the balance in my life.",
            "I feel satisfied with the growth experienced.",
        ],
    }

    # Create unique samples
    unique_samples = []

    for emotion, templates in emotion_templates.items():
        for i, template in enumerate(templates):
            unique_samples.append(
                {"text": template, "emotion": emotion, "sample_id": f"{emotion}_{i+1}"}
            )

    # Shuffle the samples for better training
    random.shuffle(unique_samples)

    print(f"✅ Created {len(unique_samples)} UNIQUE samples")
    print(f"📊 Samples per emotion: {len(unique_samples) // 12}")

    # Verify no duplicates
    texts = [sample["text"] for sample in unique_samples]
    unique_texts = set(texts)
    print(f"🔍 Duplicate check: {len(texts)} total, {len(unique_texts)} unique")

    if len(texts) != len(unique_texts):
        print("❌ WARNING: DUPLICATES FOUND!")
        return None

    print("✅ All samples are unique!")

    # Save the dataset
    with open("data/unique_fallback_dataset.json", "w") as f:
        json.dump(unique_samples, f, indent=2)

    print("💾 Saved unique fallback dataset to data/unique_fallback_dataset.json")

    # Show emotion distribution
    emotion_counts = {}
    for sample in unique_samples:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("\n📊 Emotion Distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} samples")

    return unique_samples


if __name__ == "__main__":
    print("🚀 CREATE UNIQUE FALLBACK DATASET")
    print("=" * 40)
    create_unique_fallback_dataset()
    print("\n🎉 Unique fallback dataset created successfully!")

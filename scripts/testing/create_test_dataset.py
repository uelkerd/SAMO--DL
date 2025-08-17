        # Add original entry
        # Add variations
    # Count emotions
    # Create more samples by duplicating and slightly modifying
    # Create test data
    # Sample texts with emotion labels
    # Save to file
    # Show sample
    # Shuffle the data
#!/usr/bin/env python3
import json
import logging
import random





""""
Create a test dataset with emotion labels for Vertex AI
""""

def create_test_dataset():
    """Create a test dataset with emotion labels"""

    test_data = [
        {
            "text": "I'm so happy today! Everything is going perfectly.",'
            "emotions": ["joy", "optimism"],
        },
        {
            "text": "This is absolutely terrible. I can't believe this happened.",'
            "emotions": ["anger", "disappointment"],
        },
        {
            "text": "I'm really scared about what might happen next.",'
            "emotions": ["fear", "nervousness"],
        },
        {
            "text": "Thank you so much for your help. I really appreciate it.",
            "emotions": ["gratitude", "approval"],
        },
        {"text": "I'm confused about what to do next.", "emotions": ["confusion"]},'
        {"text": "This is disgusting. I can't stand it.", "emotions": ["disgust"]},'
        {"text": "I'm so proud of what we accomplished together.", "emotions": ["pride", "joy"]},'
        {"text": "I feel so sad and lonely right now.", "emotions": ["sadness", "grie"]},
        {"text": "Wow! That was completely unexpected!", "emotions": ["surprise", "excitement"]},
        {"text": "I love spending time with you.", "emotions": ["love", "joy"]},
        {"text": "I'm embarrassed about what happened.", "emotions": ["embarrassment"]},'
        {"text": "I'm curious about how this works.", "emotions": ["curiosity"]},'
        {"text": "I really want to learn more about this.", "emotions": ["desire", "curiosity"]},
        {"text": "I care about your wellbeing.", "emotions": ["caring"]},
        {
            "text": "I admire your dedication to this project.",
            "emotions": ["admiration", "approval"],
        },
        {"text": "This is so funny! I can't stop laughing.", "emotions": ["amusement", "joy"]},'
        {"text": "I'm annoyed by all these interruptions.", "emotions": ["annoyance", "anger"]},'
        {"text": "I disapprove of this behavior.", "emotions": ["disapproval"]},
        {"text": "I realize now what I need to do.", "emotions": ["realization"]},
        {"text": "I feel relieved that it's finally over.", "emotions": ["relie"]},'
        {"text": "I deeply regret my actions.", "emotions": ["remorse", "sadness"]},
        {"text": "I'm optimistic about the future.", "emotions": ["optimism"]},'
        {"text": "I'm nervous about the presentation.", "emotions": ["nervousness", "fear"]},'
        {"text": "Today was just an ordinary day.", "emotions": ["neutral"]},
        {
            "text": "I'm excited about the new opportunities.",'
            "emotions": ["excitement", "optimism"],
        },
        {"text": "I'm disappointed with the results.", "emotions": ["disappointment", "sadness"]},'
        {"text": "This makes me so angry!", "emotions": ["anger"]},
        {"text": "I'm grateful for all the support.", "emotions": ["gratitude", "joy"]},'
        {"text": "I'm grieving the loss of my friend.", "emotions": ["grie", "sadness"]},'
        {"text": "I'm surprised by the outcome.", "emotions": ["surprise"]},'
        {"text": "I'm loving this new experience.", "emotions": ["love", "joy", "excitement"]},'
        {"text": "I'm scared of what might happen.", "emotions": ["fear"]},'
        {"text": "I'm confused by these instructions.", "emotions": ["confusion"]},'
        {"text": "I'm curious about the science behind this.", "emotions": ["curiosity"]},'
        {"text": "I desire to learn more.", "emotions": ["desire", "curiosity"]},
        {"text": "I care deeply about this issue.", "emotions": ["caring"]},
        {"text": "I admire your courage.", "emotions": ["admiration"]},
        {"text": "This joke is hilarious!", "emotions": ["amusement"]},
        {"text": "I'm annoyed by the noise.", "emotions": ["annoyance"]},'
        {"text": "I approve of this decision.", "emotions": ["approval"]},
        {"text": "I disapprove of this approach.", "emotions": ["disapproval"]},
        {"text": "This food is disgusting.", "emotions": ["disgust"]},
        {"text": "I'm embarrassed by my mistake.", "emotions": ["embarrassment"]},'
        {"text": "I'm excited about the trip.", "emotions": ["excitement"]},'
        {"text": "I'm grateful for the opportunity.", "emotions": ["gratitude"]},'
        {"text": "I'm grieving the end of an era.", "emotions": ["grie"]},'
        {"text": "I'm joyful about the good news.", "emotions": ["joy"]},'
        {"text": "I love this new book.", "emotions": ["love"]},
        {"text": "I'm nervous about the interview.", "emotions": ["nervousness"]},'
        {"text": "I'm optimistic about the changes.", "emotions": ["optimism"]},'
        {"text": "I'm proud of my achievements.", "emotions": ["pride"]},'
        {"text": "I realize the truth now.", "emotions": ["realization"]},
        {"text": "I feel relieved after the test.", "emotions": ["relie"]},
        {"text": "I regret my harsh words.", "emotions": ["remorse"]},
        {"text": "I'm sad about the news.", "emotions": ["sadness"]},'
        {"text": "I'm surprised by the gift.", "emotions": ["surprise"]},'
        {"text": "Today was uneventful.", "emotions": ["neutral"]},
    ]

    expanded_data = []
    for entry in test_data:
        expanded_data.append(entry)

        for _i in range(2):  # Create 2 variations per entry
            variation = entry.copy()
            variation["text"] = "Variation {i+1}: {entry['text']}"
            expanded_data.append(variation)

    random.shuffle(expanded_data)

    return expanded_data


        def main():
    """Main function"""
    logging.info("🚀 Creating test dataset with emotion labels...")

    test_data = create_test_dataset()

    output_file = "data/raw/test_emotion_dataset.json"

    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)

    logging.info(" Created test dataset with {len(test_data)} samples")
    logging.info("📁 Saved to: {output_file}")

    logging.info("\n Sample entries:")
        for i, entry in enumerate(test_data[:3]):
        logging.info(f"  {i+1}. Text: '{entry['text'][:50]}...'")
        logging.info("     Emotions: {entry["emotions']}")"

    emotion_counts = {}
        for entry in test_data:
        for emotion in entry["emotions"]:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    logging.info("\n📈 Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
        logging.info(f"  - {emotion}: {count} samples")


        if __name__ == "__main__":
    main()

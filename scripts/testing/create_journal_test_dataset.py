#!/usr/bin/env python3
"""
Create Journal Entry Test Dataset for Domain Adaptation

This script generates a realistic test dataset of journal entries for domain adaptation
testing as required by REQ-DL-012. The dataset will be used to validate that our
emotion detection model performs well on journal-style text (personal, reflective,
longer-form) rather than just Reddit comments.

Target: 100+ journal entries with realistic emotional content
Success Metric: 70% F1 score on this test set
"""

import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Realistic journal entry templates that reflect personal, reflective writing
JOURNAL_TEMPLATES = [
    # Personal reflection templates
    "Today I found myself thinking deeply about {topic}. {emotion_context} {reflection}",
    "I've been struggling with {topic} lately. {emotion_context} {reflection}",
    "This week has been challenging when it comes to {topic}. {emotion_context} {reflection}",
    "I'm feeling {emotion} about {topic}. {emotion_context} {reflection}",
    "My thoughts on {topic} have been consuming me. {emotion_context} {reflection}",
    "I had a breakthrough moment with {topic} today. {emotion_context} {reflection}",
    "I'm trying to understand why {topic} affects me so deeply. {emotion_context} {reflection}",
    "Looking back on my relationship with {topic}, I realize {emotion_context} {reflection}",
    "I've been avoiding thinking about {topic}, but today I couldn't ignore it. {emotion_context} {reflection}",
    "My journey with {topic} has taught me so much. {emotion_context} {reflection}",
]

# Realistic topics that people actually journal about
JOURNAL_TOPICS = [
    "my relationship with my family",
    "work stress and burnout",
    "my health journey",
    "personal growth and self-improvement",
    "my creative projects",
    "financial worries",
    "my social life and friendships",
    "my spiritual journey",
    "my career goals",
    "my mental health",
    "my relationship with food",
    "my sleep patterns",
    "my exercise routine",
    "my relationship with technology",
    "my environmental impact",
    "my learning goals",
    "my relationship with money",
    "my sense of purpose",
    "my boundaries with others",
    "my relationship with myself",
]

# Emotion contexts that provide realistic emotional depth
EMOTION_CONTEXTS = {
    "happy": [
        "I feel a genuine sense of joy and contentment.",
        "There's this lightness in my chest that I haven't felt in a while.",
        "I'm genuinely excited about the possibilities ahead.",
        "I feel grateful for this moment of clarity.",
        "There's a warmth spreading through me that I want to hold onto.",
    ],
    "sad": [
        "I feel a heaviness that's hard to shake.",
        "There's this emptiness that I can't seem to fill.",
        "I'm feeling really down and I'm not sure why.",
        "The sadness feels like it's sitting in my chest.",
        "I miss something I can't quite name.",
    ],
    "anxious": [
        "My mind keeps racing with worst-case scenarios.",
        "I feel like I'm constantly on edge.",
        "There's this knot in my stomach that won't go away.",
        "I'm worried about things I can't control.",
        "My thoughts keep spiraling into negative territory.",
    ],
    "excited": [
        "I can barely contain my enthusiasm.",
        "There's this energy bubbling up inside me.",
        "I feel like I'm on the verge of something amazing.",
        "My heart is racing with anticipation.",
        "I'm practically bouncing with excitement.",
    ],
    "calm": [
        "I feel centered and at peace.",
        "There's a quiet confidence within me.",
        "I feel grounded and present.",
        "My mind feels clear and focused.",
        "I feel like I'm exactly where I need to be.",
    ],
    "frustrated": [
        "I'm hitting wall after wall and it's exhausting.",
        "Nothing seems to be working out the way I planned.",
        "I feel like I'm constantly fighting an uphill battle.",
        "My patience is wearing thin.",
        "I'm tired of things not going my way.",
    ],
    "hopeful": [
        "I can see a light at the end of the tunnel.",
        "I feel optimistic about what's coming.",
        "There's this sense that things are going to get better.",
        "I believe in the possibility of positive change.",
        "I feel like I'm moving in the right direction.",
    ],
    "tired": [
        "I feel drained in a way that sleep can't fix.",
        "My energy levels are at an all-time low.",
        "I'm exhausted from trying so hard.",
        "I feel like I'm running on empty.",
        "My body and mind are begging for rest.",
    ],
    "grateful": [
        "I'm overwhelmed by how much I have to be thankful for.",
        "I feel blessed beyond measure.",
        "My heart is full of appreciation.",
        "I'm reminded of how lucky I am.",
        "I feel like the universe has been kind to me.",
    ],
    "overwhelmed": [
        "I feel like I'm drowning in responsibilities.",
        "Everything feels like too much right now.",
        "I'm struggling to keep my head above water.",
        "I feel like I'm being pulled in too many directions.",
        "The weight of everything is crushing me.",
    ],
    "proud": [
        "I feel a deep sense of accomplishment.",
        "I'm proud of how far I've come.",
        "I feel like I'm finally getting it right.",
        "I'm impressed with my own resilience.",
        "I feel like I'm becoming the person I want to be.",
    ],
    "content": [
        "I feel satisfied with where I am right now.",
        "There's a quiet happiness in my heart.",
        "I feel like I have everything I need.",
        "I'm at peace with my current situation.",
        "I feel complete and whole.",
    ],
}

# Reflective statements that add depth and personal insight
REFLECTIVE_STATEMENTS = [
    "I'm starting to understand that this is all part of my journey.",
    "Maybe this is exactly what I needed to learn right now.",
    "I'm realizing that I have more control than I thought.",
    "This experience is teaching me something important about myself.",
    "I think I'm finally ready to make some changes.",
    "Looking back, I can see how far I've come.",
    "I'm beginning to see patterns in my behavior that I want to change.",
    "This feels like a turning point in my life.",
    "I'm learning to be kinder to myself through this process.",
    "I think this is helping me grow in ways I didn't expect.",
    "I'm starting to trust my instincts more.",
    "This is showing me what I'm truly capable of.",
    "I'm realizing that I don't have to have all the answers.",
    "This journey is revealing parts of myself I didn't know existed.",
    "I'm learning to embrace uncertainty.",
]

def generate_journal_content(topic: str, emotion: str) -> str:
    """Generate realistic journal entry content."""
    template = random.choice(JOURNAL_TEMPLATES)
    emotion_context = random.choice(EMOTION_CONTEXTS.get(emotion, ["I'm feeling this way."]))
    reflection = random.choice(REFLECTIVE_STATEMENTS)
    
    content = template.format(
        topic=topic,
        emotion=emotion,
        emotion_context=emotion_context,
        reflection=reflection
    )
    
    # Add more depth with additional sentences
    if random.random() > 0.3:  # 70% chance of adding more detail
        additional_context = random.choice(EMOTION_CONTEXTS.get(emotion, ["I'm processing this."]))
        content += f" {additional_context}"
    
    if random.random() > 0.5:  # 50% chance of adding another reflection
        second_reflection = random.choice(REFLECTIVE_STATEMENTS)
        content += f" {second_reflection}"
    
    return content

def generate_journal_entry(entry_id: int, user_id: int, created_at: datetime) -> Dict[str, Any]:
    """Generate a single realistic journal entry."""
    topic = random.choice(JOURNAL_TOPICS)
    emotion = random.choice(list(EMOTION_CONTEXTS.keys()))
    
    return {
        "id": entry_id,
        "user_id": user_id,
        "title": f"Journal Entry {entry_id}",
        "content": generate_journal_content(topic, emotion),
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "is_private": True,
        "topic": topic,
        "emotion": emotion,
        "entry_type": "journal",  # Distinguish from Reddit-style content
        "word_count": len(generate_journal_content(topic, emotion).split()),
    }

def create_journal_test_dataset(
    num_entries: int = 150,
    num_users: int = 10,
    days_back: int = 90
) -> List[Dict[str, Any]]:
    """Create a comprehensive journal test dataset."""
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    datetime.now(timezone.utc)
    
    entries = []
    for i in range(num_entries):
        user_id = random.randint(1, num_users)
        
        # Random date within the range
        days_offset = random.randint(0, days_back)
        entry_date = start_date + timedelta(days=days_offset)
        
        # Random time during the day (more realistic for journaling)
        entry_date = entry_date.replace(
            hour=random.randint(6, 23),  # Early morning to late night
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )
        
        entry = generate_journal_entry(i + 1, user_id, entry_date)
        entries.append(entry)
    
    return entries

def save_test_dataset(entries: List[Dict[str, Any]], output_path: str) -> None:
    """Save the test dataset to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=2)
    
    print(f"✅ Saved {len(entries)} journal entries to {output_path}")

def create_dataset_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of the dataset for validation."""
    df = pd.DataFrame(entries)
    
    summary = {
        "total_entries": len(entries),
        "unique_users": df["user_id"].nunique(),
        "emotion_distribution": df["emotion"].value_counts().to_dict(),
        "topic_distribution": df["topic"].value_counts().to_dict(),
        "avg_word_count": df["word_count"].mean(),
        "date_range": {
            "start": min(df["created_at"]),
            "end": max(df["created_at"])
        },
        "sample_entries": entries[:3]  # First 3 entries as examples
    }
    
    return summary

def main():
    """Main function to create the journal test dataset."""
    print("🚀 Creating Journal Entry Test Dataset for Domain Adaptation")
    print("=" * 60)
    
    # Create the dataset
    entries = create_journal_test_dataset(
        num_entries=150,  # Exceeds the 100+ requirement
        num_users=10,
        days_back=90
    )
    
    # Save to data directory
    output_path = "data/journal_test_dataset.json"
    save_test_dataset(entries, output_path)
    
    # Create and save summary
    summary = create_dataset_summary(entries)
    summary_path = "data/journal_test_dataset_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved dataset summary to {summary_path}")
    
    # Print key statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total Entries: {summary['total_entries']}")
    print(f"   Unique Users: {summary['unique_users']}")
    print(f"   Average Word Count: {summary['avg_word_count']:.1f}")
    print(f"   Date Range: {summary['date_range']['start'][:10]} to {summary['date_range']['end'][:10]}")
    
    print("\n🎯 Emotion Distribution:")
    for emotion, count in summary['emotion_distribution'].items():
        percentage = (count / summary['total_entries']) * 100
        print(f"   {emotion}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Journal Test Dataset Created Successfully!")
    print("   This dataset will be used for REQ-DL-012 domain adaptation testing")
    print("   Target: 70% F1 score on journal-style text vs Reddit comments")

if __name__ == "__main__":
    main()

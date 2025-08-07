        # Add hour/minute/second for more realistic timestamps
        # Create the entry
        # Generate a random date within the range
        # Randomly select user_id
    # Convert datetime objects to strings for JSON serialization
    # Convert string dates back to datetime
    # Ensure output directory exists
    # Generate 100 entries from 5 users over the past 60 days
    # Save to data/raw directory
# Additional sentences to add variety
# Emotion categories for entries
# Sample topics to generate journal entries about
# Templates for journal entry content
# Title templates
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
import json
import pandas as pd
import random




TOPICS = [
    "work",
    "family",
    "health",
    "exercise",
    "food",
    "travel",
    "learning",
    "hobbies",
    "goals",
    "emotions",
    "relationships",
    "finance",
    "home",
    "pets",
    "nature",
    "dreams",
    "reflection",
]

EMOTIONS = [
    "happy",
    "sad",
    "anxious",
    "excited",
    "calm",
    "frustrated",
    "hopeful",
    "tired",
    "grateful",
    "overwhelmed",
    "proud",
    "content",
]

ENTRY_TEMPLATES = [
    "Today I felt {emotion} about {topic}. {additional_sentence}",
    "I spent time on {topic} today. {additional_sentence} Overall I'm feeling {emotion}.",
    "I've been thinking a lot about {topic} lately. {additional_sentence} It makes me feel {emotion}.",
    "My {topic} journey continues. {additional_sentence} I'm {emotion} about my progress.",
    "{topic} has been on my mind. {additional_sentence} I'm feeling {emotion} about it.",
    "I had an experience with {topic} today that left me feeling {emotion}. {additional_sentence}",
    "I'm {emotion} about my {topic} situation. {additional_sentence}",
    "When it comes to {topic}, I'm feeling {emotion}. {additional_sentence}",
    "My thoughts on {topic} today: {additional_sentence} I feel {emotion}.",
    "Today's {topic} activities made me feel {emotion}. {additional_sentence}",
]

REFLECTION_TEMPLATES = [
    "It really makes me wonder about what's next.",
    "I need to spend more time thinking about why I feel this way.",
    "This whole experience has taught me something important about myself.",
    "Looking back, I can see a clear pattern emerging here.",
    "I'm not sure what the right move is, but I know I need to do something.",
    "It's a powerful reminder of what's truly important to me.",
]

DETAIL_TEMPLATES = [
    "The main reason for this is the pressure from the upcoming project deadline.",
    "It all started after that conversation with my manager earlier this week.",
    "I've been trying to balance this with all of my other responsibilities, and it's tough.",
    "The small details of the situation are what seem to be causing the most stress.",
    "I'm trying to focus on the positive aspects, but it's proving to be difficult.",
    "The situation with {topic} has been evolving for a few weeks now, and it's coming to a head.",
]


ADDITIONAL_SENTENCES = [
    "I'm hoping things will improve soon.",
    "I'm trying to maintain a positive outlook.",
    "I need to focus more on this area.",
    "I'm making good progress.",
    "I'm still working through some challenges.",
    "It's been a journey with ups and downs.",
    "I've noticed some interesting patterns.",
    "I want to explore this further.",
    "This has been a priority for me lately.",
    "I'm learning new things every day.",
    "I'm trying different approaches to see what works best.",
    "It's important for me to reflect on this regularly.",
    "I've been discussing this with friends.",
    "I'm researching new strategies.",
    "This has taken more time than expected.",
    "The results have been surprising.",
    "I need to find more balance here.",
    "I'm proud of what I've accomplished so far.",
    "There's still much to learn and discover.",
    "I'm being patient with the process.",
]

TITLE_TEMPLATES = [
    "Thoughts on {topic}",
    "My {topic} journey",
    "Reflecting on {topic}",
    "Today's {topic} experience",
    "{topic} insights",
    "Exploring my {topic}",
    "Notes on {topic}",
    "{topic} reflections",
    "{topic} diary entry",
    "Processing my {topic} feelings",
    "{topic} update",
    "{emotion} about {topic}",
    "{topic} progress",
    "{topic} challenges and wins",
    "My relationship with {topic}",
]


def generate_title(topic: str, emotion: str) -> str:
    """Generate a journal entry title."""
    template = random.choice(TITLE_TEMPLATES)
    return template.format(topic=topic, emotion=emotion)


def generate_content(topic: str, emotion: str) -> str:
    """Generate journal entry content."""
    template = random.choice(ENTRY_TEMPLATES)
    base_sentence = random.choice(ADDITIONAL_SENTENCES)
    content = template.format(topic=topic, emotion=emotion, additional_sentence=base_sentence)

    # Add more complexity with a chance of a second or third sentence
    if random.random() > 0.4:  # 60% chance of adding more detail
        content += f" {random.choice(DETAIL_TEMPLATES).format(topic=topic)}"
    if random.random() > 0.6:  # 40% chance of adding a reflection
        content += f" {random.choice(REFLECTION_TEMPLATES)}"
    return content

def generate_entry(user_id: int, created_at: datetime, id_start: int = 1) -> Dict[str, Any]:
    """Generate a single journal entry."""
    topic = random.choice(TOPICS)
    emotion = random.choice(EMOTIONS)

    return {
        "id": id_start,
        "user_id": user_id,
        "title": generate_title(topic, emotion),
        "content": generate_content(topic, emotion),
        "created_at": created_at,
        "updated_at": created_at,
        "is_private": random.choice([True, False]),
        "topic": topic,  # Additional metadata for testing
        "emotion": emotion,  # Additional metadata for testing
    }


def generate_entries(
    num_entries: int = 100,
    num_users: int = 5,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Generate a list of synthetic journal entries.

    Args:
        num_entries: Number of entries to generate
        num_users: Number of unique users to create entries for
        start_date: Start date for entries (defaults to 60 days ago)
        end_date: End date for entries (defaults to today)

    Returns:
        List of dictionaries containing journal entries

    """
    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now(timezone.utc)

    date_range = (end_date - start_date).days
    entries = []

    for i in range(num_entries):
        user_id = random.randint(1, num_users)

        days_offset = random.randint(0, date_range)
        entry_date = start_date + timedelta(days=days_offset)

        entry_date = entry_date.replace(
            hour=random.randint(7, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

        entry = generate_entry(user_id, entry_date, id_start=i + 1)
        entries.append(entry)

    return entries


def save_entries_to_json(entries: List[Dict[str, Any]], output_path: str) -> None:
    """Save generated entries to a JSON file.

    Args:
        entries: List of entry dictionaries
        output_path: Path to save the JSON file

    """
    Path(Path(output_path).parent).mkdir(parents=True, exist_ok=True)

    serializable_entries = []
    for entry in entries:
        serializable_entry = entry.copy()
        serializable_entry["created_at"] = entry["created_at"].isoformat()
        serializable_entry["updated_at"] = entry["updated_at"].isoformat()
        serializable_entries.append(serializable_entry)

    with Path(output_path).open("w") as f:
        json.dump(serializable_entries, f, indent=2)


def load_sample_entries(json_path: str) -> pd.DataFrame:
    """Load sample entries from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        DataFrame containing the entries

    """
    with open(json_path) as f:
        entries = json.load(f)

    df = pd.DataFrame(entries)

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])

    return df


if __name__ == "__main__":
    entries = generate_entries(num_entries=100, num_users=5)

    output_dir = Path(
        Path(__file__).parent.parent.parent,
        "data",
        "raw",
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir, "sample_journal_entries.json").as_posix()

    save_entries_to_json(entries, output_path)

#!/usr/bin/env python3
"""
Emotion Labels for SAMO-DL Emotion Detection

This module defines the emotion categories and labels used by the
SAMO emotion detection system, based on the GoEmotions dataset.

The GoEmotions dataset includes 27 emotion categories plus neutral,
providing comprehensive coverage of emotional states for journal analysis.
"""

from typing import List, Dict, Tuple

# GoEmotions emotion categories (27 emotions + neutral = 28 total)
GOEMOTIONS_EMOTIONS = [
    "admiration",      # 0
    "amusement",       # 1
    "anger",          # 2
    "annoyance",      # 3
    "approval",       # 4
    "caring",         # 5
    "confusion",      # 6
    "curiosity",      # 7
    "desire",         # 8
    "disappointment", # 9
    "disapproval",    # 10
    "disgust",        # 11
    "embarrassment",  # 12
    "excitement",     # 13
    "fear",           # 14
    "gratitude",      # 15
    "grief",          # 16
    "joy",            # 17
    "love",           # 18
    "nervousness",    # 19
    "optimism",       # 20
    "pride",          # 21
    "realization",    # 22
    "relief",         # 23
    "remorse",        # 24
    "sadness",        # 25
    "surprise",       # 26
    "neutral",        # 27
]

# Emotion categories grouped by valence (positive, negative, neutral)
EMOTION_VALENCE_GROUPS = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "curiosity", 
        "desire", "excitement", "gratitude", "joy", "love", 
        "optimism", "pride", "realization", "relief"
    ],
    "negative": [
        "anger", "annoyance", "confusion", "disappointment", "disapproval",
        "disgust", "embarrassment", "fear", "grief", "nervousness", 
        "remorse", "sadness"
    ],
    "neutral": [
        "neutral"
    ]
}

# Emotion categories grouped by arousal (high, medium, low)
EMOTION_AROUSAL_GROUPS = {
    "high": [
        "anger", "excitement", "fear", "joy", "nervousness", "surprise"
    ],
    "medium": [
        "amusement", "annoyance", "confusion", "curiosity", "desire",
        "disappointment", "disgust", "embarrassment", "gratitude", 
        "love", "optimism", "pride", "relief", "remorse", "sadness"
    ],
    "low": [
        "admiration", "approval", "caring", "grief", "realization", "neutral"
    ]
}

# Emotion categories grouped by dominance (high, medium, low)
EMOTION_DOMINANCE_GROUPS = {
    "high": [
        "anger", "approval", "disapproval", "pride", "realization"
    ],
    "medium": [
        "admiration", "amusement", "annoyance", "caring", "curiosity",
        "desire", "excitement", "gratitude", "joy", "love", "optimism", "relief"
    ],
    "low": [
        "confusion", "disappointment", "disgust", "embarrassment", "fear",
        "grief", "nervousness", "remorse", "sadness", "surprise", "neutral"
    ]
}

# Emotion intensity levels (for future enhancement)
EMOTION_INTENSITY_LEVELS = {
    "very_low": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "very_high": 1.0
}

# Emotion descriptions for better understanding
EMOTION_DESCRIPTIONS = {
    "admiration": "A feeling of respect and approval for someone or something",
    "amusement": "A feeling of being entertained or finding something funny",
    "anger": "A strong feeling of displeasure and hostility",
    "annoyance": "A feeling of slight anger or irritation",
    "approval": "A feeling of agreement with or support for something",
    "caring": "A feeling of concern and kindness for others",
    "confusion": "A feeling of being puzzled or unclear about something",
    "curiosity": "A strong desire to know or learn something",
    "desire": "A strong feeling of wanting something",
    "disappointment": "A feeling of sadness because something didn't meet expectations",
    "disapproval": "A feeling of disagreement with or opposition to something",
    "disgust": "A strong feeling of revulsion or repugnance",
    "embarrassment": "A feeling of self-consciousness or shame",
    "excitement": "A feeling of great enthusiasm and eagerness",
    "fear": "An unpleasant emotion caused by the threat of danger or pain",
    "gratitude": "A feeling of thankfulness and appreciation",
    "grief": "Deep sorrow, especially caused by someone's death",
    "joy": "A feeling of great pleasure and happiness",
    "love": "An intense feeling of deep affection",
    "nervousness": "A feeling of anxiety or unease",
    "optimism": "A feeling of hopefulness and confidence about the future",
    "pride": "A feeling of satisfaction in one's achievements",
    "realization": "A moment of sudden understanding or awareness",
    "relief": "A feeling of reassurance and relaxation",
    "remorse": "A feeling of deep regret for a wrong committed",
    "sadness": "A feeling of sorrow and unhappiness",
    "surprise": "A feeling of astonishment or amazement",
    "neutral": "A state of being neither positive nor negative"
}

# Emotion synonyms for better text matching
EMOTION_SYNONYMS = {
    "admiration": ["respect", "esteem", "reverence", "veneration"],
    "amusement": ["entertainment", "fun", "delight", "merriment"],
    "anger": ["rage", "fury", "wrath", "irritation", "madness"],
    "annoyance": ["irritation", "bother", "vexation", "aggravation"],
    "approval": ["endorsement", "support", "agreement", "acceptance"],
    "caring": ["concern", "compassion", "empathy", "kindness"],
    "confusion": ["bewilderment", "perplexity", "puzzlement", "disorientation"],
    "curiosity": ["inquisitiveness", "interest", "wonder", "inquiry"],
    "desire": ["want", "wish", "longing", "yearning", "craving"],
    "disappointment": ["letdown", "dismay", "discouragement", "frustration"],
    "disapproval": ["disagreement", "opposition", "objection", "dissent"],
    "disgust": ["revulsion", "repugnance", "loathing", "abhorrence"],
    "embarrassment": ["shame", "humiliation", "self-consciousness", "awkwardness"],
    "excitement": ["enthusiasm", "eagerness", "anticipation", "thrill"],
    "fear": ["anxiety", "worry", "dread", "terror", "panic"],
    "gratitude": ["thankfulness", "appreciation", "recognition", "acknowledgment"],
    "grief": ["sorrow", "mourning", "anguish", "heartache"],
    "joy": ["happiness", "delight", "elation", "bliss", "cheerfulness"],
    "love": ["affection", "adoration", "fondness", "devotion"],
    "nervousness": ["anxiety", "unease", "tension", "apprehension"],
    "optimism": ["hopefulness", "confidence", "positivity", "cheerfulness"],
    "pride": ["satisfaction", "accomplishment", "achievement", "honor"],
    "realization": ["understanding", "awareness", "insight", "comprehension"],
    "relief": ["reassurance", "comfort", "ease", "relaxation"],
    "remorse": ["regret", "guilt", "penitence", "contrition"],
    "sadness": ["sorrow", "melancholy", "gloom", "despair", "unhappiness"],
    "surprise": ["astonishment", "amazement", "shock", "wonder"],
    "neutral": ["indifferent", "impartial", "unbiased", "objective"]
}


def get_emotion_index(emotion: str) -> int:
    """
    Get the index of an emotion in the GoEmotions list.

    Args:
        emotion: Emotion name

    Returns:
        Index of the emotion (0-27)

    Raises:
        ValueError: If emotion is not found
    """
    try:
        return GOEMOTIONS_EMOTIONS.index(emotion.lower())
    except ValueError as e:
        raise ValueError(f"Emotion '{emotion}' not found in GoEmotions list") from e


def get_emotion_name(index: int) -> str:
    """
    Get the emotion name from its index.

    Args:
        index: Emotion index (0-27)

    Returns:
        Emotion name

    Raises:
        IndexError: If index is out of range
    """
    if 0 <= index < len(GOEMOTIONS_EMOTIONS):
        return GOEMOTIONS_EMOTIONS[index]
    raise IndexError(f"Index {index} out of range for GoEmotions list")


def get_emotions_by_valence(valence: str) -> List[str]:
    """
    Get emotions by valence group.

    Args:
        valence: Valence group ('positive', 'negative', 'neutral')

    Returns:
        List of emotions in the valence group
    """
    return EMOTION_VALENCE_GROUPS.get(valence, [])


def get_emotions_by_arousal(arousal: str) -> List[str]:
    """
    Get emotions by arousal group.

    Args:
        arousal: Arousal group ('high', 'medium', 'low')

    Returns:
        List of emotions in the arousal group
    """
    return EMOTION_AROUSAL_GROUPS.get(arousal, [])


def get_emotions_by_dominance(dominance: str) -> List[str]:
    """
    Get emotions by dominance group.

    Args:
        dominance: Dominance group ('high', 'medium', 'low')

    Returns:
        List of emotions in the dominance group
    """
    return EMOTION_DOMINANCE_GROUPS.get(dominance, [])


def get_emotion_description(emotion: str) -> str:
    """
    Get description of an emotion.

    Args:
        emotion: Emotion name

    Returns:
        Description of the emotion
    """
    return EMOTION_DESCRIPTIONS.get(emotion.lower(), "No description available")


def get_emotion_synonyms(emotion: str) -> List[str]:
    """
    Get synonyms for an emotion.

    Args:
        emotion: Emotion name

    Returns:
        List of synonyms for the emotion
    """
    return EMOTION_SYNONYMS.get(emotion.lower(), [])


def get_all_emotions() -> List[str]:
    """
    Get all emotion names.

    Returns:
        List of all emotion names
    """
    return GOEMOTIONS_EMOTIONS.copy()


def get_emotion_count() -> int:
    """
    Get total number of emotions.

    Returns:
        Number of emotions (28)
    """
    return len(GOEMOTIONS_EMOTIONS)


def validate_emotion(emotion: str) -> bool:
    """
    Check if an emotion is valid.

    Args:
        emotion: Emotion name to validate

    Returns:
        True if emotion is valid, False otherwise
    """
    return emotion.lower() in GOEMOTIONS_EMOTIONS


def get_emotion_statistics() -> Dict[str, int]:
    """
    Get statistics about emotion categories.

    Returns:
        Dictionary with emotion statistics
    """
    return {
        "total_emotions": len(GOEMOTIONS_EMOTIONS),
        "positive_emotions": len(EMOTION_VALENCE_GROUPS["positive"]),
        "negative_emotions": len(EMOTION_VALENCE_GROUPS["negative"]),
        "neutral_emotions": len(EMOTION_VALENCE_GROUPS["neutral"]),
        "high_arousal_emotions": len(EMOTION_AROUSAL_GROUPS["high"]),
        "medium_arousal_emotions": len(EMOTION_AROUSAL_GROUPS["medium"]),
        "low_arousal_emotions": len(EMOTION_AROUSAL_GROUPS["low"]),
    }


if __name__ == "__main__":
    # Test the emotion labels module
    print("🧪 Testing Emotion Labels Module")
    print("=" * 50)

    print(f"Total emotions: {get_emotion_count()}")
    print(f"All emotions: {get_all_emotions()}")

    print(f"\nPositive emotions: {get_emotions_by_valence('positive')}")
    print(f"Negative emotions: {get_emotions_by_valence('negative')}")
    print(f"Neutral emotions: {get_emotions_by_valence('neutral')}")

    print(f"\nHigh arousal emotions: {get_emotions_by_arousal('high')}")
    print(f"Medium arousal emotions: {get_emotions_by_arousal('medium')}")
    print(f"Low arousal emotions: {get_emotions_by_arousal('low')}")

    print("\nEmotion descriptions:")
    for emotion_name in ["joy", "sadness", "anger", "fear"]:
        print(f"  {emotion_name}: {get_emotion_description(emotion_name)}")

    print(f"\nEmotion synonyms for 'joy': {get_emotion_synonyms('joy')}")
    print(f"Emotion synonyms for 'sadness': {get_emotion_synonyms('sadness')}")

    print(f"\nEmotion statistics: {get_emotion_statistics()}")

    print("\n✅ Emotion labels module test completed successfully!")
